"""
PoE-World creation synthesizer for the Crafter environment.

This module implements the expert synthesis algorithm that generates Python code
to explain observed object lifecycle events (creation, deletion, replacement)
in the Crafter environment.
"""

import ast
from typing import List, Optional, Protocol

from loguru import logger

from ..litellm_utils import GeminiLiteLlmParams, LiteLlmMessage, LiteLlmRequest
from ..local_code_execution import ExecWithLimitedNamespace
from ..typing_utils import implements
from .core import (
    ExpertFunction,
    ExpertSynthesizerProtocol,
    SymbolicTransition,
    WeightedExpert,
)
from typing import TypeVar, Generic, Callable
from ..litellm_utils import NonStreamingModelResponse
from .core import ExpertFunctionWrapper
import re
from typing import Optional

SymbolicStateT = TypeVar("SymbolicStateT")


def extract_code_markdown(markdown_text: str) -> Optional[str]:
    """
    Extracts code from the first markdown fenced code block using line-by-line parsing.

    This function is designed to be robust against malformed markdown,
    such as a missing closing fence.

    Args:
        markdown_text: The string containing markdown text.

    Returns:
        The extracted code as a string. If a starting fence is found but
        no closing fence, it returns all lines until the end of the string.
        Returns None if no starting fence is found.
    """
    lines = markdown_text.splitlines()
    code_lines = []

    # Use an iterator to allow consuming lines within the loop
    line_iterator = iter(lines)

    opening_fence = None

    # 1. Find the start of the first code block
    for line in line_iterator:
        stripped_line = line.strip()
        if stripped_line.startswith("```"):
            opening_fence = "```"
            break
        elif stripped_line.startswith("~~~"):
            opening_fence = "~~~"
            break

    # If no opening fence was found in the entire text, return None
    if opening_fence is None:
        return None

    # 2. Collect lines until the closing fence or end of text
    for line in line_iterator:
        # The closing fence must match the opening one and be on its own line
        if line.strip().startswith(opening_fence):
            # Found the end of the block
            break
        code_lines.append(line)

    return "\n".join(code_lines)


class SynthesisDependenciesProvider(Protocol[SymbolicStateT]):
    def get_synthesis_prompt(
        self, transition: SymbolicTransition[SymbolicStateT], object_type: str
    ) -> str:
        """Get the synthesis prompt for a specific transition."""
        raise NotImplementedError

    def get_system_prompt(self) -> str:
        """Get the system prompt for synthesis."""
        raise NotImplementedError

    def get_executor(self) -> ExecWithLimitedNamespace:
        """Get the executor for the synthesizer."""
        raise NotImplementedError


class GenericSynthesizer(Generic[SymbolicStateT]):
    """
    This synthesizer uses LLM calls to generate Python expert functions that
    explain observed object lifecycle events (creation, deletion, replacement).
    It follows the PoE-World approach of surprise-driven synthesis, only generating
    experts for transitions that the current model cannot explain well.
    """

    def __init__(
        self,
        dependencies_provider: SynthesisDependenciesProvider[SymbolicStateT],
        llm_params: Optional[GeminiLiteLlmParams] = None,
        llm_client: Callable[
            [LiteLlmRequest], NonStreamingModelResponse
        ] = lambda request: request(),
    ):
        """
        Initialize the synthesizer.

        Args:
            llm_params: LLM parameters for synthesis. If None, uses default Gemini params.
            llm_client: Callable that takes a LiteLlmRequest and returns a NonStreamingModelResponse.
                This is mostly useful for testing, since it can be used to return a mocked response.
        """
        self.dependencies_provider = dependencies_provider
        self.llm_params = llm_params or GeminiLiteLlmParams()
        self.llm_client = llm_client

    async def synthesize_experts(
        self,
        transitions: List[SymbolicTransition[SymbolicStateT]],
        object_type: str,
    ) -> List[WeightedExpert]:
        """
        Synthesize expert programs from state transitions.

        This method expects transitions that have already been filtered for "surprising"
        ones by the calling ObjModelLearner. The synthesizer focuses purely on
        generating experts from the provided transitions.

        Args:
            transitions: Sequence of state transitions to analyze (already filtered for surprising ones)
            object_type: Type of object to synthesize experts for

        Returns:
            List of WeightedExpert objects containing compiled expert functions
        """
        if not transitions:
            return []

        # Generate experts for all provided transitions (assumed to be surprising)
        experts: List[WeightedExpert] = []
        for transition in transitions:
            try:
                expert = await self._synthesize_expert_for_transition(
                    transition, object_type
                )
                if expert:
                    experts.append(expert)
            except Exception as e:
                logger.error(f"Failed to synthesize expert for transition: {e}")
                continue

        return experts

    async def _synthesize_expert_for_transition(
        self,
        transition: SymbolicTransition[SymbolicStateT],
        object_type: str,
    ) -> Optional[WeightedExpert]:
        """
        Synthesize a single expert for a specific transition.

        Args:
            transition: The state transition to explain
            object_type: Type of object to focus on

        Returns:
            WeightedExpert or None if synthesis failed
        """
        # Create prompt for the LLM
        prompt = self.dependencies_provider.get_synthesis_prompt(
            transition, object_type
        )

        # Call LLM
        request = LiteLlmRequest(
            messages=[
                LiteLlmMessage(
                    role="system",
                    content=self.dependencies_provider.get_system_prompt(),
                ),
                LiteLlmMessage(role="user", content=prompt),
            ],
            params=self.llm_params,
        )

        try:
            response = self.llm_client(request)
            code = response.choices[0].message.content

            if not code:
                logger.warning("Empty response from LLM")
                return None

            # Extract and validate the generated code
            expert_code = self._extract_expert_function(code)
            if not expert_code:
                logger.warning(
                    "Failed to extract valid expert function from LLM response"
                )
                return None

            # Validate the code
            if not self._validate_expert_code(expert_code):
                logger.warning(
                    f"Generated expert code failed validation:\n{expert_code}"
                )
                return None

            # Compile the expert function
            expert_function = self._compile_expert_function(expert_code, object_type)
            if not expert_function:
                logger.warning("Failed to compile expert function")
                return None

            return WeightedExpert(
                expert_function=expert_function,
                weight=1.0,
                is_fitted=False,
            )

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def _extract_expert_function(self, llm_response: str) -> Optional[str]:
        """Extract the expert function code from the LLM response using AST parsing."""
        try:

            code = extract_code_markdown(llm_response)
            if not code:
                # Assume that the code is not markdown fenced and try to
                # parse the whole response
                code = llm_response

            # Parse the entire response to get the AST
            tree = ast.parse(code)

            # Find all function definitions
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Look for the function that matches our expected naming pattern
                    if node.name.startswith("alter_"):
                        # Extract the function source code directly from the AST
                        function_code = ast.unparse(node)
                        functions.append(function_code)

            # Return the first matching function, or None if none found
            if functions:
                logger.info(f"Found {len(functions)} matching functions")
                return functions[0]
            else:
                logger.warning(
                    f"No functions starting with 'alter_' found in LLM response: {llm_response}"
                )
                return None

        except SyntaxError as e:
            logger.error(f"Failed to parse LLM response as Python code: {e}")
            logger.debug(f"LLM response that caused syntax error: {llm_response}...")
            return None

        except Exception as e:
            logger.error(f"Unexpected error during function extraction: {e}")
            logger.debug(f"LLM response: {llm_response}...")
            return None

    def _validate_expert_code(self, code: str) -> bool:
        """Validate that the generated expert code is syntactically correct."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _compile_expert_function(
        self, code: str, object_type: str
    ) -> Optional[ExpertFunction[SymbolicStateT]]:
        """Compile the generated code into a callable expert function."""
        try:
            function_name = f"alter_{object_type}_objects"

            # Create executor with access to required classes
            executor = self.dependencies_provider.get_executor()

            # Compile the code
            executor(code)

            # Extract the compiled function from the namespace
            compiled_function = executor.namespace[function_name]

            expert_function = ExpertFunctionWrapper(compiled_function, code)

            logger.success(f"Compiled expert function: {expert_function.__name__}")

            return expert_function

        except Exception as e:
            logger.error(f"Failed to compile expert function: {e}")
            return None


implements(ExpertSynthesizerProtocol)(GenericSynthesizer)


class NoOpSynthesizer(Generic[SymbolicStateT]):
    """
    A synthesizer that does nothing.
    """

    def __init__(self):
        pass

    async def synthesize_experts(
        self, transitions: List[SymbolicTransition[SymbolicStateT]], object_type: str
    ) -> List[WeightedExpert]:
        return []


implements(ExpertSynthesizerProtocol)(NoOpSynthesizer)
