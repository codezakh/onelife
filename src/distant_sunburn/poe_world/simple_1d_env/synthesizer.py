"""
PoE-World synthesizer for the simple 1D environment.

This module implements the expert synthesis algorithm that generates Python code
to explain observed state transitions in the simple 1D environment.
"""

from typing import List, Optional
import ast
from loguru import logger

from ...simple_1d_env.environment import GameState, Action
from ..core import (
    SymbolicTransition,
    ExpertSynthesizerProtocol,
    WeightedExpert,
    ExpertFunction,
)
from ...litellm_utils import LiteLlmRequest, LiteLlmMessage, GeminiLiteLlmParams
from ...typing_utils import implements
from ...local_code_execution import ExecWithLimitedNamespace


class Simple1DExpertSynthesizer:
    """
    Expert synthesizer for the simple 1D environment.

    This synthesizer uses LLM calls to generate Python expert functions that
    explain observed state transitions in the 1D environment. It follows the
    PoE-World approach of surprise-driven synthesis, only generating experts
    for transitions that the current model cannot explain well.
    """

    def __init__(self, llm_params: Optional[GeminiLiteLlmParams] = None):
        """
        Initialize the synthesizer.

        Args:
            llm_params: LLM parameters for synthesis. If None, uses default Gemini params.
        """
        self.llm_params = llm_params or GeminiLiteLlmParams()

    async def synthesize_experts(
        self,
        transitions: List[SymbolicTransition[GameState]],
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
        experts = []
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
        transition: SymbolicTransition[GameState],
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
        prompt = self._create_synthesis_prompt(transition, object_type)

        # Call LLM
        request = LiteLlmRequest(
            messages=[
                LiteLlmMessage(role="system", content=self._get_system_prompt()),
                LiteLlmMessage(role="user", content=prompt),
            ],
            params=self.llm_params,
        )

        try:
            response = request()
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
                logger.warning("Generated expert code failed validation")
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

    def _get_system_prompt(self) -> str:
        """Get the system prompt for expert synthesis."""
        return """You are an expert at analyzing 1D game state transitions and generating Python functions that explain the observed changes.

Your task is to generate a Python function that modifies a GameState object to explain the observed state transition. The function should:

1. Take a GameState object and an action as input
2. Modify the state in-place by assigning DiscreteDistribution objects to attributes
3. Only modify attributes that are relevant to the observed changes
4. Use DiscreteDistribution(support=[value]) to make deterministic predictions
5. Return None (modify state in-place)

The function should be named `alter_{object_type}_objects` where {object_type} is the type of object being modified.

IMPORTANT: Only modify observable attributes that are tracked by the system:
- player.position: Player's position in the 1D world (0-11)
- lights: List of Light objects, where each Light has:
  * position: The light's position in the world
  * is_on: Whether the light is on (boolean)

Do NOT modify other attributes like config, rng, or non-observable properties.

Example:
```python
def alter_player_objects(current_state: GameState, action: Action) -> None:
    if action == Action.MOVE_RIGHT:
        new_position = min(current_state.config.width - 1, current_state.player.position + 1)
        current_state.player.position = DiscreteDistribution(support=[new_position])

def alter_light_objects(current_state: GameState, action: Action) -> None:
    # Example: toggle the first light
    if len(current_state.lights) > 0:
        current_state.lights[0].is_on = DiscreteDistribution(support=[1])
```

Generate only the function code, no explanations or markdown formatting."""

    def _create_synthesis_prompt(
        self,
        transition: SymbolicTransition[GameState],
        object_type: str,
    ) -> str:
        """Create a prompt for synthesizing an expert for a specific transition."""

        # Extract key changes
        changes = self._extract_state_changes(transition)

        prompt = f"""Analyze this state transition and generate a Python function that explains the changes:

**Action:** {transition.action}

**Key Changes:**
{changes}

**Object Type:** {object_type}

Please generate a Python function named `alter_{object_type}_objects` that explains these changes. The function should:
- Take `current_state: GameState` and `action: Action` as parameters
- Modify the current_state in-place by assigning DiscreteDistribution objects to relevant attributes
- Return None
- Only modify observable attributes that are relevant to the observed changes:
  * player.position (0-11)
  * lights[i].is_on (boolean for each light in the lights list)

Generate only the function code:"""

        return prompt

    def _extract_state_changes(self, transition: SymbolicTransition[GameState]) -> str:
        """Extract a human-readable description of state changes for observable attributes only."""
        prev_state = transition.prev_metadata
        next_state = transition.next_metadata
        changes = []

        # Player position changes (observable: player_position)
        if prev_state.player.position != next_state.player.position:
            changes.append(
                f"- Player moved from position {prev_state.player.position} to {next_state.player.position}"
            )

        # Light state changes (observable: light_0_is_on, light_1_is_on)
        for i, (prev_light, next_light) in enumerate(
            zip(prev_state.lights, next_state.lights)
        ):
            if prev_light.is_on != next_light.is_on:
                changes.append(
                    f"- Light {i} changed from {'on' if prev_light.is_on else 'off'} to {'on' if next_light.is_on else 'off'}"
                )

        return "\n".join(changes) if changes else "- No significant changes detected"

    def _extract_expert_function(self, llm_response: str) -> Optional[str]:
        """Extract the expert function code from the LLM response."""
        # Look for function definition
        lines = llm_response.strip().split("\n")
        function_lines = []
        in_function = False

        for line in lines:
            if line.strip().startswith("def "):
                in_function = True
                function_lines.append(line)
            elif in_function:
                if line.strip() == "" or line.strip().startswith("```"):
                    break
                function_lines.append(line)

        if function_lines:
            return "\n".join(function_lines)

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
    ) -> Optional[ExpertFunction[GameState]]:
        """Compile the generated code into a callable expert function."""
        try:
            function_name = f"alter_{object_type}_objects"

            # Create executor with access to required classes
            from ...simple_1d_env.environment import GameState, Action
            from ..core import DiscreteDistribution

            executor = ExecWithLimitedNamespace(
                inherited_scope={
                    "GameState": GameState,
                    "Action": Action,
                    "DiscreteDistribution": DiscreteDistribution,
                },
                allowed_names={"GameState", "Action", "DiscreteDistribution"},
            )

            # Compile the code
            executor(code)

            # Extract the compiled function from the namespace
            expert_function = executor.namespace[function_name]

            return expert_function

        except Exception as e:
            logger.error(f"Failed to compile expert function: {e}")
            return None


implements(ExpertSynthesizerProtocol[GameState])(Simple1DExpertSynthesizer)
