"""
PoE-World synthesizer for the Crafter environment.

This module implements the expert synthesis algorithm that generates Python code
to explain observed state transitions in the Crafter environment.
"""

import asyncio
from typing import Any, List, Protocol, Tuple, Optional
from dataclasses import dataclass
import ast
import re
from loguru import logger

from crafter.state_export import WorldState
from crafter.constants import ActionT as CrafterAction
from ..core import SymbolicTransition, ExpertFunction, DiscreteDistribution
from ...litellm_utils import LiteLlmRequest, LiteLlmMessage, GeminiLiteLlmParams


@dataclass
class SynthesizedExpert:
    """A synthesized expert function."""

    code: str
    object_type: str
    description: str


class ExpertSynthesizerProtocol(Protocol):
    """Protocol for expert synthesis from state transitions."""

    async def synthesize_experts(
        self,
        transitions: List[SymbolicTransition[WorldState]],
        object_type: str,
    ) -> List[SynthesizedExpert]:
        """
        Synthesize expert programs from state transitions.

        This method expects transitions that have already been filtered for "surprising"
        ones by the calling ObjModelLearner. The synthesizer focuses purely on
        generating experts from the provided transitions.
        """
        ...


class CrafterExpertSynthesizer:
    """
    General-purpose expert synthesizer for the Crafter environment.

    This synthesizer uses LLM calls to generate Python expert functions that
    explain observed state transitions. It follows the PoE-World approach of
    surprise-driven synthesis, only generating experts for transitions that
    the current model cannot explain well.
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
        transitions: List[SymbolicTransition[WorldState]],
        object_type: str,
    ) -> List[SynthesizedExpert]:
        """
        Synthesize expert programs from state transitions.

        This method expects transitions that have already been filtered for "surprising"
        ones by the calling ObjModelLearner. The synthesizer focuses purely on
        generating experts from the provided transitions.

        Args:
            transitions: Sequence of state transitions to analyze (already filtered for surprising ones)
            object_type: Type of object to synthesize experts for

        Returns:
            List of synthesized expert programs
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
                # TODO: Consider this out of scope for now as per Q&A
                continue

        return experts

    async def _synthesize_expert_for_transition(
        self,
        transition: SymbolicTransition[WorldState],
        object_type: str,
    ) -> Optional[SynthesizedExpert]:
        """
        Synthesize a single expert for a specific transition.

        Args:
            transition: The state transition to explain
            object_type: Type of object to focus on

        Returns:
            Synthesized expert or None if synthesis failed
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

            return SynthesizedExpert(
                code=expert_code,
                object_type=object_type,
                description=f"Generated expert for {object_type} based on transition",
            )

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def _get_system_prompt(self) -> str:
        """Get the system prompt for expert synthesis."""
        return """You are an expert at analyzing game state transitions and generating Python functions that explain the observed changes.

Your task is to generate a Python function that modifies a WorldState object to explain the observed state transition. The function should:

1. Take a WorldState object and an action as input
2. Modify the state in-place by assigning DiscreteDistribution objects to attributes
3. Only modify attributes that are relevant to the observed changes
4. Use DiscreteDistribution(support=[value]) to make deterministic predictions
5. Return the modified state

The function should be named `alter_{object_type}_objects` where {object_type} is the type of object being modified.

IMPORTANT: Only modify observable attributes that are tracked by the system:
- player_position_x, player_position_y, player_health
- entity_{entity_id}_position_x, entity_{entity_id}_position_y, entity_{entity_id}_health

Do NOT modify inventory, achievements, or other non-observable attributes.

Example:
```python
def alter_player_objects(state: WorldState, action: str) -> WorldState:
    if action == "move_right":
        new_x = min(state.size[0] - 1, state.player.position.x + 1)
        state.player.position.x = DiscreteDistribution(support=[new_x])
    return state
```

Generate only the function code, no explanations or markdown formatting."""

    def _create_synthesis_prompt(
        self,
        transition: SymbolicTransition[WorldState],
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
- Take `state: WorldState` and `action: str` as parameters
- Modify the state in-place by assigning DiscreteDistribution objects to relevant attributes
- Return the modified state
- Only modify observable attributes that are relevant to the observed changes:
  * player_position_x, player_position_y, player_health
  * entity_{{entity_id}}_position_x, entity_{{entity_id}}_position_y, entity_{{entity_id}}_health

Generate only the function code:"""

        return prompt

    def _extract_state_changes(self, transition: SymbolicTransition[WorldState]) -> str:
        """Extract a human-readable description of state changes for observable attributes only."""
        prev_state = transition.prev_metadata
        next_state = transition.next_metadata
        changes = []

        # Player position changes (observable: player_position_x, player_position_y)
        if (
            prev_state.player.position.x != next_state.player.position.x
            or prev_state.player.position.y != next_state.player.position.y
        ):
            changes.append(
                f"- Player moved from ({prev_state.player.position.x}, {prev_state.player.position.y}) to ({next_state.player.position.x}, {next_state.player.position.y})"
            )

        # Player health changes (observable: player_health)
        if prev_state.player.health != next_state.player.health:
            changes.append(
                f"- Player health changed from {prev_state.player.health} to {next_state.player.health}"
            )

        # Entity changes (observable: entity_{entity_id}_position_x, entity_{entity_id}_position_y, entity_{entity_id}_health)
        prev_entities = {e.entity_id: e for e in prev_state.objects}
        next_entities = {e.entity_id: e for e in next_state.objects}

        # Check for entity movement, health changes, or new entities
        for entity_id in set(prev_entities.keys()) | set(next_entities.keys()):
            if entity_id in prev_entities and entity_id in next_entities:
                prev_entity = prev_entities[entity_id]
                next_entity = next_entities[entity_id]

                # Movement (observable: entity_{entity_id}_position_x, entity_{entity_id}_position_y)
                if (
                    prev_entity.position.x != next_entity.position.x
                    or prev_entity.position.y != next_entity.position.y
                ):
                    changes.append(
                        f"- {prev_entity.name} (ID: {entity_id}) moved from ({prev_entity.position.x}, {prev_entity.position.y}) to ({next_entity.position.x}, {next_entity.position.y})"
                    )

                # Health change (observable: entity_{entity_id}_health)
                if prev_entity.health != next_entity.health:
                    changes.append(
                        f"- {prev_entity.name} (ID: {entity_id}) health changed from {prev_entity.health} to {next_entity.health}"
                    )

            elif entity_id in next_entities:
                # New entity
                entity = next_entities[entity_id]
                changes.append(
                    f"- New {entity.name} (ID: {entity_id}) appeared at ({entity.position.x}, {entity.position.y}) with health {entity.health}"
                )

            elif entity_id in prev_entities:
                # Entity disappeared
                entity = prev_entities[entity_id]
                changes.append(f"- {entity.name} (ID: {entity_id}) disappeared")

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


# Convenience function for creating a synthesizer
def create_crafter_synthesizer(
    llm_params: Optional[GeminiLiteLlmParams] = None,
) -> ExpertSynthesizerProtocol:
    """Create a Crafter expert synthesizer."""
    return CrafterExpertSynthesizer(llm_params=llm_params)
