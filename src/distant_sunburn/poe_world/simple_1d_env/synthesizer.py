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
from ..synthesizer import GenericSynthesizer, SynthesisDependenciesProvider
from ...typing_utils import implements
from ...simple_1d_env.environment import GameState, Action
from ..core import DiscreteDistribution


class Simple1DSynthesisDependenciesProvider:
    def get_system_prompt(self) -> str:
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

    def get_synthesis_prompt(
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

    def get_executor(self) -> ExecWithLimitedNamespace:

        return ExecWithLimitedNamespace(
            inherited_scope={
                "GameState": GameState,
                "Action": Action,
                "DiscreteDistribution": DiscreteDistribution,
            },
            allowed_names={"GameState", "Action", "DiscreteDistribution"},
        )


implements(SynthesisDependenciesProvider[GameState])(
    Simple1DSynthesisDependenciesProvider
)


Simple1DExpertSynthesizer = GenericSynthesizer[GameState]
implements(ExpertSynthesizerProtocol[GameState])(Simple1DExpertSynthesizer)
