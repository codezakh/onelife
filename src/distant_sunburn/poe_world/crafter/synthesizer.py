"""
PoE-World synthesizer for the Crafter environment.

This module implements the expert synthesis algorithm that generates Python code
to explain observed state transitions in the Crafter environment.
"""

from typing import List, Optional
import ast
from loguru import logger

from crafter.state_export import WorldState
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
from crafter.state_export import (
    ArrowState,
    CowState,
    FenceState,
    PlantState,
    Position,
    SkeletonState,
    WorldState,
    ZombieState,
)
from ..core import DiscreteDistribution
from functools import partial


class CrafterSynthesisDependenciesProvider:
    def get_system_prompt(self) -> str:
        """Get the system prompt for expert synthesis."""
        return """You are an expert at analyzing game state transitions and generating Python functions that explain the observed changes.

Your task is to generate a Python function that modifies a WorldState object to explain the observed state transition. The function should:

1. Take a WorldState object and an action as input
2. Modify the state in-place by assigning DiscreteDistribution objects to attributes
3. Only modify attributes that are relevant to the observed changes
4. Use DiscreteDistribution(support=[value]) to make deterministic predictions
5. Return None (modify state in-place)

The function should be named `alter_{object_type}_objects` where {object_type} is the type of object being modified.

IMPORTANT: Use the correct attribute paths that match the actual WorldState structure:

**Player attributes:**
- current_state.player.position.x, current_state.player.position.y
- current_state.player.health
- current_state.player.facing.x, current_state.player.facing.y

**Entity attributes (for objects in current_state.objects):**
- Find entities by iterating through current_state.objects
- Access: obj.position.x, obj.position.y, obj.health, obj.entity_id
- Example: for obj in current_state.objects: if obj.name == "cow": obj.health = new_value

**World attributes:**
- current_state.size[0], current_state.size[1] (world dimensions)
- current_state.daylight, current_state.step_count

Do NOT modify inventory, achievements, or other non-observable attributes.

Example:
```python
def alter_cow_objects(current_state: WorldState, action: str) -> None:
    if action == "do":
        # Find the cow and modify its health
        for obj in current_state.objects:
            if obj.name == "cow":
                obj.health = DiscreteDistribution(support=[max(0, obj.health - 2)])
                break
```

Generate only the function code, no explanations or markdown formatting."""

    def get_synthesis_prompt(
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
- Take `current_state: WorldState` and `action: str` as parameters
- Modify the current_state in-place by assigning DiscreteDistribution objects to relevant attributes
- Return None
- Use the correct attribute paths:
  * Player: current_state.player.position.x, current_state.player.position.y, current_state.player.health
  * Entities: Iterate through current_state.objects and access obj.position.x, obj.position.y, obj.health, obj.entity_id
  * Example: for obj in current_state.objects: if obj.name == "cow": obj.health = new_value

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

    def get_executor(self) -> ExecWithLimitedNamespace:
        """Get the executor for the synthesizer."""
        return ExecWithLimitedNamespace(
            inherited_scope={
                "WorldState": WorldState,
                "CowState": CowState,
                "ZombieState": ZombieState,
                "SkeletonState": SkeletonState,
                "ArrowState": ArrowState,
                "PlantState": PlantState,
                "FenceState": FenceState,
                "Position": Position,
                "DiscreteDistribution": DiscreteDistribution,
            },
            allowed_names={
                "WorldState",
                "CowState",
                "ZombieState",
                "SkeletonState",
                "ArrowState",
                "PlantState",
                "FenceState",
                "Position",
                "DiscreteDistribution",
            },
        )


CrafterExpertSynthesizer = GenericSynthesizer[WorldState]

implements(ExpertSynthesizerProtocol[WorldState])(CrafterExpertSynthesizer)
