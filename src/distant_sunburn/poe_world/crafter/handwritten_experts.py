"""
Hand-written experts for the Crafter environment.

This module contains both correct and incorrect expert functions that model
the mechanics of the Crafter environment. These experts are used to test
the PoE-World weight-fitting pipeline.

Correct experts perfectly model the true environment mechanics, while incorrect
experts introduce deliberate flaws to test the system's ability to distinguish
between good and bad models.

The experts focus on observable mechanics:
- Player movement (position changes)
- Combat mechanics (health changes)
- Entity AI behavior (movement and health changes)
"""

from typing import Any, Literal, Callable, TypeVar
import numpy as np
import inspect

from ..core import DiscreteDistribution
from crafter.state_export import WorldState
from crafter.constants import ActionT
from ..core import ExpertFunctionWrapper
from loguru import logger

# Action types for Crafter
# Action = Literal[
#     "move_left", "move_right", "move_up", "move_down", "do", "sleep", "place", "make"
# ]

logger.disable(__name__)


def correct_player_movement_expert(
    current_state: WorldState, action: ActionT, **context: Any
) -> None:
    """
    Correct expert that models player movement mechanics.

    This expert models:
    - Player moves in the direction of the action (left, right, up, down)
    - Movement is bounded by world boundaries
    - Player position is updated immediately for movement actions
    - Non-movement actions don't change player position

    Args:
        current_state: The current game state to modify in-place
        action: The action being taken
        **context: Additional context (unused)
    """
    # Only movement actions change player position
    if action not in ["move_left", "move_right", "move_up", "move_down"]:
        return

    # Calculate new position based on action
    new_x = current_state.player.position.x
    new_y = current_state.player.position.y

    if action == "move_left":
        new_x = max(0, new_x - 1)
    elif action == "move_right":
        new_x = min(current_state.size[0] - 1, new_x + 1)
    elif action == "move_up":
        new_y = max(0, new_y - 1)
    elif action == "move_down":
        new_y = min(current_state.size[1] - 1, new_y + 1)

    # Assign predictions using DiscreteDistribution
    current_state.player.position.x = DiscreteDistribution(support=[new_x])  # type: ignore
    current_state.player.position.y = DiscreteDistribution(support=[new_y])  # type: ignore


def correct_combat_damage_expert(
    current_state: WorldState, action: ActionT, focus: str = "all", **context: Any
) -> None:
    """
    Correct expert that models combat damage mechanics.

    This expert models:
    - 'do' action can damage entities when facing them
    - Player can take damage from combat
    - Entities can take damage and be removed at zero health
    - Base damage values for different entity types

    Args:
        current_state: The current game state to modify in-place
        action: The action being taken
        focus: Entity type to focus on ('all', 'cow', 'zombie', 'skeleton', etc.)
        **context: Additional context (unused)
    """
    if action != "do":
        return

    # Get the tile the player is facing
    facing_x = current_state.player.position.x + current_state.player.facing.x
    facing_y = current_state.player.position.y + current_state.player.facing.y

    # Check if there's an entity at the facing position
    target_entity = None
    for entity in current_state.objects:
        if (
            entity.position.x == facing_x
            and entity.position.y == facing_y
            and entity.entity_id != current_state.player.entity_id
        ):
            target_entity = entity
            break

    if target_entity:
        # Check if we should focus on this entity type
        if focus != "all" and target_entity.name != focus:
            return  # Skip this entity if it doesn't match the focus

        # Calculate damage based on player's weapons
        base_damage = 1
        if current_state.player.inventory.wood_sword > 0:
            base_damage = 2
        elif current_state.player.inventory.stone_sword > 0:
            base_damage = 3
        elif current_state.player.inventory.iron_sword > 0:
            base_damage = 5

        # Apply damage to target entity
        new_health = max(0, target_entity.health - base_damage)
        target_entity.health = DiscreteDistribution(support=[new_health])  # type: ignore


def correct_entity_ai_expert(
    current_state: WorldState, action: ActionT, focus: str = "all", **context: Any
) -> None:
    """
    Correct expert that models entity AI behavior.

    This expert models:
    - Cows move randomly with 50% probability
    - Zombies pursue players within 8 tiles and attack when adjacent
    - Skeletons flee when close, shoot arrows when in range
    - Entities are removed when health reaches 0

    Args:
        current_state: The current game state to modify in-place
        action: The action being taken (affects entity behavior)
        focus: Entity type to focus on ('all', 'cow', 'zombie', 'skeleton', etc.)
        **context: Additional context (unused)
    """
    # Use the state's random number generator for deterministic behavior
    rng = current_state.random_state

    # Track total damage to player from all entities
    total_player_damage = 0

    with logger.contextualize(focus=focus):
        for entity in current_state.objects:
            if entity.entity_id == current_state.player.entity_id:
                continue  # Skip player

            # Calculate distance to player
            distance = abs(entity.position.x - current_state.player.position.x) + abs(
                entity.position.y - current_state.player.position.y
            )

            # Check if we should focus on this entity type
            if focus != "all" and entity.name != focus:
                continue  # Skip this entity if it doesn't match the focus

            # Handle different entity types
            if entity.name == "cow":
                # Cows move randomly with 50% probability
                if rng.uniform() < 0.5:
                    # Random direction
                    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    dx, dy = directions[rng.randint(0, 3)]
                    new_x = max(
                        0, min(current_state.size[0] - 1, entity.position.x + dx)
                    )
                    new_y = max(
                        0, min(current_state.size[1] - 1, entity.position.y + dy)
                    )
                    entity.position.x = DiscreteDistribution(support=[new_x])  # type: ignore
                    entity.position.y = DiscreteDistribution(support=[new_y])  # type: ignore

            elif entity.name == "zombie":
                # Zombies pursue players within 8 tiles
                chase_roll = rng.uniform() < 0.9
                long_axis_roll = rng.uniform() < 0.8
                logger.debug(
                    f"Zombie {entity.entity_id} chase_roll: {chase_roll}, long_axis_roll: {long_axis_roll}"
                )
                if distance <= 8 and chase_roll:
                    # Move toward player
                    dx = np.sign(current_state.player.position.x - entity.position.x)
                    dy = np.sign(current_state.player.position.y - entity.position.y)

                    # Prefer movement along the longer axis
                    if abs(current_state.player.position.x - entity.position.x) > abs(
                        current_state.player.position.y - entity.position.y
                    ):
                        if long_axis_roll:
                            new_x = max(
                                0,
                                min(current_state.size[0] - 1, entity.position.x + dx),
                            )
                            entity.position.x = DiscreteDistribution(support=[new_x])  # type: ignore
                        else:
                            new_y = max(
                                0,
                                min(current_state.size[1] - 1, entity.position.y + dy),
                            )
                            entity.position.y = DiscreteDistribution(support=[new_y])  # type: ignore
                    else:
                        if long_axis_roll:
                            new_y = max(
                                0,
                                min(current_state.size[1] - 1, entity.position.y + dy),
                            )
                            entity.position.y = DiscreteDistribution(support=[new_y])  # type: ignore
                        else:
                            new_x = max(
                                0,
                                min(current_state.size[0] - 1, entity.position.x + dx),
                            )
                            entity.position.x = DiscreteDistribution(support=[new_x])  # type: ignore

                    # Attack if adjacent (simplified - just damage player)
                    if distance <= 1:
                        damage = 7 if current_state.player.sleeping else 2
                        total_player_damage += damage

            elif entity.name == "skeleton":
                # Skeletons flee when close, shoot when in range
                if distance <= 3:
                    # Flee from player
                    dx = np.sign(entity.position.x - current_state.player.position.x)
                    dy = np.sign(entity.position.y - current_state.player.position.y)
                    new_x = max(
                        0, min(current_state.size[0] - 1, entity.position.x + dx)
                    )
                    new_y = max(
                        0, min(current_state.size[1] - 1, entity.position.y + dy)
                    )
                    entity.position.x = DiscreteDistribution(support=[new_x])  # type: ignore
                    entity.position.y = DiscreteDistribution(support=[new_y])  # type: ignore
                elif distance <= 5 and rng.uniform() < 0.5:
                    # Shoot arrow (simplified - just damage player)
                    total_player_damage += 2

        # Apply accumulated damage to player at the end
        if total_player_damage > 0:
            new_health = max(0, current_state.player.health - total_player_damage)
            current_state.player.health = DiscreteDistribution(support=[new_health])  # type: ignore


def incorrect_player_movement_expert_teleports(
    current_state: WorldState, action: ActionT, **context: Any
) -> None:
    """
    Incorrect expert that makes player teleport to random positions.

    This expert is obviously wrong - any movement action teleports the player
    to a random position instead of moving one step in the intended direction.

    Args:
        current_state: The current game state to modify in-place
        action: The action being taken
        **context: Additional context (unused)
    """
    # Any movement action causes teleportation
    if action in ["move_left", "move_right", "move_up", "move_down"]:
        # Teleport to random position (obviously wrong!)
        rng = current_state.random_state
        new_x = rng.randint(0, current_state.size[0] - 1)
        new_y = rng.randint(0, current_state.size[1] - 1)

        current_state.player.position.x = DiscreteDistribution(support=[new_x])  # type: ignore
        current_state.player.position.y = DiscreteDistribution(support=[new_y])  # type: ignore


def incorrect_combat_damage_expert_instakills(
    current_state: WorldState, action: ActionT, **context: Any
) -> None:
    """
    Incorrect expert that instantly kills any entity when attacked.

    This expert is obviously wrong - any 'do' action against an entity
    immediately sets their health to 0, regardless of weapons or damage.

    Args:
        current_state: The current game state to modify in-place
        action: The action being taken
        **context: Additional context (unused)
    """
    if action != "do":
        return

    # Get the tile the player is facing
    facing_x = current_state.player.position.x + current_state.player.facing.x
    facing_y = current_state.player.position.y + current_state.player.facing.y

    # Check if there's an entity at the facing position
    target_entity = None
    for entity in current_state.objects:
        if (
            entity.position.x == facing_x
            and entity.position.y == facing_y
            and entity.entity_id != current_state.player.entity_id
        ):
            target_entity = entity
            break

    if target_entity:
        # Instantly kill the entity (obviously wrong!)
        target_entity.health = DiscreteDistribution(support=[0])  # type: ignore


def incorrect_entity_ai_expert_self_destructs(
    current_state: WorldState, action: ActionT, **context: Any
) -> None:
    """
    Incorrect expert that makes entities self-destruct.

    This expert is obviously wrong - all entities immediately set their own
    health to 0 on every action, causing them to disappear.

    Args:
        current_state: The current game state to modify in-place
        action: The action being taken
        **context: Additional context (unused)
    """
    for entity in current_state.objects:
        if entity.entity_id == current_state.player.entity_id:
            continue  # Skip player

        # All entities self-destruct (obviously wrong!)
        entity.health = DiscreteDistribution(support=[0])  # type: ignore


def incorrect_entity_lifecycle_expert_spurious_spawning(
    current_state: WorldState, action: ActionT, **context: Any
) -> None:
    """
    Incorrect expert that makes spurious predictions about entity spawning and deletion.

    This expert is obviously wrong - it predicts that entities will be created
    or deleted in situations where they shouldn't be, specifically:
    - Predicts cows spawn when player moves
    - Predicts zombies disappear when player attacks
    - Predicts skeletons multiply when player sleeps

    Args:
        current_state: The current game state to modify in-place
        action: The action being taken
        **context: Additional context (unused)
    """
    # Bad prediction: Cows spawn when player moves
    if action in ["move_left", "move_right", "move_up", "move_down"]:
        # This is wrong - movement shouldn't create cows
        # We can't actually create entities here, but we can predict their existence
        # by setting their health to a non-zero value (indicating they should exist)
        for entity in current_state.objects:
            if entity.name == "cow":
                # Predict cow will exist with high health (wrong prediction)
                entity.health = DiscreteDistribution(support=[50])  # type: ignore

    # Bad prediction: Zombies disappear when player attacks
    if action == "do":
        # This is wrong - attacking shouldn't make zombies disappear
        for entity in current_state.objects:
            if entity.name == "zombie":
                # Predict zombie will be deleted (wrong prediction)
                entity.health = DiscreteDistribution(support=[0])  # type: ignore

    # Bad prediction: Skeletons multiply when player sleeps
    if action == "sleep":
        # This is wrong - sleeping shouldn't create skeletons
        for entity in current_state.objects:
            if entity.name == "skeleton":
                # Predict skeleton will exist with high health (wrong prediction)
                entity.health = DiscreteDistribution(support=[30])  # type: ignore


# Collection of all experts for easy access
CORRECT_EXPERTS = [
    ExpertFunctionWrapper.from_non_runtime_created(correct_player_movement_expert),
    ExpertFunctionWrapper.from_non_runtime_created(correct_combat_damage_expert),
    ExpertFunctionWrapper.from_non_runtime_created(correct_entity_ai_expert),
]

INCORRECT_EXPERTS = [
    ExpertFunctionWrapper.from_non_runtime_created(
        incorrect_player_movement_expert_teleports
    ),
    ExpertFunctionWrapper.from_non_runtime_created(
        incorrect_combat_damage_expert_instakills
    ),
    ExpertFunctionWrapper.from_non_runtime_created(
        incorrect_entity_ai_expert_self_destructs
    ),
    ExpertFunctionWrapper.from_non_runtime_created(
        incorrect_entity_lifecycle_expert_spurious_spawning
    ),
]

ALL_EXPERTS = CORRECT_EXPERTS + INCORRECT_EXPERTS
