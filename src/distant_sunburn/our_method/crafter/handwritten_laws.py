from ..core import LawFunctionWrapper
from crafter.state_export import WorldState
from crafter.constants import ActionT as CrafterAction
from loguru import logger
from ...poe_world.core import DiscreteDistribution
import numpy as np
from crafter.state_export import Inventory


class CorrectPlayerMovementLaw:
    def precondition(self, current_state: WorldState, action: CrafterAction) -> bool:
        return action in {"move_left", "move_right", "move_up", "move_down"}

    def effect(self, current_state: WorldState, action: CrafterAction) -> None:
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


class CorrectCombatDamageLaw:
    def __init__(self, focus: str = "all"):
        self.focus = focus

    def precondition(self, current_state: WorldState, action: CrafterAction) -> bool:
        return action == "do"

    def effect(self, current_state: WorldState, action: CrafterAction) -> None:
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
            if self.focus != "all" and target_entity.name != self.focus:
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


class CorrectEntityAILaw:
    def __init__(self, focus: str = "all"):
        self.focus = focus

    def precondition(self, current_state: WorldState, action: CrafterAction) -> bool:
        return True

    def effect(
        self,
        current_state: WorldState,
        action: CrafterAction,
        focus: str = "all",
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
        focus = self.focus
        # Use the state's random number generator for deterministic behavior
        rng = current_state.random_state

        # Track total damage to player from all entities
        total_player_damage = 0

        with logger.contextualize(focus=focus):
            for entity in current_state.objects:
                if entity.entity_id == current_state.player.entity_id:
                    continue  # Skip player

                # Calculate distance to player
                distance = abs(
                    entity.position.x - current_state.player.position.x
                ) + abs(entity.position.y - current_state.player.position.y)

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
                        dx = np.sign(
                            current_state.player.position.x - entity.position.x
                        )
                        dy = np.sign(
                            current_state.player.position.y - entity.position.y
                        )

                        # Prefer movement along the longer axis
                        if abs(
                            current_state.player.position.x - entity.position.x
                        ) > abs(current_state.player.position.y - entity.position.y):
                            if long_axis_roll:
                                new_x = max(
                                    0,
                                    min(
                                        current_state.size[0] - 1,
                                        entity.position.x + dx,
                                    ),
                                )
                                entity.position.x = DiscreteDistribution(support=[new_x])  # type: ignore
                            else:
                                new_y = max(
                                    0,
                                    min(
                                        current_state.size[1] - 1,
                                        entity.position.y + dy,
                                    ),
                                )
                                entity.position.y = DiscreteDistribution(support=[new_y])  # type: ignore
                        else:
                            if long_axis_roll:
                                new_y = max(
                                    0,
                                    min(
                                        current_state.size[1] - 1,
                                        entity.position.y + dy,
                                    ),
                                )
                                entity.position.y = DiscreteDistribution(support=[new_y])  # type: ignore
                            else:
                                new_x = max(
                                    0,
                                    min(
                                        current_state.size[0] - 1,
                                        entity.position.x + dx,
                                    ),
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
                        dx = np.sign(
                            entity.position.x - current_state.player.position.x
                        )
                        dy = np.sign(
                            entity.position.y - current_state.player.position.y
                        )
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


class UniversalInertialPriorLaw:
    def precondition(self, current_state: WorldState, action: CrafterAction) -> bool:
        return True

    def effect(self, current_state: WorldState, action: CrafterAction) -> None:
        """
        Wrap targeted observable primitives with DiscreteDistribution placing
        all mass on their current value (inertial prior: "nothing changes").

        Targets:
        - Player: position.x, position.y, health
        - Each non-player object: position.x, position.y, health
        - Player inventory: all integer inventory fields

        Leaves existing DiscreteDistribution values unchanged.
        """

        def _wrap_int_like(value):
            if isinstance(value, DiscreteDistribution):
                return value
            # Treat numpy integer types as ints
            if isinstance(value, (int, np.integer)):
                return DiscreteDistribution(support=[int(value)])
            return value

        # Player core observables
        player = current_state.player
        player.position.x = _wrap_int_like(player.position.x)  # type: ignore
        player.position.y = _wrap_int_like(player.position.y)  # type: ignore
        player.health = _wrap_int_like(player.health)  # type: ignore

        # Non-player entities observables
        for entity in current_state.objects:
            if entity.entity_id == player.entity_id:
                continue
            entity.position.x = _wrap_int_like(entity.position.x)  # type: ignore
            entity.position.y = _wrap_int_like(entity.position.y)  # type: ignore
            entity.health = _wrap_int_like(entity.health)  # type: ignore

        # Player inventory observables
        inv = player.inventory
        # Iterate over Inventory model fields to avoid missing any
        for field_name in Inventory.model_fields.keys():
            current_val = getattr(inv, field_name)
            wrapped_val = _wrap_int_like(current_val)
            setattr(inv, field_name, wrapped_val)


class IncorrectPlayerMovementLawTeleports:
    def precondition(self, current_state: WorldState, action: CrafterAction) -> bool:
        return action in {"move_left", "move_right", "move_up", "move_down"}

    def effect(self, current_state: WorldState, action: CrafterAction) -> None:
        rng = current_state.random_state
        new_x = rng.randint(0, current_state.size[0] - 1)
        new_y = rng.randint(0, current_state.size[1] - 1)

        current_state.player.position.x = DiscreteDistribution(support=[new_x])  # type: ignore
        current_state.player.position.y = DiscreteDistribution(support=[new_y])  # type: ignore


class IncorrectCombatDamageLawInstakills:
    def precondition(self, current_state: WorldState, action: CrafterAction) -> bool:
        return action == "do"

    def effect(self, current_state: WorldState, action: CrafterAction) -> None:
        """
        Incorrect expert that instantly kills any entity when attacked.

        This expert is obviously wrong - any 'do' action against an entity
        immediately sets their health to 0, regardless of weapons or damage.

        Args:
            current_state: The current game state to modify in-place
            action: The action being taken
            **context: Additional context (unused)
        """

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


class IncorrectEntityAiLawSelfDestructs:
    def precondition(self, current_state: WorldState, action: CrafterAction) -> bool:
        return True

    def effect(self, current_state: WorldState, action: CrafterAction) -> None:
        for entity in current_state.objects:
            if entity.entity_id == current_state.player.entity_id:
                continue  # Skip player
            entity.health = DiscreteDistribution(support=[0])  # type: ignore


class IncorrectEntityLifecycleLawSpuriousSpawning:
    def precondition(self, current_state: WorldState, action: CrafterAction) -> bool:
        if action in ["move_left", "move_right", "move_up", "move_down", "do", "sleep"]:
            return True
        return False

    def effect(self, current_state: WorldState, action: CrafterAction) -> None:
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


CORRECT_EXPERTS = [
    LawFunctionWrapper.from_non_runtime_created(CorrectPlayerMovementLaw()),
    LawFunctionWrapper.from_non_runtime_created(CorrectCombatDamageLaw()),
    LawFunctionWrapper.from_non_runtime_created(CorrectEntityAILaw()),
]

INCORRECT_EXPERTS = [
    LawFunctionWrapper.from_non_runtime_created(IncorrectPlayerMovementLawTeleports()),
    LawFunctionWrapper.from_non_runtime_created(IncorrectCombatDamageLawInstakills()),
    LawFunctionWrapper.from_non_runtime_created(IncorrectEntityAiLawSelfDestructs()),
    LawFunctionWrapper.from_non_runtime_created(
        IncorrectEntityLifecycleLawSpuriousSpawning()
    ),
]

PRIOR_EXPERTS = [
    # LawFunctionWrapper.from_non_runtime_created(UniversalInertialPriorLaw()),
]

ALL_EXPERTS = CORRECT_EXPERTS + INCORRECT_EXPERTS + PRIOR_EXPERTS
