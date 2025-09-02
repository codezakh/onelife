"""
Scenario definitions for Crafter evaluation.
"""

import copy
from loguru import logger
from typing import Protocol, Callable, TypeVar, cast
from crafter.functional_env import (
    reconstruct_world_from_state,
    export_world_state,
)
from crafter.state_export import WorldState, ZombieState, SkeletonState
from crafter.constants import ActionT
from .utils import find_player, find_all_objects_for_type, find_object_in_state
from crafter.testing_helpers import (
    player_utils,
    world_utils,
)
from crafter.functional_env import initial_state
from crafter import objects
from ...typing_utils import implements
from ..core import SymbolicTransition

from crafter.state_export import CowState
from dataclasses import dataclass
from typing import Optional
from crafter.functional_env import transition as crafter_transition_fn
from .utils import MAP_ACTION_TO_INDEX
import random
import functools
from typing import Sequence
from crafter.constants import ActionT as CrafterAction


@dataclass
class GoalChecked:
    value: bool
    message: Optional[str] = None

    def __bool__(self) -> bool:
        return self.value


def _check_steps_taken(
    transitions: list[SymbolicTransition[WorldState, CrafterAction]], max_steps: int
) -> GoalChecked:
    if transitions[-1].next_metadata.step_count >= max_steps:
        return GoalChecked(
            True, f"Have taken {transitions[-1].next_metadata.step_count} steps"
        )
    return GoalChecked(False, f"Have not yet taken {max_steps} steps")


class Scenario(Protocol):
    """Protocol for scenario definitions."""

    @property
    def name(self) -> str:
        """The name of this scenario."""
        ...

    def get_initial_state(self) -> WorldState:
        """Creates and returns the specific starting WorldState for this scenario."""
        ...

    def policy(self, state: WorldState) -> ActionT:
        """Returns the action to take at the given state."""
        ...

    @property
    def max_steps(self) -> int:
        """Returns the maximum number of steps to take in the scenario."""
        ...

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        """Returns True if the scenario has been achieved."""
        ...


T = TypeVar("T", bound=Callable[..., GoalChecked])


def require_max_steps(goal_test_method: T) -> T:
    """
    Decorator that ensures max_steps have been reached before calling the original goal_test method.

    This decorator should be applied to goal_test methods of scenarios that need to run
    until max_steps have been reached. It will return False if max_steps haven't been
    reached yet, otherwise it will call the original goal_test method.
    """

    @functools.wraps(goal_test_method)
    def wrapper(
        self: Scenario, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        # Check if we've reached max_steps
        if not (target_steps_goal := _check_steps_taken(transitions, self.max_steps)):
            return target_steps_goal

        # If we have reached max_steps, call the original goal_test method
        return goal_test_method(self, transitions)

    return cast(T, wrapper)


@dataclass
class ScenarioRunResult:
    scenario: Scenario
    transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    goal_test: GoalChecked
    run_for_steps: int


def run_scenarios(scenarios: Sequence[Scenario]) -> list[ScenarioRunResult]:
    results: list[ScenarioRunResult] = []

    for scenario in scenarios:
        state = scenario.get_initial_state()
        transitions: list[SymbolicTransition[WorldState, CrafterAction]] = []
        # In order to satisfy the type checker that goal_test and step are bound,
        # we will initialize them here and then re-assign them in the loop.
        goal_test = GoalChecked(False, "Scenario not started")
        step = 0
        for step in range(scenario.max_steps):
            action = scenario.policy(state)
            next_state, _ = crafter_transition_fn(
                copy.deepcopy(state), MAP_ACTION_TO_INDEX[action]
            )

            transition = SymbolicTransition[WorldState, CrafterAction](
                state, action, next_state
            )
            transitions.append(transition)
            state = next_state

            goal_test = scenario.goal_test(transitions)
            if goal_test:
                break

        results.append(ScenarioRunResult(scenario, transitions, goal_test, step + 1))

    return results


class CraftWoodenPickaxeScenario:
    """Scenario for testing crafting a wooden pickaxe."""

    @property
    def name(self) -> str:
        return "craft_wooden_pickaxe"

    def get_initial_state(self) -> WorldState:
        """
        Creates a temporary environment, configures it to the desired
        starting conditions, and returns the resulting WorldState.
        """
        view = (9, 9)
        state = initial_state(area=(9, 9), view=view, seed=1)
        world = reconstruct_world_from_state(state)

        player = find_player(world)
        player_utils.set_player_position(player, (5, 5))
        player_utils.set_player_facing(player, (0, 1))
        world_utils.set_tile_material(world, (5, 6), "table")
        player_utils.set_player_inventory_item(player, "wood", 2)
        player_utils.set_player_inventory_item(player, "wood_pickaxe", 0)

        return export_world_state(world, view=view, step_count=0)

    def policy(self, state: WorldState) -> ActionT:
        """Returns the action to take at the given state."""
        return "make_wood_pickaxe"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        if transitions[0].prev_metadata.player.inventory.wood_pickaxe != 0:
            return GoalChecked(
                False,
                "Player already has a wooden pickaxe",
            )

        post_pickaxe_count = transitions[0].next_metadata.player.inventory.wood_pickaxe
        if post_pickaxe_count != 1:
            return GoalChecked(
                False,
                f"Player has {post_pickaxe_count} wooden pickaxes instead of 1",
            )

        return GoalChecked(True, "Player has a wooden pickaxe")

    @property
    def max_steps(self) -> int:
        """Returns the maximum number of steps to take in the scenario."""
        return 1


implements(Scenario)(CraftWoodenPickaxeScenario)


class ZombieDefeatScenario:
    def __init__(self, max_steps: int = 5):
        self.target_zombie: Optional[ZombieState] = None
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "zombie_defeat"

    def get_initial_state(self) -> WorldState:
        view = (9, 9)
        state = initial_state(area=(9, 9), view=view, seed=1)
        world = reconstruct_world_from_state(state)

        # Clear all the other tiles around the world to be grass
        for x in range(view[0]):
            for y in range(view[1]):
                world_utils.set_tile_material(world, (x, y), "grass")

        player = find_player(world)
        player_utils.set_player_position(player, (5, 5))

        # Clear all entities from the world (except the player)
        for obj in world.objects:
            if isinstance(obj, objects.Player):
                continue
            world.remove(obj)

        # Add a zombie to the right of the player
        zombie = objects.Zombie(world, (6, 5), player)
        world.add(zombie)

        # Make the player face the zombie
        player_utils.set_player_facing(player, (1, 0))

        state = export_world_state(world, view=view, step_count=0)
        # Find the zombie's entity id
        self.target_zombie = find_object_in_state(
            state,
            entity_id=zombie.entity_id,
            entity_type=ZombieState,
        )
        return state

    def policy(self, state: WorldState) -> ActionT:
        return "do"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:

        assert self.target_zombie is not None

        zombie_present: list[ZombieState | None] = []

        for t in transitions:
            zombie = find_object_in_state(
                t.next_metadata,
                entity_id=self.target_zombie.entity_id,
                entity_type=ZombieState,
            )
            zombie_present.append(zombie)

        if (final_zombie := zombie_present[-1]) is not None:
            return GoalChecked(
                False,
                f"Zombie(entity_id={final_zombie.entity_id}, health={final_zombie.health}) not defeated",
            )

        return GoalChecked(True, "Zombie defeated")


implements(Scenario)(ZombieDefeatScenario)


class DefeatSkeletonScenario:
    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self.target_skeleton: Optional[SkeletonState] = None
        self.logger = logger.bind(scenario="defeat_skeleton")

    @property
    def name(self) -> str:
        return "defeat_skeleton"

    def get_initial_state(self) -> WorldState:
        view = (9, 9)
        state = initial_state(area=(9, 9), view=view, seed=1)
        world = reconstruct_world_from_state(state)

        player = find_player(world)
        player_utils.set_player_position(player, (5, 5))

        # Clear all the other tiles around the world to be grass
        for x in range(view[0]):
            for y in range(view[1]):
                world_utils.set_tile_material(world, (x, y), "grass")

        # Add a skeleton to the right of the player
        skeleton = objects.Skeleton(world, (6, 5), player)
        world.add(skeleton)

        # Make the player face the skeleton
        player_utils.set_player_facing(player, (1, 0))

        state = export_world_state(world, view=view, step_count=0)
        # Find the skeleton's entity id
        self.target_skeleton = find_object_in_state(
            state,
            entity_id=skeleton.entity_id,
            entity_type=SkeletonState,
        )
        return state

    def policy(self, state: WorldState) -> ActionT:
        # Find the target skeleton
        assert self.target_skeleton is not None
        target_skeleton = find_object_in_state(
            state,
            entity_id=self.target_skeleton.entity_id,
            entity_type=SkeletonState,
        )

        if target_skeleton is None:
            self.logger.warning("No skeleton found to attack")
            return "noop"

        self.logger.debug(f"Target skeleton: {target_skeleton}")

        player = state.player

        # Check if skeleton is adjacent
        distance = player.distance(target_skeleton.position)

        if distance == 1:
            # Adjacent - check if facing the skeleton
            if player.facing + player.position == target_skeleton.position:
                self.logger.debug("Facing skeleton - attacking")
                return "do"  # Attack
            else:
                self.logger.debug("Not facing skeleton - turning towards")
                # Turn towards skeleton
                direction = player.toward(target_skeleton.position)
                if direction[0] == -1:
                    return "move_left"
                elif direction[0] == 1:
                    return "move_right"
                elif direction[1] == -1:
                    return "move_up"
                else:  # direction[1] == 1
                    return "move_down"
        else:
            self.logger.debug("Not adjacent - moving towards skeleton")
            # Not adjacent - move towards skeleton
            direction = player.toward(target_skeleton.position)
            if direction[0] == -1:
                return "move_left"
            elif direction[0] == 1:
                return "move_right"
            elif direction[1] == -1:
                return "move_up"
            else:  # direction[1] == 1
                return "move_down"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        assert self.target_skeleton is not None
        skeleton = find_object_in_state(
            transitions[-1].next_metadata,
            entity_id=self.target_skeleton.entity_id,
            entity_type=SkeletonState,
        )

        skeleton_present: list[SkeletonState | None] = []

        for t in transitions:
            skeleton = find_object_in_state(
                t.next_metadata,
                entity_id=self.target_skeleton.entity_id,
                entity_type=SkeletonState,
            )
            skeleton_present.append(skeleton)

        if (final_skeleton := skeleton_present[-1]) is not None:
            return GoalChecked(
                False,
                f"Skeleton(entity_id={final_skeleton.entity_id}, health={final_skeleton.health}) not defeated",
            )
        return GoalChecked(True, "Skeleton defeated")


implements(Scenario)(DefeatSkeletonScenario)


class EatCowScenario:
    def __init__(self, max_steps: int = 15):
        self.max_steps = max_steps
        self.target_cow: Optional[CowState] = None
        self.logger = logger.bind(scenario="eat_cow")

    @property
    def name(self) -> str:
        return "eat_cow"

    def get_initial_state(self) -> WorldState:
        view = (9, 9)
        state = initial_state(area=(9, 9), view=view, seed=1)
        world = reconstruct_world_from_state(state)

        player = find_player(world)
        player_utils.set_player_position(player, (5, 5))

        # Clear all the other tiles around the world to be grass
        for x in range(view[0]):
            for y in range(view[1]):
                world_utils.set_tile_material(world, (x, y), "grass")

        # Add a cow to the right of the player
        cow = objects.Cow(world, (6, 5))
        world.add(cow)

        # Make the player face the cow
        player_utils.set_player_facing(player, (1, 0))

        state = export_world_state(world, view=view, step_count=0)
        # Find the cow's entity id
        self.target_cow = find_object_in_state(
            state,
            entity_id=cow.entity_id,
            entity_type=CowState,
        )
        return state

    def policy(self, state: WorldState) -> ActionT:
        # Find the target cow
        assert self.target_cow is not None
        target_cow = find_object_in_state(
            state,
            entity_id=self.target_cow.entity_id,
            entity_type=CowState,
        )

        if target_cow is None:
            self.logger.warning("No skeleton found to attack")
            return "noop"

        self.logger.debug(f"Target cow: {target_cow}")

        player = state.player

        # Check if cow is adjacent
        distance = player.distance(target_cow.position)

        if distance == 1:
            # Adjacent - check if facing the cow
            if player.facing + player.position == target_cow.position:
                self.logger.debug("Facing cow - attacking")
                return "do"  # Attack
            else:
                self.logger.debug("Not facing cow - turning towards")
                # Turn towards cow
                direction = player.toward(target_cow.position)
                if direction[0] == -1:
                    return "move_left"
                elif direction[0] == 1:
                    return "move_right"
                elif direction[1] == -1:
                    return "move_up"
                else:  # direction[1] == 1
                    return "move_down"
        else:
            self.logger.debug("Not adjacent - moving towards cow")
            # Not adjacent - move towards cow
            direction = player.toward(target_cow.position)
            if direction[0] == -1:
                return "move_left"
            elif direction[0] == 1:
                return "move_right"
            elif direction[1] == -1:
                return "move_up"
            else:  # direction[1] == 1
                return "move_down"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        assert self.target_cow is not None
        cow = find_object_in_state(
            transitions[-1].next_metadata,
            entity_id=self.target_cow.entity_id,
            entity_type=CowState,
        )

        cow_present: list[CowState | None] = []
        for t in transitions:
            cow = find_object_in_state(
                t.next_metadata,
                entity_id=self.target_cow.entity_id,
                entity_type=CowState,
            )
            cow_present.append(cow)

        if (final_cow := cow_present[-1]) is not None:
            return GoalChecked(
                False,
                f"Cow(entity_id={final_cow.entity_id}, health={final_cow.health}) not eaten",
            )
        return GoalChecked(True, "Cow eaten")


class CowMovementScenario:
    """Scenario for testing cow movement behavior."""

    def __init__(self, max_steps: int = 30):
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "cow_movement"

    def get_initial_state(self) -> WorldState:
        """
        Creates a temporary environment with a cow near the player.
        """
        view = (9, 9)
        state = initial_state(area=(9, 9), view=view, seed=1)
        world = reconstruct_world_from_state(state)

        # Clear all the other tiles around the world to be grass
        for x in range(view[0]):
            for y in range(view[1]):
                world_utils.set_tile_material(world, (x, y), "grass")

        player = find_player(world)
        player_utils.set_player_position(player, (5, 5))

        # Clear all entities from the world (except the player)
        for obj in world.objects:
            if isinstance(obj, objects.Player):
                continue
            world.remove(obj)

        # Add a cow near the player
        cow = objects.Cow(world, (6, 6))
        world.add(cow)

        return export_world_state(world, view=view, step_count=0)

    def policy(self, state: WorldState) -> ActionT:
        """Returns the action to take at the given state."""
        return "noop"

    @require_max_steps
    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        # Check if the cow has moved from its iniital position.
        cows = find_all_objects_for_type(
            transitions[0].prev_metadata, entity_type=CowState
        )
        if not cows:
            return GoalChecked(False, "No cow found in initial state")

        initial_cow = cows[0]

        # Now we step through the transitions and check if the cow has moved.
        for transition in transitions:
            cow = find_object_in_state(
                transition.next_metadata,
                entity_id=initial_cow.entity_id,
                entity_type=CowState,
            )
            if not cow:
                return GoalChecked(False, "Cow disappeared from the world")

            if cow.position != initial_cow.position:
                return GoalChecked(True, "Cow moved")

        return GoalChecked(False, f"Cow did not move in {len(transitions)} steps")


implements(Scenario)(CowMovementScenario)


class RandomMovementPolicy:
    """Policy that returns random movement actions."""

    def __init__(self, policy_seed: int):
        self.policy_rng = random.Random(policy_seed)
        self.movement_actions: list[ActionT] = [
            "move_left",
            "move_right",
            "move_up",
            "move_down",
        ]

    def __call__(self, state: WorldState) -> ActionT:
        """Returns a random movement action."""
        return self.policy_rng.choice(self.movement_actions)


class RandomMovementScenario:
    """Scenario for testing random movement behavior."""

    def __init__(self, max_steps: int = 30, policy_seed: int = 1):
        self.max_steps = max_steps
        self.policy = RandomMovementPolicy(policy_seed)

    @property
    def name(self) -> str:
        return "random_movement"

    def get_initial_state(self) -> WorldState:
        """
        Creates a simple initial state with default world configuration.
        """
        view = (9, 9)
        state = initial_state(area=(9, 9), view=view, seed=1)
        return state

    def policy(self, state: WorldState) -> ActionT:
        """Returns a random movement action."""
        return self.policy(state)

    @require_max_steps
    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        # Check if the player has moved from the initial position
        if not transitions:
            return GoalChecked(False, "No transitions occurred")

        initial_position = transitions[0].prev_metadata.player.position

        for transition in transitions:
            if transition.next_metadata.player.position != initial_position:
                return GoalChecked(True, "Player moved from initial position")

        return GoalChecked(False, f"Player did not move in {len(transitions)} steps")


implements(Scenario)(RandomMovementScenario)
