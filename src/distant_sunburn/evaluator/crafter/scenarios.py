"""
Scenario definitions for Crafter evaluation.
"""

import copy
from typing_extensions import assert_never
import numpy as np
from loguru import logger
from typing import Protocol, Callable, TypeVar, cast
from crafter.functional_env import (
    reconstruct_world_from_state,
    export_world_state,
)
from crafter.state_export import WorldState, ZombieState, SkeletonState, PlantState
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
from crafter.constants import MaterialT, CraftingStationT
from crafter import objects as crafter_objects
from crafter import engine as crafter_engine


def create_collection_scenario_base_state(
    target_material: MaterialT,
) -> tuple[crafter_engine.World, crafter_objects.Player, tuple[int, int]]:
    view = (9, 9)
    state = initial_state(area=(9, 9), view=view, seed=1)
    world = reconstruct_world_from_state(state)

    player = find_player(world)
    player_utils.set_player_position(player, (5, 5))

    # Clear all the other tiles around the world to be grass
    for x in range(view[0]):
        for y in range(view[1]):
            world_utils.set_tile_material(world, (x, y), "grass")

    # Set the tile to the right of the player to the target material
    world_utils.set_tile_material(world, (6, 5), target_material)

    # Make the player face the target material
    player_utils.set_player_facing(player, (1, 0))

    return world, player, view


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


class CollectCoalScenario:
    def __init__(self, max_steps: int = 1):
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "collect_coal"

    def get_initial_state(self) -> WorldState:
        world, player, view = create_collection_scenario_base_state("coal")

        # Give the player a wood pickaxe
        player_utils.set_player_inventory_item(player, "wood_pickaxe", 1)

        state = export_world_state(world, view=view, step_count=0)
        return state

    def policy(self, state: WorldState) -> ActionT:
        return "do"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:

        first_transition = transitions[0]

        next_state = first_transition.next_metadata

        if next_state.player.inventory.coal == 1:
            return GoalChecked(True, "Coal collected")
        return GoalChecked(False, "Coal not collected")


implements(Scenario)(CollectCoalScenario)


class UnsuccessfulCollectCoalScenario:
    def __init__(self, max_steps: int = 1):
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "collect_coal"

    def get_initial_state(self) -> WorldState:
        world, player, view = create_collection_scenario_base_state("coal")

        # Make sure a player has no pickaxe strong enough to collect the coal
        player_utils.set_player_inventory_item(player, "wood_pickaxe", 0)
        player_utils.set_player_inventory_item(player, "stone_pickaxe", 0)
        player_utils.set_player_inventory_item(player, "iron_pickaxe", 0)

        state = export_world_state(world, view=view, step_count=0)
        return state

    def policy(self, state: WorldState) -> ActionT:
        return "do"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:

        first_transition = transitions[0]

        next_state = first_transition.next_metadata

        if next_state.player.inventory.coal == 0:
            return GoalChecked(True, "Coal not collected")
        return GoalChecked(False, "Coal collected")


implements(Scenario)(UnsuccessfulCollectCoalScenario)


class CollectDiamondScenario:
    def __init__(self, max_steps: int = 1):
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "collect_diamond"

    def get_initial_state(self) -> WorldState:
        world, player, view = create_collection_scenario_base_state("diamond")

        # Give the player an iron pickaxe
        player_utils.set_player_inventory_item(player, "iron_pickaxe", 1)

        state = export_world_state(world, view=view, step_count=0)
        return state

    def policy(self, state: WorldState) -> ActionT:
        return "do"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        first_transition = transitions[0]
        next_state = first_transition.next_metadata
        if next_state.player.inventory.diamond == 1:
            return GoalChecked(True, "Diamond collected")
        return GoalChecked(False, "Diamond not collected")


implements(Scenario)(CollectDiamondScenario)


class UnsuccessfulCollectDiamondScenario:
    def __init__(self, max_steps: int = 1):
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "collect_diamond"

    def get_initial_state(self) -> WorldState:
        world, player, view = create_collection_scenario_base_state("diamond")

        # Make sure a player has no pickaxe strong enough to collect the diamond
        player_utils.set_player_inventory_item(player, "wood_pickaxe", 1)
        player_utils.set_player_inventory_item(player, "stone_pickaxe", 1)
        player_utils.set_player_inventory_item(player, "iron_pickaxe", 0)

        state = export_world_state(world, view=view, step_count=0)
        return state

    def policy(self, state: WorldState) -> ActionT:
        return "do"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        first_transition = transitions[0]
        next_state = first_transition.next_metadata
        if next_state.player.inventory.diamond == 0:
            return GoalChecked(True, "Diamond not collected")
        return GoalChecked(False, "Diamond collected")


class CollectIronScenario:
    def __init__(self, max_steps: int = 1):
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "collect_iron"

    def get_initial_state(self) -> WorldState:
        world, player, view = create_collection_scenario_base_state("iron")

        # Give the player a stone pickaxe
        player_utils.set_player_inventory_item(player, "stone_pickaxe", 1)

        state = export_world_state(world, view=view, step_count=0)
        return state

    def policy(self, state: WorldState) -> ActionT:
        return "do"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        first_transition = transitions[0]
        next_state = first_transition.next_metadata
        if next_state.player.inventory.iron == 1:
            return GoalChecked(True, "Iron collected")
        return GoalChecked(False, "Iron not collected")


implements(Scenario)(CollectIronScenario)


class UnsuccessfulCollectIronScenario:
    def __init__(self, max_steps: int = 1):
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "collect_iron"

    def get_initial_state(self) -> WorldState:
        world, player, view = create_collection_scenario_base_state("iron")

        # Make sure a player has no pickaxe strong enough to collect the iron
        player_utils.set_player_inventory_item(player, "wood_pickaxe", 1)
        player_utils.set_player_inventory_item(player, "stone_pickaxe", 0)
        player_utils.set_player_inventory_item(player, "iron_pickaxe", 0)

        state = export_world_state(world, view=view, step_count=0)
        return state

    def policy(self, state: WorldState) -> ActionT:
        return "do"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        first_transition = transitions[0]
        next_state = first_transition.next_metadata
        if next_state.player.inventory.iron == 0:
            return GoalChecked(True, "Iron not collected")
        return GoalChecked(False, "Iron collected")


implements(Scenario)(UnsuccessfulCollectIronScenario)


class CollectStoneScenario:
    def __init__(self, max_steps: int = 1):
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "collect_stone"

    def get_initial_state(self) -> WorldState:
        world, player, view = create_collection_scenario_base_state("stone")

        # Give the player a wood pickaxe
        player_utils.set_player_inventory_item(player, "wood_pickaxe", 1)

        state = export_world_state(world, view=view, step_count=0)
        return state

    def policy(self, state: WorldState) -> ActionT:
        return "do"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        first_transition = transitions[0]
        next_state = first_transition.next_metadata
        if next_state.player.inventory.stone == 1:
            return GoalChecked(True, "Stone collected")
        return GoalChecked(False, "Stone not collected")


implements(Scenario)(CollectStoneScenario)


class UnsuccessfulCollectStoneScenario:
    def __init__(self, max_steps: int = 1):
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "collect_stone"

    def get_initial_state(self) -> WorldState:
        world, player, view = create_collection_scenario_base_state("stone")

        # Make sure a player has no pickaxe strong enough to collect the stone
        player_utils.set_player_inventory_item(player, "wood_pickaxe", 0)
        player_utils.set_player_inventory_item(player, "stone_pickaxe", 0)
        player_utils.set_player_inventory_item(player, "iron_pickaxe", 0)

        state = export_world_state(world, view=view, step_count=0)
        return state

    def policy(self, state: WorldState) -> ActionT:
        return "do"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        first_transition = transitions[0]
        next_state = first_transition.next_metadata
        if next_state.player.inventory.stone == 0:
            return GoalChecked(True, "Stone not collected")
        return GoalChecked(False, "Stone collected")


implements(Scenario)(UnsuccessfulCollectStoneScenario)


class CollectDrinkScenario:
    def __init__(self, max_steps: int = 1):
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "collect_drink"

    def get_initial_state(self) -> WorldState:
        world, player, view = create_collection_scenario_base_state("water")

        player_utils.set_player_inventory_item(player, "drink", 0)

        state = export_world_state(world, view=view, step_count=0)
        return state

    def policy(self, state: WorldState) -> ActionT:
        return "do"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        first_transition = transitions[0]
        next_state = first_transition.next_metadata
        if next_state.player.inventory.drink == 1:
            return GoalChecked(True, "Drink collected")
        return GoalChecked(False, "Drink not collected")


implements(Scenario)(CollectDrinkScenario)


class CollectWoodScenario:
    def __init__(self, max_steps: int = 1):
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "collect_wood"

    def get_initial_state(self) -> WorldState:
        world, player, view = create_collection_scenario_base_state("tree")

        state = export_world_state(world, view=view, step_count=0)
        return state

    def policy(self, state: WorldState) -> ActionT:
        return "do"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        first_transition = transitions[0]
        next_state = first_transition.next_metadata
        if next_state.player.inventory.wood == 1:
            return GoalChecked(True, "Wood collected")
        return GoalChecked(False, "Wood not collected")


implements(Scenario)(CollectWoodScenario)


class EatPlantScenario:
    def __init__(self, max_steps: int = 1):
        self.max_steps = max_steps
        self.target_plant: Optional[PlantState] = None

    @property
    def name(self) -> str:
        return "eat_plant"

    def get_initial_state(self) -> WorldState:
        world, player, view = create_collection_scenario_base_state("grass")

        # Add a plant to the right of the player
        plant = objects.Plant(world, player.pos + np.array([1, 0]))
        plant.grown = 301  # Make it ripe
        world.add(plant)

        # Ensure the player has no food
        player_utils.set_player_inventory_item(player, "food", 0)

        state = export_world_state(world, view=view, step_count=0)
        self.target_plant = find_object_in_state(
            state,
            entity_id=plant.entity_id,
            entity_type=PlantState,
        )
        return state

    def policy(self, state: WorldState) -> ActionT:
        return "do"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        first_transition = transitions[0]
        next_state = first_transition.next_metadata
        food_collected = next_state.player.inventory.food == 4
        assert self.target_plant is not None
        plant = find_object_in_state(
            next_state,
            entity_id=self.target_plant.entity_id,
            entity_type=PlantState,
        )
        assert plant is not None
        plant_reset = plant.grown == 1

        match (food_collected, plant_reset):
            case (True, True):
                return GoalChecked(True, "Food collected and plant reset")
            case (True, False):
                return GoalChecked(False, "Food collected but plant not reset")
            case (False, True):
                return GoalChecked(False, "Food not collected but plant reset")
            case (False, False):
                return GoalChecked(False, "Food not collected and plant not reset")
            case _:
                assert_never(food_collected, plant_reset)


implements(Scenario)(EatPlantScenario)


class UnsuccessfulEatPlantScenario:
    def __init__(self, max_steps: int = 1):
        self.max_steps = max_steps
        self.target_plant: Optional[PlantState] = None

    @property
    def name(self) -> str:
        return "eat_plant"

    def get_initial_state(self) -> WorldState:
        world, player, view = create_collection_scenario_base_state("grass")

        # Add a plant to the right of the player
        plant = objects.Plant(world, player.pos + np.array([1, 0]))
        plant.grown = 100  # Make it not ripe
        world.add(plant)

        # Ensure the player has no food
        player_utils.set_player_inventory_item(player, "food", 0)

        state = export_world_state(world, view=view, step_count=0)
        self.target_plant = find_object_in_state(
            state,
            entity_id=plant.entity_id,
            entity_type=PlantState,
        )
        return state

    def policy(self, state: WorldState) -> ActionT:
        return "do"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        first_transition = transitions[0]
        next_state = first_transition.next_metadata
        food_collected = next_state.player.inventory.food == 4
        assert self.target_plant is not None
        plant = find_object_in_state(
            next_state,
            entity_id=self.target_plant.entity_id,
            entity_type=PlantState,
        )
        assert plant is not None

        plant_reset = plant.grown == 1

        match (food_collected, plant_reset):
            case (True, True):
                return GoalChecked(
                    False, "Food collected and plant reset, indicating successful eat"
                )
            case (True, False):
                return GoalChecked(
                    False,
                    "Food collected but plant not reset, indicating unsuccessful eat",
                )
            case (False, True):
                return GoalChecked(
                    False,
                    "Food not collected but plant reset, indicating unsuccessful eat",
                )
            case (False, False):
                return GoalChecked(
                    True,
                    "Food not collected and plant not reset, indicating unsuccessful eat",
                )
            case _:
                assert_never(food_collected, plant_reset)


implements(Scenario)(UnsuccessfulEatPlantScenario)


class CraftIronPickaxeScenario:
    def __init__(self, max_steps: int = 1):
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "craft_iron_pickaxe"

    def get_initial_state(self) -> WorldState:
        world, player, view = create_collection_scenario_base_state("table")

        # Set the player to have the required resources
        player_utils.set_player_inventory_item(player, "wood", 1)
        player_utils.set_player_inventory_item(player, "coal", 1)
        player_utils.set_player_inventory_item(player, "iron", 1)

        # Add a furnace to the left of the player
        world_utils.set_tile_material(world, player.pos - np.array([1, 0]), "furnace")

        state = export_world_state(world, view=view, step_count=0)
        return state

    def policy(self, state: WorldState) -> ActionT:
        return "make_iron_pickaxe"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        first_transition = transitions[0]
        next_state = first_transition.next_metadata
        if next_state.player.inventory.iron_pickaxe == 1:
            return GoalChecked(True, "Iron pickaxe crafted")
        return GoalChecked(False, "Iron pickaxe not crafted")


implements(Scenario)(CraftIronPickaxeScenario)


class UnsuccessfulCraftIronPickaxeScenario:
    def __init__(self, max_steps: int = 1):
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "craft_iron_pickaxe"

    def get_initial_state(self) -> WorldState:
        world, player, view = create_collection_scenario_base_state("table")

        # Ensure the player is missing a required resource
        player_utils.set_player_inventory_item(player, "wood", 0)
        player_utils.set_player_inventory_item(player, "coal", 1)
        player_utils.set_player_inventory_item(player, "iron", 1)

        # Add a furnace to the left of the player
        world_utils.set_tile_material(world, player.pos - np.array([1, 0]), "furnace")

        state = export_world_state(world, view=view, step_count=0)
        return state

    def policy(self, state: WorldState) -> ActionT:
        return "make_iron_pickaxe"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        first_transition = transitions[0]
        next_state = first_transition.next_metadata
        if next_state.player.inventory.iron_pickaxe == 1:
            return GoalChecked(False, "Iron pickaxe crafted")
        return GoalChecked(True, "Iron pickaxe not crafted")


class CraftIronSwordScenario:
    def __init__(self, max_steps: int = 1):
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "craft_iron_sword"

    def get_initial_state(self) -> WorldState:
        world, player, view = create_collection_scenario_base_state("table")

        # Set the player to have the required resources
        player_utils.set_player_inventory_item(player, "wood", 1)
        player_utils.set_player_inventory_item(player, "coal", 1)
        player_utils.set_player_inventory_item(player, "iron", 1)

        # Add a furnace to the left of the player
        world_utils.set_tile_material(world, player.pos - np.array([1, 0]), "furnace")

        state = export_world_state(world, view=view, step_count=0)
        return state

    def policy(self, state: WorldState) -> ActionT:
        return "make_iron_sword"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        first_transition = transitions[0]
        next_state = first_transition.next_metadata
        if next_state.player.inventory.iron_sword == 1:
            return GoalChecked(True, "Iron sword crafted")
        return GoalChecked(False, "Iron sword not crafted")


implements(Scenario)(CraftIronSwordScenario)


class UnsuccessfulCraftIronSwordScenario:
    def __init__(self, max_steps: int = 1):
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "craft_iron_sword"

    def get_initial_state(self) -> WorldState:
        world, player, view = create_collection_scenario_base_state("table")

        # Ensure the player is missing a required resource
        player_utils.set_player_inventory_item(player, "wood", 0)
        player_utils.set_player_inventory_item(player, "coal", 1)
        player_utils.set_player_inventory_item(player, "iron", 1)

        # Add a furnace to the left of the player
        world_utils.set_tile_material(world, player.pos - np.array([1, 0]), "furnace")

        state = export_world_state(world, view=view, step_count=0)
        return state

    def policy(self, state: WorldState) -> ActionT:
        return "make_iron_sword"

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState, CrafterAction]]
    ) -> GoalChecked:
        first_transition = transitions[0]
        next_state = first_transition.next_metadata
        if next_state.player.inventory.iron_sword == 1:
            return GoalChecked(False, "Iron sword crafted")
        return GoalChecked(True, "Iron sword not crafted")


implements(Scenario)(UnsuccessfulCraftIronSwordScenario)
