"""
Scenario definitions for Crafter evaluation.
"""

from typing import Protocol
from crafter.functional_env import (
    reconstruct_world_from_state,
    export_world_state,
)
from crafter.state_export import WorldState
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


@dataclass
class GoalChecked:
    value: bool
    message: Optional[str] = None

    def __bool__(self) -> bool:
        return self.value


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
        self, transitions: list[SymbolicTransition[WorldState]]
    ) -> GoalChecked:
        """Returns True if the scenario has been achieved."""
        ...


@dataclass
class ScenarioRunResult:
    scenario: Scenario
    transitions: list[SymbolicTransition[WorldState]]
    goal_test: GoalChecked
    run_for_steps: int


def run_scenarios(scenarios: list[Scenario]) -> list[ScenarioRunResult]:
    results: list[ScenarioRunResult] = []
    for scenario in scenarios:
        state = scenario.get_initial_state()
        transitions: list[SymbolicTransition[WorldState]] = []
        # In order to satisfy the type checker that goal_test and step are bound,
        # we will initialize them here and then re-assign them in the loop.
        goal_test = GoalChecked(False, "Scenario not started")
        step = -1
        for step in range(scenario.max_steps):
            action = scenario.policy(state)
            next_state, _ = crafter_transition_fn(state, MAP_ACTION_TO_INDEX[action])
            state = next_state
            transition = SymbolicTransition(state, action, next_state)
            transitions.append(transition)

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
        self, transitions: list[SymbolicTransition[WorldState]]
    ) -> GoalChecked:
        # Check the the first transition to see if the player has a wooden pickaxe
        # If so, the scenario has been achieved.
        return GoalChecked(
            transitions[-1].prev_metadata.player.inventory.wood_pickaxe > 0,
            "Player has a wooden pickaxe",
        )

    @property
    def max_steps(self) -> int:
        """Returns the maximum number of steps to take in the scenario."""
        return 1


implements(Scenario)(CraftWoodenPickaxeScenario)


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

    def goal_test(
        self, transitions: list[SymbolicTransition[WorldState]]
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
