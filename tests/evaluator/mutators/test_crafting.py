from crafter.state_export import WorldState
from crafter.functional_env import (
    initial_state,
    reconstruct_world_from_state,
    export_world_state,
    transition,
)
import copy
from crafter.testing_helpers import player_utils, world_utils
from distant_sunburn.evaluator.crafter.utils import MAP_ACTION_TO_INDEX
import pytest
from crafter.constants import ActionT
from crafter import engine as crafter_engine
from distant_sunburn.evaluator.crafter.mutators.crafting import (
    CraftIllegalItemMutator,
    CRAFTING_ACTIONS,
)
from distant_sunburn.evaluator.crafter.utils import find_player
import numpy as np
from crafter.constants import actions


@pytest.fixture
def crafting_station_state() -> tuple[WorldState, crafter_engine.World]:
    view = (9, 9)
    state = initial_state(area=(9, 9), view=view, seed=1)
    world = reconstruct_world_from_state(state)

    player = find_player(world)
    player_utils.set_player_position(player, (5, 5))

    # Clear all the other tiles around the world to be grass
    for x in range(view[0]):
        for y in range(view[1]):
            world_utils.set_tile_material(world, (x, y), "grass")

    # Add a crafting table next to the player
    world_utils.set_tile_material(world, player.pos + np.array([1, 0]), "table")

    # Add a furnace next to the player
    world_utils.set_tile_material(world, player.pos + np.array([1, 1]), "furnace")

    # Give the player required resources
    player_utils.set_player_inventory_item(player, "wood", 3)
    player_utils.set_player_inventory_item(player, "coal", 3)
    player_utils.set_player_inventory_item(player, "iron", 3)

    return export_world_state(world, view=view, step_count=0), world


@pytest.mark.parametrize("action", CRAFTING_ACTIONS)
def test_crafting_illegal_item_mutator(
    crafting_station_state: tuple[WorldState, crafter_engine.World], action: ActionT
):
    state = crafting_station_state[0]

    mutator = CraftIllegalItemMutator()

    assert mutator.precondition(state, action)

    true_next_state, _ = transition(copy.deepcopy(state), MAP_ACTION_TO_INDEX[action])

    mutated_next_state = mutator(state, action)

    # Now we compare the two inventories. The mutated inventory
    # should have an item that the true next state does not, and
    # additionally, the mutation will not have consumed any resources.

    assert mutated_next_state.player.inventory != true_next_state.player.inventory

    # We do this check just to confirm that the __eq__ method is working
    # correctly.
    assert true_next_state.player.inventory == copy.deepcopy(
        true_next_state.player.inventory
    ), (
        "The __eq__ method on inventories should assert equality when the inventories are the same"
        " but this is not the case; this test is invalid."
    )


@pytest.mark.parametrize("action", set(actions) - CRAFTING_ACTIONS)
def test_mutator_inactive_on_non_crafting_actions(
    crafting_station_state: tuple[WorldState, crafter_engine.World], action: ActionT
):
    state = crafting_station_state[0]

    mutator = CraftIllegalItemMutator()

    assert not mutator.precondition(state, action)
