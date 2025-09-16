import pytest
from crafter.state_export import WorldState
from crafter.functional_env import (
    initial_state,
    reconstruct_world_from_state,
    export_world_state,
)
from crafter.testing_helpers import player_utils, world_utils
from crafter import engine as crafter_engine
from crafter import objects
from crafter.state_export import CowState
from distant_sunburn.evaluator.crafter.mutators.entity_position import (
    EntityPositionMutator,
)
from distant_sunburn.evaluator.crafter.utils import find_all_objects_for_type


@pytest.fixture
def world_with_cow() -> tuple[WorldState, crafter_engine.World]:
    """Create a simple world with one cow."""
    view = (9, 9)
    state = initial_state(area=(9, 9), view=view, seed=1)
    world = reconstruct_world_from_state(state)

    player = world.objects[0]  # Player is first object
    player_utils.set_player_position(player, (5, 5))

    # Clear all tiles to grass
    for x in range(view[0]):
        for y in range(view[1]):
            world_utils.set_tile_material(world, (x, y), "grass")

    # Add a cow
    world_utils.add_object_to_world(world, objects.Cow, (3, 3))

    return export_world_state(world, view=view, step_count=0), world


def test_entity_position_mutator_moves_entities_by_at_least_2_tiles(world_with_cow):
    """Test that entities are moved by at least 2 tiles."""
    state, _ = world_with_cow

    # Find the cow using utils
    cows = find_all_objects_for_type(state, CowState)
    assert len(cows) == 1, "Should have exactly one cow"
    cow = cows[0]
    original_pos = cow.position

    # Apply mutator
    mutator = EntityPositionMutator()
    mutated_state = mutator(state, "noop")

    # Find the cow in mutated state using utils
    mutated_cows = find_all_objects_for_type(mutated_state, CowState)
    assert len(mutated_cows) == 1, "Should have exactly one cow after mutation"
    mutated_cow = mutated_cows[0]

    # Calculate distance moved
    distance = abs(mutated_cow.position.x - original_pos.x) + abs(
        mutated_cow.position.y - original_pos.y
    )

    assert distance >= 2, f"Cow only moved {distance} tiles, expected at least 2"
