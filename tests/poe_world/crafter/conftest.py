"""
Shared fixtures and utilities for PoE-World Crafter tests.

This module provides common test setup and scenario creation functions
to eliminate duplication across test files.
"""

import pytest
import numpy as np
from crafter.state_export import WorldState
from crafter.functional_env import (
    initial_state,
    reconstruct_world_from_state,
    export_world_state,
    transition,
)
from crafter.constants import ActionT
from crafter import objects, constants

from distant_sunburn.poe_world.core import SymbolicTransition


@pytest.fixture
def simple_world_state() -> WorldState:
    """Create a simple world state with just a player."""
    return initial_state(area=(5, 5), view=(3, 3), seed=42)


@pytest.fixture
def cow_attack_scenario() -> SymbolicTransition[WorldState]:
    """
    Create a test scenario where a player attacks a cow.

    This creates a state where:
    1. Player is at position (2, 2) facing right
    2. Cow is at position (3, 2) with 5 health
    3. Player has a wood sword
    4. After 'do' action, cow health should decrease
    """
    # Create initial state
    view = (9, 9)
    initial_state_obj = initial_state(area=(9, 9), view=view, seed=42)
    world = reconstruct_world_from_state(initial_state_obj)

    # Find and configure the player
    player = _find_player(world)
    _configure_player_for_attack(player)

    # Add a cow in front of player
    cow = objects.Cow(world, (3, 2))
    world.add(cow)

    # Export the modified initial state
    initial_state_obj = export_world_state(world, view=view, step_count=0)

    # Use the functional transition to get the next state
    action_index = constants.actions.index("do")
    next_state_obj, _ = transition(initial_state_obj, action_index)

    return SymbolicTransition(
        prev_metadata=initial_state_obj,
        action="do",
        next_metadata=next_state_obj,
    )


@pytest.fixture
def zombie_attack_scenario() -> SymbolicTransition[WorldState]:
    """
    Create a test scenario where a zombie attacks a player.

    This creates a state where:
    1. Player is at position (2, 2) facing right
    2. Zombie is at position (3, 2) with 5 health
    3. After 'do' action, player health should decrease
    """
    # Create initial state
    view = (9, 9)
    initial_state_obj = initial_state(area=(9, 9), view=view, seed=42)
    world = reconstruct_world_from_state(initial_state_obj)

    # Find and configure the player
    player = _find_player(world)
    _configure_player_for_attack(player)

    # Add a zombie in front of player
    zombie = objects.Zombie(world, (3, 2), player)
    world.add(zombie)

    # Export the modified initial state
    initial_state_obj = export_world_state(world, view=view, step_count=0)

    # Use the functional transition to get the next state
    action_index = constants.actions.index("do")
    next_state_obj, _ = transition(initial_state_obj, action_index)

    return SymbolicTransition(
        prev_metadata=initial_state_obj,
        action="do",
        next_metadata=next_state_obj,
    )


@pytest.fixture
def mixed_entity_world() -> WorldState:
    """
    Create a world state with multiple entity types for testing filtering.

    Contains:
    - 1 player
    - 1 cow at (3, 3)
    - 1 zombie at (5, 5)  # Changed from (4, 4) to avoid position conflicts
    """
    view = (9, 9)
    state = initial_state(area=(9, 9), view=view, seed=42)
    world = reconstruct_world_from_state(state)

    # Find the player
    player = _find_player(world)

    # Clear existing cows and zombies to ensure we have exactly what we want
    from crafter.objects import Cow, Zombie

    to_remove = [obj for obj in world.objects if isinstance(obj, (Cow, Zombie))]
    for obj in to_remove:
        world.remove(obj)

    # Add exactly one cow and one zombie at specific positions
    cow = objects.Cow(world, (3, 3))
    zombie = objects.Zombie(world, (5, 5), player)
    world.add(cow)
    world.add(zombie)

    return export_world_state(world, view=view, step_count=0)


def _find_player(world) -> objects.Player:
    """Find the player object in the world."""
    for obj in world.objects:
        if isinstance(obj, objects.Player):
            return obj
    raise ValueError("No player found in world")


def _configure_player_for_attack(player: objects.Player) -> None:
    """Configure a player for attack scenarios."""
    player.pos = np.array((2, 2))
    player.facing = (1, 0)  # Facing right
    player.inventory["wood_sword"] = 1  # Has weapon


def create_movement_transition(
    action: str, start_pos: tuple[int, int] = (2, 2), end_pos: tuple[int, int] = (3, 2)
) -> SymbolicTransition[WorldState]:
    """
    Create a simple movement transition for testing.

    Args:
        action: The action to perform (e.g., "move_right")
        start_pos: Starting position of the player
        end_pos: Expected ending position of the player

    Returns:
        A transition representing the movement
    """
    # Create initial state
    view = (9, 9)
    initial_state_obj = initial_state(area=(9, 9), view=view, seed=42)
    world = reconstruct_world_from_state(initial_state_obj)

    # Configure player
    player = _find_player(world)
    player.pos = np.array(start_pos)

    # Export initial state
    initial_state_obj = export_world_state(world, view=view, step_count=0)

    # Create expected next state
    world = reconstruct_world_from_state(initial_state_obj)
    player = _find_player(world)
    player.pos = np.array(end_pos)
    next_state_obj = export_world_state(world, view=view, step_count=1)

    return SymbolicTransition(
        prev_metadata=initial_state_obj,
        action=action,
        next_metadata=next_state_obj,
    )
