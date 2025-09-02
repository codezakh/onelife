"""
Tests for the Crafter creation synthesizer.

This module tests that the creation synthesizer can generate expert functions
from state transitions involving object lifecycle events (creation, deletion, replacement)
in the Crafter environment.
"""

import pytest
from crafter.state_export import (
    WorldState,
    CowState,
    ZombieState,
    Position,
    PlayerState,
    Achievements,
    Inventory,
)

from distant_sunburn.poe_world.crafter.creation_synthesizer import (
    CrafterCreationSynthesizer,
)
from distant_sunburn.poe_world.core import (
    SymbolicTransition,
)
from loguru import logger


@pytest.mark.asyncio
async def test_zombie_defeat_transition():
    """Test that zombie defeat transitions are correctly detected."""
    synthesizer = CrafterCreationSynthesizer()

    # Create initial state with zombies
    initial_state = WorldState(
        size=(100, 100),
        chunk_size=(16, 16),
        view=(8, 8),
        daylight=0.5,
        objects=[
            PlayerState(
                entity_id=0,
                position=Position(x=50, y=50),
                health=10,
                name="player",
                facing=Position(x=1, y=0),
                action="idle",
                sleeping=False,
                achievements=Achievements(),
                inventory=Inventory(),
                thirst=0.0,
                hunger=0.0,
                fatigue=0.0,
                recover=0.0,
                last_health=10,
            ),
            ZombieState(
                entity_id=1,
                position=Position(x=30, y=40),
                health=20,
                name="zombie",
                cooldown=0,
            ),
            ZombieState(
                entity_id=2,
                position=Position(x=31, y=41),
                health=20,
                name="zombie",
                cooldown=0,
            ),
        ],
        entity_id_counter_state=3,
        chunks=[],
        player=PlayerState(
            entity_id=0,
            position=Position(x=50, y=50),
            health=10,
            name="player",
            facing=Position(x=1, y=0),
            action="idle",
            sleeping=False,
            achievements=Achievements(),
            inventory=Inventory(),
            thirst=0.0,
            hunger=0.0,
            fatigue=0.0,
            recover=0.0,
            last_health=10,
        ),
        materials=[[]],
        step_count=0,
        serialized_random_state="",
        event_bus=[],
    )

    # Create next state with zombies removed
    next_state = WorldState(
        size=(100, 100),
        chunk_size=(16, 16),
        view=(8, 8),
        daylight=0.5,
        objects=[
            PlayerState(
                entity_id=0,
                position=Position(x=50, y=50),
                health=10,
                name="player",
                facing=Position(x=1, y=0),
                action="idle",
                sleeping=False,
                achievements=Achievements(),
                inventory=Inventory(),
                thirst=0.0,
                hunger=0.0,
                fatigue=0.0,
                recover=0.0,
                last_health=10,
            )
        ],
        entity_id_counter_state=3,
        chunks=[],
        player=PlayerState(
            entity_id=0,
            position=Position(x=50, y=50),
            health=10,
            name="player",
            facing=Position(x=1, y=0),
            action="idle",
            sleeping=False,
            achievements=Achievements(),
            inventory=Inventory(),
            thirst=0.0,
            hunger=0.0,
            fatigue=0.0,
            recover=0.0,
            last_health=10,
        ),
        materials=[[]],
        step_count=1,
        serialized_random_state="",
        event_bus=[],
    )

    transition = SymbolicTransition(
        prev_metadata=initial_state, action="do", next_metadata=next_state
    )

    with logger.contextualize(object_type="zombie"):
        # Test that the synthesizer can generate experts for zombie defeat
        experts = await synthesizer.synthesize_experts(
            transitions=[transition], object_type="zombie"
        )

    # Should generate at least one expert for zombie defeat
    assert len(experts) >= 1, "Should generate experts for zombie defeat transition"


@pytest.mark.asyncio
async def test_cow_spawning_transition():
    """Test that cow spawning transitions are correctly detected."""
    synthesizer = CrafterCreationSynthesizer()

    # Create initial state without cows
    initial_state = WorldState(
        size=(100, 100),
        chunk_size=(16, 16),
        view=(8, 8),
        daylight=0.5,
        objects=[
            PlayerState(
                entity_id=0,
                position=Position(x=50, y=50),
                health=10,
                name="player",
                facing=Position(x=1, y=0),
                action="idle",
                sleeping=False,
                achievements=Achievements(),
                inventory=Inventory(),
                thirst=0.0,
                hunger=0.0,
                fatigue=0.0,
                recover=0.0,
                last_health=10,
            )
        ],
        entity_id_counter_state=1,
        chunks=[],
        player=PlayerState(
            entity_id=0,
            position=Position(x=50, y=50),
            health=10,
            name="player",
            facing=Position(x=1, y=0),
            action="idle",
            sleeping=False,
            achievements=Achievements(),
            inventory=Inventory(),
            thirst=0.0,
            hunger=0.0,
            fatigue=0.0,
            recover=0.0,
            last_health=10,
        ),
        materials=[[]],
        step_count=0,
        serialized_random_state="",
        event_bus=[],
    )

    # Create next state with cow added
    next_state = WorldState(
        size=(100, 100),
        chunk_size=(16, 16),
        view=(8, 8),
        daylight=0.5,
        objects=[
            PlayerState(
                entity_id=0,
                position=Position(x=50, y=50),
                health=10,
                name="player",
                facing=Position(x=1, y=0),
                action="idle",
                sleeping=False,
                achievements=Achievements(),
                inventory=Inventory(),
                thirst=0.0,
                hunger=0.0,
                fatigue=0.0,
                recover=0.0,
                last_health=10,
            ),
            CowState(entity_id=1, position=Position(x=60, y=60), health=10, name="cow"),
        ],
        entity_id_counter_state=2,
        chunks=[],
        player=PlayerState(
            entity_id=0,
            position=Position(x=50, y=50),
            health=10,
            name="player",
            facing=Position(x=1, y=0),
            action="idle",
            sleeping=False,
            achievements=Achievements(),
            inventory=Inventory(),
            thirst=0.0,
            hunger=0.0,
            fatigue=0.0,
            recover=0.0,
            last_health=10,
        ),
        materials=[[]],
        step_count=1,
        serialized_random_state="",
        event_bus=[],
    )

    transition = SymbolicTransition(
        prev_metadata=initial_state, action="noop", next_metadata=next_state
    )

    # Test that the synthesizer can generate experts for cow spawning
    experts = await synthesizer.synthesize_experts(
        transitions=[transition], object_type="cow"
    )

    # Should generate at least one expert for cow spawning
    assert len(experts) >= 1, "Should generate experts for cow spawning transition"
