"""
Tests for Crafter scenarios that verify actual behavior and outcomes.
"""

import pytest

from distant_sunburn.evaluator.crafter.scenarios import (
    CraftWoodenPickaxeScenario,
    CowMovementScenario,
)
from distant_sunburn.evaluator.crafter.components import RandomMovementPolicy
from distant_sunburn.evaluator.crafter.scenarios import run_scenarios


class TestCraftWoodenPickaxeScenario:
    """Test the wooden pickaxe crafting scenario."""

    def test_scenario_creates_wooden_pickaxe(self):
        """Test that running the scenario results in a wooden pickaxe being crafted."""
        # Arrange
        scenario = CraftWoodenPickaxeScenario()

        # Act - Get initial state and run the scenario
        initial_state = scenario.get_initial_state()

        # Verify initial state doesn't have pickaxe
        assert (
            initial_state.player.inventory.wood_pickaxe == 0
        ), "Should not have pickaxe initially"

        results = run_scenarios([scenario])

        # Verify the goal test succeeded
        assert results[0].goal_test


class TestCowMovementScenario:
    """Test the cow movement scenario."""

    def test_scenario_has_cow_in_world(self):
        """Test that the scenario creates a world with a cow present."""
        # Arrange
        scenario = CowMovementScenario()

        # Act
        initial_state = scenario.get_initial_state()

        # Assert - Verify there's a cow in the world
        cows = [obj for obj in initial_state.objects if obj.name == "cow"]
        assert len(cows) == 1

        results = run_scenarios([scenario])

        # Verify the goal test succeeded
        assert results[0].goal_test


def test_random_movement_scenario():
    scenario = RandomMovementPolicy(policy_seed=1, num_transitions=100)

    transitions = scenario()

    # Assert that the player moved from the initial position
    initial_position = transitions[0].prev_metadata.player.position

    for t in transitions:
        if t.prev_metadata.player.position != initial_position:
            # Test passed
            break
    else:
        # Fail the test if the player didn't move at all
        pytest.fail(f"Player did not move in {len(transitions)} actions")
