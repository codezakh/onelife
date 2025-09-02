"""
Tests for Crafter scenarios that verify actual behavior and outcomes.
"""

import pytest

from distant_sunburn.evaluator.crafter.scenarios import (
    CraftWoodenPickaxeScenario,
    CowMovementScenario,
    RandomMovementScenario,
    ZombieDefeatScenario,
)
from distant_sunburn.evaluator.crafter.scenarios import run_scenarios


class TestScenarioRunner:
    def test_transitions_correct(self):
        scenario = CraftWoodenPickaxeScenario()
        results = run_scenarios([scenario])

        # Check that the step count has incremented
        assert results[0].transitions[0].prev_metadata.step_count == 0
        assert results[0].transitions[0].next_metadata.step_count == 1


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
    """Test that the random movement scenario results in player movement."""
    # Arrange
    scenario = RandomMovementScenario(max_steps=100, policy_seed=1)

    # Act
    results = run_scenarios([scenario])

    # Assert - Verify the goal test succeeded (player moved)
    assert results[
        0
    ].goal_test, "Player should have moved during random movement scenario"


def test_zombie_defeat_scenario():
    """Test that the zombie defeat scenario results in the zombie being defeated."""
    # Arrange
    scenario = ZombieDefeatScenario(max_steps=5)

    # Act
    results = run_scenarios([scenario])

    # Assert - Verify the goal test succeeded (zombie defeated)
    assert results[
        0
    ].goal_test, "Zombie should be defeated during zombie defeat scenario"
