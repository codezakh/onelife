"""
Tests for Crafter scenarios that verify actual behavior and outcomes.
"""

import pytest

from distant_sunburn.evaluator.crafter import scenarios

from distant_sunburn.evaluator.crafter.scenarios import (
    CraftWoodenPickaxeScenario,
    CowMovementScenario,
    RandomMovementScenario,
    ZombieDefeatScenario,
    DefeatSkeletonScenario,
    EatCowScenario,
    CollectCoalScenario,
    UnsuccessfulCollectCoalScenario,
    CollectDiamondScenario,
    UnsuccessfulCollectDiamondScenario,
    CollectIronScenario,
    UnsuccessfulCollectIronScenario,
    CollectStoneScenario,
    UnsuccessfulCollectStoneScenario,
)
from distant_sunburn.evaluator.crafter.scenarios import run_scenarios
from distant_sunburn.evaluator.crafter.scenarios import Scenario
from typing import Type


class TestScenarioRunner:
    def test_transitions_correct(self):
        scenario = CraftWoodenPickaxeScenario()
        results = run_scenarios([scenario])

        # Check that the step count has incremented
        assert results[0].transitions[0].prev_metadata.step_count == 0
        assert results[0].transitions[0].next_metadata.step_count == 1


# def test_craft_wooden_pickaxe_scenario():
#     """Test that running the scenario results in a wooden pickaxe being crafted."""
#     # Arrange
#     scenario = CraftWoodenPickaxeScenario()

#     # Act - Get initial state and run the scenario
#     initial_state = scenario.get_initial_state()

#     # Verify initial state doesn't have pickaxe
#     assert (
#         initial_state.player.inventory.wood_pickaxe == 0
#     ), "Should not have pickaxe initially"

#     results = run_scenarios([scenario])

#     # Verify the goal test succeeded
#     assert results[0].goal_test


# def test_cow_movement_scenario():
#     """Test that the scenario creates a world with a cow present."""
#     # Arrange
#     scenario = CowMovementScenario()

#     # Act
#     initial_state = scenario.get_initial_state()

#     # Assert - Verify there's a cow in the world
#     cows = [obj for obj in initial_state.objects if obj.name == "cow"]
#     assert len(cows) == 1

#     results = run_scenarios([scenario])

#     # Verify the goal test succeeded
#     assert results[0].goal_test


# def test_random_movement_scenario():
#     """Test that the random movement scenario results in player movement."""
#     # Arrange
#     scenario = RandomMovementScenario(max_steps=100, policy_seed=1)

#     # Act
#     results = run_scenarios([scenario])

#     # Assert - Verify the goal test succeeded (player moved)
#     assert results[
#         0
#     ].goal_test, "Player should have moved during random movement scenario"


# def test_zombie_defeat_scenario():
#     """Test that the zombie defeat scenario results in the zombie being defeated."""
#     # Arrange
#     scenario = ZombieDefeatScenario(max_steps=5)

#     # Act
#     results = run_scenarios([scenario])

#     # Assert - Verify the goal test succeeded (zombie defeated)
#     assert results[
#         0
#     ].goal_test, "Zombie should be defeated during zombie defeat scenario"


# def test_defeat_skeleton_scenario():
#     """Test that the skeleton defeat scenario results in the skeleton being defeated."""
#     # Arrange
#     scenario = DefeatSkeletonScenario(max_steps=10)

#     # Act
#     results = run_scenarios([scenario])

#     # Assert - Verify the goal test succeeded (skeleton defeated)
#     assert results[
#         0
#     ].goal_test, "Skeleton should be defeated during skeleton defeat scenario"


# def test_eat_cow_scenario():
#     """Test that the cow eat scenario results in the cow being eaten."""
#     # Arrange
#     scenario = EatCowScenario()

#     # Act
#     results = run_scenarios([scenario])

#     # Assert - Verify the goal test succeeded (cow eaten)
#     assert results[0].goal_test, "Cow should be eaten during cow eat scenario"


# def test_collect_coal_scenario():
#     """Test that the coal collect scenario results in the coal being collected."""
#     # Arrange
#     scenario = CollectCoalScenario()

#     # Act
#     results = run_scenarios([scenario])

#     # Assert - Verify the goal test succeeded (coal collected)
#     assert results[0].goal_test, "Coal should be collected during coal collect scenario"


# def test_unsuccessful_collect_coal_scenario():
#     """Test that the coal collect scenario results in the coal being collected."""
#     # Arrange
#     scenario = UnsuccessfulCollectCoalScenario()

#     # Act
#     results = run_scenarios([scenario])

#     # Assert - Verify the goal test succeeded (coal not collected)
#     assert results[
#         0
#     ].goal_test, "Coal should not be collected during coal collect scenario"


# def test_collect_diamond_scenario():
#     """Test that the diamond collect scenario results in the diamond being collected."""
#     # Arrange
#     scenario = CollectDiamondScenario()

#     # Act
#     results = run_scenarios([scenario])

#     # Assert - Verify the goal test succeeded (diamond collected)
#     assert results[
#         0
#     ].goal_test, "Diamond should be collected during diamond collect scenario"


# def test_unsuccessful_collect_diamond_scenario():
#     """Test that the diamond collect scenario results in the diamond being collected."""
#     # Arrange
#     scenario = UnsuccessfulCollectDiamondScenario()

#     # Act
#     results = run_scenarios([scenario])

#     # Assert - Verify the goal test succeeded (diamond not collected)
#     assert results[
#         0
#     ].goal_test, "Diamond should not be collected during diamond collect scenario"


# def test_collect_iron_scenario():
#     """Test that the iron collect scenario results in the iron being collected."""
#     # Arrange
#     scenario = CollectIronScenario()

#     # Act
#     results = run_scenarios([scenario])

#     # Assert - Verify the goal test succeeded (iron collected)
#     assert results[0].goal_test, "Iron should be collected during iron collect scenario"


# def test_unsuccessful_collect_iron_scenario():
#     """Test that the iron collect scenario results in the iron being collected."""
#     # Arrange
#     scenario = UnsuccessfulCollectIronScenario()

#     # Act
#     results = run_scenarios([scenario])

#     # Assert - Verify the goal test succeeded (iron not collected)
#     assert results[
#         0
#     ].goal_test, "Iron should not be collected during iron collect scenario"


# def test_collect_stone_scenario():
#     """Test that the stone collect scenario results in the stone being collected."""
#     # Arrange
#     scenario = CollectStoneScenario()

#     # Act
#     results = run_scenarios([scenario])

#     # Assert - Verify the goal test succeeded (stone collected)
#     assert results[
#         0
#     ].goal_test, "Stone should be collected during stone collect scenario"


# def test_unsuccessful_collect_stone_scenario():
#     """Test that the stone collect scenario results in the stone being collected."""
#     # Arrange
#     scenario = UnsuccessfulCollectStoneScenario()

#     # Act
#     results = run_scenarios([scenario])

#     # Assert - Verify the goal test succeeded (stone not collected)
#     assert results[
#         0
#     ].goal_test, "Stone should not be collected during stone collect scenario"


@pytest.mark.parametrize(
    "scenario_cls",
    [
        scenarios.CraftWoodenPickaxeScenario,
        scenarios.CowMovementScenario,
        scenarios.RandomMovementScenario,
        scenarios.ZombieDefeatScenario,
        scenarios.DefeatSkeletonScenario,
        scenarios.EatCowScenario,
        scenarios.CollectCoalScenario,
        scenarios.UnsuccessfulCollectCoalScenario,
        scenarios.CollectDiamondScenario,
        scenarios.UnsuccessfulCollectDiamondScenario,
        scenarios.CollectIronScenario,
        scenarios.UnsuccessfulCollectIronScenario,
        scenarios.CollectStoneScenario,
        scenarios.UnsuccessfulCollectStoneScenario,
        scenarios.CollectDrinkScenario,
        scenarios.CollectWoodScenario,
        scenarios.EatPlantScenario,
        scenarios.UnsuccessfulEatPlantScenario,
        scenarios.CraftIronPickaxeScenario,
        scenarios.UnsuccessfulCraftIronPickaxeScenario,
        scenarios.CraftIronSwordScenario,
        scenarios.UnsuccessfulCraftIronSwordScenario,
        scenarios.CraftStonePickaxeScenario,
        scenarios.UnsuccessfulCraftStonePickaxeScenario,
        scenarios.CraftStoneSwordScenario,
        scenarios.UnsuccessfulCraftStoneSwordScenario,
        scenarios.CraftWoodenPickaxeScenario,
        scenarios.UnsuccessfulCraftWoodenPickaxeScenario,
        scenarios.CraftWoodenSwordScenario,
        scenarios.UnsuccessfulCraftWoodenSwordScenario,
        scenarios.PlaceFurnaceScenario,
        scenarios.UnsuccessfulPlaceFurnaceScenario,
        scenarios.PlacePlantScenario,
        scenarios.UnsuccessfulPlacePlantScenario,
        scenarios.PlaceStoneScenario,
        scenarios.UnsuccessfulPlaceStoneScenario,
        scenarios.PlaceTableScenario,
        scenarios.UnsuccessfulPlaceTableScenario,
        scenarios.WakeUpScenario,
        scenarios.PlayerDeathScenario,
    ],
)
def test_collection_scenario(scenario_cls: Type[Scenario]):
    """Test that the collection scenario results in the item being collected."""
    # Arrange
    scenario = scenario_cls()

    # Act
    results = run_scenarios([scenario])

    # Assert - Verify the goal test succeeded (item collected)
    assert results[0].goal_test
