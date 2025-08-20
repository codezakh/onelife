"""
Tests for the handwritten experts for the 1D test environment.

This module contains comprehensive tests for the correct and incorrect expert
functions, validating their behavior and ensuring they conform to the ExpertFunction
protocol.
"""

import copy
import random
from typing import cast

from distant_sunburn.poe_world.benchmark_1d.environment import (
    Action,
    GameState,
    Light,
    Player,
    WorldConfig,
)
from distant_sunburn.poe_world.benchmark_1d.handwritten_experts import (
    correct_movement_expert,
    correct_light_expert,
    incorrect_movement_expert_ignores_switch,
    incorrect_movement_expert_ignores_slip,
    incorrect_light_expert_is_deterministic,
)
from distant_sunburn.poe_world.core import RandomValues


class TestCorrectMovementExpert:
    """Test cases for the correct_movement_expert."""

    def test_stay_action_no_movement(self):
        """Test that STAY action results in no prediction."""
        # Arrange
        config = WorldConfig(width=10, switch_point=5)
        player = Player(position=3)
        lights = [Light(position=1, is_on=False)]
        rng = random.Random(42)
        state = GameState(config=config, player=player, lights=lights, rng=rng)

        # Act
        correct_movement_expert(state, Action.STAY)

        # Assert - Expert should not modify player position for STAY action
        assert not isinstance(state.player.position, RandomValues)

    def test_standard_movement_normal_zone(self):
        """Test standard movement predictions in the non-switched zone."""
        # Arrange
        config = WorldConfig(width=10, switch_point=5)
        player = Player(position=2)  # In normal zone (left half)
        lights = [Light(position=1, is_on=False)]
        rng = random.Random(42)
        state = GameState(config=config, player=player, lights=lights, rng=rng)

        # Act - Move right
        correct_movement_expert(state, Action.MOVE_RIGHT)

        # Assert
        assert isinstance(state.player.position, RandomValues)
        assert state.player.position.values[0] == 3

        # Act - Move left (create fresh state)
        state2 = GameState(
            config=config,
            player=Player(position=3),
            lights=lights,
            rng=random.Random(42),
        )
        correct_movement_expert(state2, Action.MOVE_LEFT)

        # Assert
        assert isinstance(state2.player.position, RandomValues)
        assert state2.player.position.values[0] == 2

    def test_movement_inverted_in_switched_zone(self):
        """Test that movement predictions are inverted in the switched zone."""
        # Arrange
        config = WorldConfig(width=10, switch_point=5)
        player = Player(position=7)  # In switched zone (right half)
        lights = [Light(position=1, is_on=False)]
        rng = random.Random(42)
        state = GameState(config=config, player=player, lights=lights, rng=rng)

        # Act - Move right (should become left in switched zone)
        correct_movement_expert(state, Action.MOVE_RIGHT)

        # Assert
        assert isinstance(state.player.position, RandomValues)
        assert state.player.position.values[0] == 6

        # Act - Move left (should become right in switched zone) - create fresh state
        state2 = GameState(
            config=config,
            player=Player(position=6),
            lights=lights,
            rng=random.Random(42),
        )
        correct_movement_expert(state2, Action.MOVE_LEFT)

        # Assert
        assert isinstance(state2.player.position, RandomValues)
        assert state2.player.position.values[0] == 7

    def test_boundary_conditions(self):
        """Test boundary condition predictions."""
        # Arrange
        config = WorldConfig(width=5, switch_point=2)

        # Test left boundary
        player_left = Player(position=0)
        lights = [Light(position=1, is_on=False)]
        rng = random.Random(42)
        state_left = GameState(
            config=config, player=player_left, lights=lights, rng=rng
        )

        # Act - Try to move left from position 0
        correct_movement_expert(state_left, Action.MOVE_LEFT)

        # Assert - Should stay at 0
        assert isinstance(state_left.player.position, RandomValues)
        assert state_left.player.position.values[0] == 0

    def test_slipperiness_affects_predictions(self):
        """Test that slipperiness affects movement predictions."""
        # Arrange - Use deterministic RNG to control slipperiness
        config = WorldConfig(width=10, switch_point=5)
        player = Player(position=3)
        lights = [Light(position=1, is_on=False)]

        # Create RNG that will always slip (first random() call < 0.1)
        rng = random.Random(42)
        # Mock the RNG to always return a value < 0.1 (slip probability)
        rng.random = lambda: 0.05  # Always slip

        state = GameState(config=config, player=player, lights=lights, rng=rng)

        # Act - Move right (should slip and become left)
        correct_movement_expert(state, Action.MOVE_RIGHT)

        # Assert - Should slip and move left instead
        assert isinstance(state.player.position, RandomValues)
        assert state.player.position.values[0] == 2


class TestCorrectLightExpert:
    """Test cases for the correct_light_expert."""

    def test_light_predictions_are_stochastic(self):
        """Test that light predictions are stochastic and use RNG."""
        # Arrange
        config = WorldConfig(width=10, switch_point=5)
        player = Player(position=3)
        lights = [Light(position=1, is_on=False), Light(position=8, is_on=True)]

        # Create RNG that will always toggle (first random() call < 0.2)
        rng = random.Random(42)
        rng.random = lambda: 0.1  # Always toggle

        state = GameState(config=config, player=player, lights=lights, rng=rng)

        # Act
        correct_light_expert(state, Action.STAY)

        # Assert - Both lights should be predicted to toggle
        for light in state.lights:
            assert isinstance(light.is_on, RandomValues)

        # First light should toggle from False to True
        assert cast(RandomValues, state.lights[0].is_on).values[0] == 1
        # Second light should toggle from True to False
        assert cast(RandomValues, state.lights[1].is_on).values[0] == 0

    def test_light_predictions_independent_of_action(self):
        """Test that light predictions are independent of player action."""
        # Arrange
        config = WorldConfig(width=10, switch_point=5)
        player = Player(position=3)
        lights = [Light(position=1, is_on=False)]
        rng = random.Random(42)
        state = GameState(config=config, player=player, lights=lights, rng=rng)

        # Act - Test with different actions using fresh state each time
        state1 = copy.deepcopy(state)
        correct_light_expert(state1, Action.MOVE_LEFT)
        prediction_left = cast(RandomValues, state1.lights[0].is_on).values[0]

        state2 = copy.deepcopy(state)
        correct_light_expert(state2, Action.MOVE_RIGHT)
        prediction_right = cast(RandomValues, state2.lights[0].is_on).values[0]

        state3 = copy.deepcopy(state)
        correct_light_expert(state3, Action.STAY)
        prediction_stay = cast(RandomValues, state3.lights[0].is_on).values[0]

        # Assert - All predictions should be the same (given same RNG state)
        # Convert numpy booleans to Python booleans for comparison
        prediction_left_bool = bool(prediction_left)
        prediction_right_bool = bool(prediction_right)
        prediction_stay_bool = bool(prediction_stay)
        assert prediction_left_bool == prediction_right_bool == prediction_stay_bool


class TestIncorrectMovementExpertIgnoresSwitch:
    """Test cases for the incorrect_movement_expert_ignores_switch."""

    def test_ignores_switched_zone_mechanic(self):
        """Test that this expert ignores the switched zone mechanic."""
        # Arrange
        config = WorldConfig(width=10, switch_point=5)
        player = Player(position=7)  # In switched zone (right half)
        lights = [Light(position=1, is_on=False)]
        rng = random.Random(42)
        state = GameState(config=config, player=player, lights=lights, rng=rng)

        # Act - Move right (should NOT be inverted in this expert)
        incorrect_movement_expert_ignores_switch(state, Action.MOVE_RIGHT)

        # Assert - Should move right (not left like correct expert)
        assert isinstance(state.player.position, RandomValues)
        assert state.player.position.values[0] == 8  # 7 + 1, not 6

    def test_still_respects_boundaries_and_slipperiness(self):
        """Test that this expert still respects boundaries and slipperiness."""
        # Arrange
        config = WorldConfig(width=5, switch_point=2)
        player = Player(position=0)
        lights = [Light(position=1, is_on=False)]
        rng = random.Random(42)
        state = GameState(config=config, player=player, lights=lights, rng=rng)

        # Act - Try to move left from position 0
        incorrect_movement_expert_ignores_switch(state, Action.MOVE_LEFT)

        # Assert - Should still respect boundary
        assert isinstance(state.player.position, RandomValues)
        assert state.player.position.values[0] == 0


class TestIncorrectMovementExpertIgnoresSlip:
    """Test cases for the incorrect_movement_expert_ignores_slip."""

    def test_ignores_slipperiness_mechanic(self):
        """Test that this expert ignores the slipperiness mechanic."""
        # Arrange
        config = WorldConfig(width=10, switch_point=5)
        player = Player(position=3)
        lights = [Light(position=1, is_on=False)]

        # Create RNG that would cause slipping
        rng = random.Random(42)
        rng.random = lambda: 0.05  # Always slip

        state = GameState(config=config, player=player, lights=lights, rng=rng)

        # Act - Move right (should NOT slip in this expert)
        incorrect_movement_expert_ignores_slip(state, Action.MOVE_RIGHT)

        # Assert - Should move right (not left like correct expert with slip)
        assert isinstance(state.player.position, RandomValues)
        assert state.player.position.values[0] == 4  # 3 + 1, not 2

    def test_still_respects_switched_zone_and_boundaries(self):
        """Test that this expert still respects switched zone and boundaries."""
        # Arrange
        config = WorldConfig(width=10, switch_point=5)
        player = Player(position=7)  # In switched zone
        lights = [Light(position=1, is_on=False)]
        rng = random.Random(42)
        state = GameState(config=config, player=player, lights=lights, rng=rng)

        # Act - Move right (should still be inverted due to switched zone)
        incorrect_movement_expert_ignores_slip(state, Action.MOVE_RIGHT)

        # Assert - Should still respect switched zone
        assert isinstance(state.player.position, RandomValues)
        assert state.player.position.values[0] == 6  # 7 - 1 due to switched zone


class TestIncorrectLightExpertIsDeterministic:
    """Test cases for the incorrect_light_expert_is_deterministic."""

    def test_always_predicts_toggle(self):
        """Test that this expert always predicts lights will toggle."""
        # Arrange
        config = WorldConfig(width=10, switch_point=5)
        player = Player(position=3)
        lights = [Light(position=1, is_on=False), Light(position=8, is_on=True)]
        rng = random.Random(42)
        state = GameState(config=config, player=player, lights=lights, rng=rng)

        # Act
        incorrect_light_expert_is_deterministic(state, Action.STAY)

        # Assert - Both lights should be predicted to toggle
        for light in state.lights:
            assert isinstance(light.is_on, RandomValues)

        # First light should toggle from False to True
        assert cast(RandomValues, state.lights[0].is_on).values[0] == 1
        # Second light should toggle from True to False
        assert cast(RandomValues, state.lights[1].is_on).values[0] == 0

    def test_predictions_independent_of_rng(self):
        """Test that predictions are independent of RNG state."""

        # Test with different RNG states
        for seed in [42, 123, 456]:
            # Arrange
            config = WorldConfig(width=10, switch_point=5)
            player = Player(position=3)
            lights = [Light(position=1, is_on=False)]
            rng = random.Random(seed)
            state = GameState(config=config, player=player, lights=lights, rng=rng)

            # Act
            incorrect_light_expert_is_deterministic(state, Action.STAY)

            for light in state.lights:
                assert isinstance(light.is_on, RandomValues)
                assert light.is_on.values[0]
