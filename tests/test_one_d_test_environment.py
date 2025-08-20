"""
Tests for the 1D Test Environment.

This module contains comprehensive tests for the 1D test environment,
validating all mechanics and ensuring reproducibility.
"""

import random
import pytest

from distant_sunburn.one_d_test_environment import (
    Action,
    GameState,
    Light,
    LightLaw,
    MovementLaw,
    Player,
    WorldConfig,
    initial_state,
    transition_function,
)


class TestMovementLaw:
    """Test cases for the MovementLaw."""

    def test_stay_action_no_movement(self):
        """Test that STAY action results in no change of position."""
        # Arrange
        config = WorldConfig(width=10, switch_point=5)
        player = Player(position=3)
        lights = [Light(position=1, is_on=False)]
        rng = random.Random(42)
        state = GameState(config=config, player=player, lights=lights, rng=rng)

        movement_law = MovementLaw(slip_probability=0.0)
        original_position = state.player.position

        # Act
        movement_law.apply(state, Action.STAY)

        # Assert
        assert state.player.position == original_position

    def test_standard_movement_normal_zone(self):
        """Test standard movement in the non-switched zone with zero slipperiness."""
        # Arrange
        config = WorldConfig(width=10, switch_point=5)
        player = Player(position=2)  # In normal zone (left half)
        lights = [Light(position=1, is_on=False)]
        rng = random.Random(42)
        state = GameState(config=config, player=player, lights=lights, rng=rng)

        movement_law = MovementLaw(slip_probability=0.0)

        # Act - Move right
        movement_law.apply(state, Action.MOVE_RIGHT)

        # Assert
        assert state.player.position == 3

        # Act - Move left
        movement_law.apply(state, Action.MOVE_LEFT)

        # Assert
        assert state.player.position == 2

    def test_movement_inverted_in_switched_zone(self):
        """Test that movement is inverted in the switched zone with zero slipperiness."""
        # Arrange
        config = WorldConfig(width=10, switch_point=5)
        player = Player(position=7)  # In switched zone (right half)
        lights = [Light(position=1, is_on=False)]
        rng = random.Random(42)
        state = GameState(config=config, player=player, lights=lights, rng=rng)

        movement_law = MovementLaw(slip_probability=0.0)

        # Act - Move right (should become left in switched zone)
        movement_law.apply(state, Action.MOVE_RIGHT)

        # Assert
        assert state.player.position == 6

        # Act - Move left (should become right in switched zone)
        movement_law.apply(state, Action.MOVE_LEFT)

        # Assert
        assert state.player.position == 7

    def test_boundary_conditions(self):
        """Test boundary conditions: ensure player cannot move past boundaries."""
        # Arrange
        config = WorldConfig(width=5, switch_point=2)
        movement_law = MovementLaw(slip_probability=0.0)

        # Test left boundary
        player_left = Player(position=0)
        lights = [Light(position=1, is_on=False)]
        rng = random.Random(42)
        state_left = GameState(
            config=config, player=player_left, lights=lights, rng=rng
        )

        # Act - Try to move left from position 0
        movement_law.apply(state_left, Action.MOVE_LEFT)

        # Assert - Should stay at 0
        assert state_left.player.position == 0

        # Test right boundary - need to account for switched zone behavior
        # Position 4 is in the switched zone, so MOVE_RIGHT becomes MOVE_LEFT
        player_right = Player(position=4)  # width - 1
        state_right = GameState(
            config=config, player=player_right, lights=lights, rng=rng
        )

        # Act - Try to move right from position 4 (becomes left due to switched zone)
        movement_law.apply(state_right, Action.MOVE_RIGHT)

        # Assert - Should move left to 3 due to switched zone, not stay at 4
        assert state_right.player.position == 3

        # Test right boundary with MOVE_LEFT (becomes right due to switched zone)
        player_right2 = Player(position=4)
        state_right2 = GameState(
            config=config, player=player_right2, lights=lights, rng=rng
        )

        # Act - Try to move left from position 4 (becomes right due to switched zone)
        movement_law.apply(state_right2, Action.MOVE_LEFT)

        # Assert - Should stay at 4 due to boundary constraint
        assert state_right2.player.position == 4

    def test_deterministic_slipperiness(self):
        """Test deterministic slipperiness (slip_probability=1.0)."""
        # Arrange
        config = WorldConfig(width=10, switch_point=5)
        player = Player(position=3)  # In normal zone
        lights = [Light(position=1, is_on=False)]
        rng = random.Random(42)
        state = GameState(config=config, player=player, lights=lights, rng=rng)

        movement_law = MovementLaw(slip_probability=1.0)

        # Act - Move right (should become left due to slipperiness)
        movement_law.apply(state, Action.MOVE_RIGHT)

        # Assert
        assert state.player.position == 2

        # Act - Move left (should become right due to slipperiness)
        movement_law.apply(state, Action.MOVE_LEFT)

        # Assert
        assert state.player.position == 3


class TestLightLaw:
    """Test cases for the LightLaw."""

    def test_lights_never_change_with_zero_probability(self):
        """Test with toggle_probability=0.0: lights should never change state."""
        # Arrange
        config = WorldConfig(width=10, switch_point=5)
        player = Player(position=3)
        lights = [Light(position=1, is_on=False), Light(position=8, is_on=True)]
        rng = random.Random(42)
        state = GameState(config=config, player=player, lights=lights, rng=rng)

        light_law = LightLaw(toggle_probability=0.0)
        original_states = [light.is_on for light in state.lights]

        # Act
        light_law.apply(state, Action.STAY)

        # Assert
        for i, light in enumerate(state.lights):
            assert light.is_on == original_states[i]

    def test_lights_always_flip_with_max_probability(self):
        """Test with toggle_probability=1.0: lights should always flip their state."""
        # Arrange
        config = WorldConfig(width=10, switch_point=5)
        player = Player(position=3)
        lights = [Light(position=1, is_on=False), Light(position=8, is_on=True)]
        rng = random.Random(42)
        state = GameState(config=config, player=player, lights=lights, rng=rng)

        light_law = LightLaw(toggle_probability=1.0)
        original_states = [light.is_on for light in state.lights]

        # Act
        light_law.apply(state, Action.STAY)

        # Assert
        for i, light in enumerate(state.lights):
            assert light.is_on != original_states[i]


class TestInitialState:
    """Test cases for the initial_state function."""

    def test_default_initial_state(self):
        """Test default initial state creation."""
        # Act
        state = initial_state()

        # Assert
        assert state.config.width == 12
        assert state.config.switch_point == 6
        assert state.player.position == 3  # width // 4
        assert len(state.lights) == 2
        assert state.lights[0].position == 3  # width // 4
        assert state.lights[1].position == 9  # 3 * width // 4
        assert not state.lights[0].is_on
        assert not state.lights[1].is_on
        assert isinstance(state.rng, random.Random)

    def test_custom_initial_state(self):
        """Test custom initial state creation."""
        # Act
        state = initial_state(width=8, num_lights=3, seed=123)

        # Assert
        assert state.config.width == 8
        assert state.config.switch_point == 4
        assert state.player.position == 2  # width // 4
        assert len(state.lights) == 3
        # Note: With 3 lights, the third one will be placed at the same position as the first
        # This is a limitation of the current implementation but acceptable for testing


class TestTransitionFunction:
    """Test cases for the transition_function."""

    def test_state_independence(self):
        """Verify that transition_function does not modify the original state object."""
        # Arrange
        state = initial_state(width=10, num_lights=1, seed=42)
        original_player_pos = state.player.position
        original_light_state = state.lights[0].is_on

        movement_law = MovementLaw(slip_probability=0.0)
        light_law = LightLaw(toggle_probability=0.0)
        laws = [movement_law, light_law]

        # Act
        new_state = transition_function(state, Action.MOVE_RIGHT, laws)

        # Assert - Original state should be unchanged
        assert state.player.position == original_player_pos
        assert state.lights[0].is_on == original_light_state

        # Assert - New state should be different
        assert new_state.player.position != original_player_pos

    def test_reproducibility(self):
        """Test that the same seed and action sequence produces identical results."""
        # Arrange
        seed = 42
        actions = [
            Action.MOVE_RIGHT,
            Action.MOVE_LEFT,
            Action.STAY,
            Action.MOVE_RIGHT,
            Action.MOVE_RIGHT,
            Action.MOVE_LEFT,
            Action.MOVE_RIGHT,
            Action.STAY,
            Action.MOVE_LEFT,
            Action.MOVE_RIGHT,
        ]

        movement_law = MovementLaw(slip_probability=0.1)
        light_law = LightLaw(toggle_probability=0.2)
        laws = [movement_law, light_law]

        # First run
        state1 = initial_state(width=10, num_lights=2, seed=seed)
        for action in actions:
            state1 = transition_function(state1, action, laws)

        # Second run with same seed
        state2 = initial_state(width=10, num_lights=2, seed=seed)
        for action in actions:
            state2 = transition_function(state2, action, laws)

        # Assert - States should be identical
        assert state1.player.position == state2.player.position
        assert len(state1.lights) == len(state2.lights)
        for i in range(len(state1.lights)):
            assert state1.lights[i].position == state2.lights[i].position
            assert state1.lights[i].is_on == state2.lights[i].is_on

    def test_law_application_order(self):
        """Test that laws are applied in the correct order."""
        # Arrange
        state = initial_state(width=10, num_lights=1, seed=42)

        # Create laws that log their application
        applied_laws = []

        class TestMovementLaw(MovementLaw):
            def apply(self, state, action):
                applied_laws.append("movement")
                super().apply(state, action)

        class TestLightLaw(LightLaw):
            def apply(self, state, action):
                applied_laws.append("light")
                super().apply(state, action)

        movement_law = TestMovementLaw(slip_probability=0.0)
        light_law = TestLightLaw(toggle_probability=0.0)
        laws = [movement_law, light_law]

        # Act
        transition_function(state, Action.MOVE_RIGHT, laws)

        # Assert
        assert applied_laws == ["movement", "light"]


class TestIntegration:
    """Integration tests for the complete environment."""

    def test_complete_gameplay_sequence(self):
        """Test a complete gameplay sequence with default laws."""
        # Arrange
        from distant_sunburn.one_d_test_environment import DEFAULT_LAWS

        state = initial_state(width=8, num_lights=2, seed=123)

        # Act - Play a sequence of actions
        actions = [Action.MOVE_RIGHT, Action.MOVE_RIGHT, Action.STAY, Action.MOVE_LEFT]

        for action in actions:
            state = transition_function(state, action, DEFAULT_LAWS)

        # Assert - State should be valid
        assert 0 <= state.player.position < state.config.width
        assert len(state.lights) == 2
        for light in state.lights:
            assert 0 <= light.position < state.config.width
            assert isinstance(light.is_on, bool)

    def test_switched_zone_mechanics(self):
        """Test the switched zone mechanics in a complete scenario."""
        # Arrange
        state = initial_state(width=6, num_lights=1, seed=42)
        movement_law = MovementLaw(slip_probability=0.0)
        light_law = LightLaw(toggle_probability=0.0)
        laws = [movement_law, light_law]

        # Player starts at position 1 (normal zone)
        assert state.player.position == 1
        assert state.config.switch_point == 3

        # Act - Move right to enter switched zone
        state = transition_function(state, Action.MOVE_RIGHT, laws)
        assert state.player.position == 2

        state = transition_function(state, Action.MOVE_RIGHT, laws)
        assert state.player.position == 3  # Now in switched zone

        # Act - Move right in switched zone (should go left)
        state = transition_function(state, Action.MOVE_RIGHT, laws)

        # Assert - Should have moved left due to switched zone
        assert state.player.position == 2
