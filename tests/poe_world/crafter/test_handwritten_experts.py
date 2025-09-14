"""
Tests for the handwritten experts for the Crafter environment.

This module tests that the correct experts model the right mechanics
and that the incorrect experts make obviously wrong predictions.
"""

import pytest
from unittest.mock import Mock
from crafter.state_export import (
    WorldState,
    Position,
    PlayerState,
    CowState,
    ZombieState,
    SkeletonState,
)
from crafter.functional_env import initial_state
from crafter.constants import ActionT

from distant_sunburn.poe_world.crafter.handwritten_experts import (
    correct_player_movement_expert,
    correct_combat_damage_expert,
    correct_entity_ai_expert,
    incorrect_player_movement_expert_teleports,
    incorrect_combat_damage_expert_instakills,
    incorrect_entity_ai_expert_self_destructs,
)
from distant_sunburn.poe_world.core import DiscreteDistribution
from loguru import logger
import distant_sunburn.poe_world.crafter.handwritten_experts


logger.enable(distant_sunburn.poe_world.crafter.handwritten_experts.__name__)


def create_simple_test_state() -> WorldState:
    """Create a simple test state with just a player."""
    return initial_state(area=(5, 5), view=(3, 3), seed=42)


def create_mock_rng_for_movement() -> Mock:
    """Create a mock RNG that always triggers movement behavior."""
    mock_rng = Mock()
    # For cows: uniform() < 0.5 should return True (0.3 < 0.5)
    # For zombies: uniform() < 0.9 should return True (0.8 < 0.9)
    # For skeletons: uniform() < 0.5 should return True (0.3 < 0.5)
    mock_rng.uniform.return_value = 0.3  # Always triggers movement
    mock_rng.randint.return_value = 0  # Always choose first direction (0, 1)
    return mock_rng


def create_mock_rng_for_no_movement() -> Mock:
    """Create a mock RNG that never triggers movement behavior."""
    mock_rng = Mock()
    # For cows: uniform() < 0.5 should return False (0.7 > 0.5)
    # For zombies: uniform() < 0.9 should return False (0.95 > 0.9)
    # For skeletons: uniform() < 0.5 should return False (0.7 > 0.5)
    mock_rng.uniform.return_value = 0.7  # Never triggers movement
    mock_rng.randint.return_value = 0  # Default direction choice
    return mock_rng


class TestCorrectExperts:
    """Test that correct experts model the right mechanics."""

    def test_correct_movement_expert_moves_player(self):
        """Test that movement expert correctly predicts player movement."""
        # Arrange
        state = create_simple_test_state()
        initial_x = state.player.position.x
        initial_y = state.player.position.y

        # Act
        correct_player_movement_expert(state, "move_right")

        # Assert - Should predict moving one step right
        assert isinstance(state.player.position.x, DiscreteDistribution)
        assert state.player.position.x.support[0] == initial_x + 1
        assert state.player.position.y.support[0] == initial_y  # type: ignore

    def test_correct_movement_expert_respects_boundaries(self):
        """Test that movement expert respects world boundaries."""
        # Arrange - Player at right edge
        state = create_simple_test_state()
        state.player.position.x = 4  # Right edge of 5x5 world
        state.player.position.y = 2

        # Act
        correct_player_movement_expert(state, "move_right")

        # Assert - Should stay at edge
        assert isinstance(state.player.position.x, DiscreteDistribution)
        assert state.player.position.x.support[0] == 4  # Should not go beyond boundary

    def test_correct_combat_expert_damages_entity(self):
        """Test that combat expert correctly predicts damage to entities."""
        # Arrange
        state = create_simple_test_state()

        # Add a cow in front of player
        cow = CowState(
            entity_id=2,
            position=Position(x=3, y=2),  # Right in front of player at (2,2)
            health=5,
            name="cow",
        )
        state.objects.append(cow)
        state.player.facing = Position(x=1, y=0)  # Facing right
        state.player.inventory.wood_sword = 1  # Has weapon

        # Act
        correct_combat_damage_expert(state, "do")

        # Assert - Should predict damage to cow
        assert isinstance(cow.health, DiscreteDistribution)
        assert cow.health.support[0] == 3  # 5 - 2 (wood sword damage)

    def test_correct_combat_expert_focus_cow_only(self):
        """Test that combat expert with focus='cow' only affects cows."""
        # Test with cow
        state_with_cow = create_simple_test_state()
        cow = CowState(
            entity_id=2,
            position=Position(x=3, y=2),  # Right in front of player at (2,2)
            health=5,
        )
        state_with_cow.objects.append(cow)
        state_with_cow.player.facing = Position(x=1, y=0)  # Facing right
        state_with_cow.player.inventory.wood_sword = 1  # Has weapon

        # Act - Focus only on cows
        correct_combat_damage_expert(state_with_cow, "do", focus="cow")

        # Assert - Cow should be damaged
        assert isinstance(cow.health, DiscreteDistribution)
        assert cow.health.support[0] == 3  # 5 - 2 (wood sword damage)

        # Test with zombie (should not be affected)
        state_with_zombie = create_simple_test_state()
        zombie = ZombieState(
            entity_id=2,
            position=Position(x=3, y=2),  # Right in front of player at (2,2)
            health=10,
            cooldown=0,
        )
        state_with_zombie.objects.append(zombie)
        state_with_zombie.player.facing = Position(x=1, y=0)  # Facing right
        state_with_zombie.player.inventory.wood_sword = 1  # Has weapon

        # Act - Focus only on cows
        correct_combat_damage_expert(state_with_zombie, "do", focus="cow")

        # Assert - Zombie should not be damaged (not a DiscreteDistribution)
        assert zombie.health == 10

    def test_correct_combat_expert_focus_zombie_only(self):
        """Test that combat expert with focus='zombie' only affects zombies."""
        # Test with zombie
        state_with_zombie = create_simple_test_state()
        zombie = ZombieState(
            entity_id=2,
            position=Position(x=3, y=2),  # Right in front of player at (2,2)
            health=10,
            cooldown=0,
        )
        state_with_zombie.objects.append(zombie)
        state_with_zombie.player.facing = Position(x=1, y=0)  # Facing right
        state_with_zombie.player.inventory.wood_sword = 1  # Has weapon

        # Act - Focus only on zombies
        correct_combat_damage_expert(state_with_zombie, "do", focus="zombie")

        # Assert - Zombie should be damaged
        assert isinstance(zombie.health, DiscreteDistribution)
        assert zombie.health.support[0] == 8  # 10 - 2 (wood sword damage)

        # Test with cow (should not be affected)
        state_with_cow = create_simple_test_state()
        cow = CowState(
            entity_id=2,
            position=Position(x=3, y=2),  # Right in front of player at (2,2)
            health=5,
        )
        state_with_cow.objects.append(cow)
        state_with_cow.player.facing = Position(x=1, y=0)  # Facing right
        state_with_cow.player.inventory.wood_sword = 1  # Has weapon

        # Act - Focus only on zombies
        correct_combat_damage_expert(state_with_cow, "do", focus="zombie")

        # Assert - Cow should not be damaged (not a DiscreteDistribution)
        assert cow.health == 5

    def test_correct_combat_expert_focus_all_default(self):
        """Test that combat expert with focus='all' affects all entities (default behavior)."""
        # Test with cow (should be affected by focus='all')
        state_with_cow = create_simple_test_state()
        cow = CowState(
            entity_id=2,
            position=Position(x=3, y=2),  # Right in front of player at (2,2)
            health=5,
        )
        state_with_cow.objects.append(cow)
        state_with_cow.player.facing = Position(x=1, y=0)  # Facing right
        state_with_cow.player.inventory.wood_sword = 1  # Has weapon

        # Act - Focus on all entities (default)
        correct_combat_damage_expert(state_with_cow, "do", focus="all")

        # Assert - Cow should be damaged
        assert isinstance(cow.health, DiscreteDistribution)
        assert cow.health.support[0] == 3  # 5 - 2 (wood sword damage)

        # Test with zombie (should also be affected by focus='all')
        state_with_zombie = create_simple_test_state()
        zombie = ZombieState(
            entity_id=2,
            position=Position(x=3, y=2),  # Right in front of player at (2,2)
            health=10,
            cooldown=0,
        )
        state_with_zombie.objects.append(zombie)
        state_with_zombie.player.facing = Position(x=1, y=0)  # Facing right
        state_with_zombie.player.inventory.wood_sword = 1  # Has weapon

        # Act - Focus on all entities (default)
        correct_combat_damage_expert(state_with_zombie, "do", focus="all")

        # Assert - Zombie should be damaged
        assert isinstance(zombie.health, DiscreteDistribution)
        assert zombie.health.support[0] == 8  # 10 - 2 (wood sword damage)

    def test_correct_entity_ai_expert_focus_cow_only(self):
        """Test that entity AI expert with focus='cow' only affects cows."""
        # Test with cow (should be affected)
        state_with_cow = create_simple_test_state()
        cow = CowState(
            entity_id=2,
            position=Position(x=1, y=1),
            health=5,
        )
        state_with_cow.objects.append(cow)
        state_with_cow.random_state = create_mock_rng_for_movement()

        # Act - Focus only on cows
        correct_entity_ai_expert(state_with_cow, "move_right", focus="cow")

        # Assert - Cow should move. This should pass as long as any
        # of the position coordinates change.
        pos_changed = [
            isinstance(cow.position.x, DiscreteDistribution),
            isinstance(cow.position.y, DiscreteDistribution),
        ]
        assert any(pos_changed)

        # Test with zombie (should not be affected)
        state_with_zombie = create_simple_test_state()
        zombie = ZombieState(
            entity_id=2,
            position=Position(x=3, y=3),
            health=10,
            cooldown=0,
        )
        state_with_zombie.objects.append(zombie)
        state_with_zombie.random_state = create_mock_rng_for_movement()

        # Act - Focus only on cows
        correct_entity_ai_expert(state_with_zombie, "move_right", focus="cow")

        # Assert - Zombie should not move (position remains as integer)
        assert zombie.position.x == 3
        assert zombie.position.y == 3

    def test_correct_entity_ai_expert_focus_zombie_only(self):
        """Test that entity AI expert with focus='zombie' only affects zombies."""
        # Test with zombie (should be affected)
        state_with_zombie = create_simple_test_state()
        zombie = ZombieState(
            entity_id=2,
            position=Position(x=0, y=0),
            health=10,
            cooldown=0,
        )
        state_with_zombie.objects.append(zombie)
        state_with_zombie.random_state = create_mock_rng_for_movement()

        # Act - Focus only on zombies
        correct_entity_ai_expert(state_with_zombie, "noop", focus="zombie")

        # Assert - Zombie should move, and it should prefer the long axis
        # (though this is not important for the test)
        pos_changed = [
            isinstance(zombie.position.x, DiscreteDistribution),
            isinstance(zombie.position.y, DiscreteDistribution),
        ]
        assert any(pos_changed)

        # Test with cow (should not be affected)
        state_with_cow = create_simple_test_state()
        cow = CowState(
            entity_id=2,
            position=Position(x=1, y=1),
            health=5,
        )
        state_with_cow.objects.append(cow)
        state_with_cow.random_state = create_mock_rng_for_movement()

        # Act - Focus only on zombies
        correct_entity_ai_expert(state_with_cow, "move_right", focus="zombie")

        # Assert - Cow should not move (position remains as integer)
        assert cow.position.x == 1
        assert cow.position.y == 1

    def test_correct_entity_ai_expert_focus_skeleton_only(self):
        """Test that entity AI expert with focus='skeleton' only affects skeletons."""
        # Test with skeleton (should be affected)
        state_with_skeleton = create_simple_test_state()
        skeleton = SkeletonState(
            entity_id=2,
            position=Position(x=3, y=3),
            health=10,
            reload=0,
        )
        state_with_skeleton.objects.append(skeleton)
        state_with_skeleton.random_state = create_mock_rng_for_movement()

        # Act - Focus only on skeletons
        correct_entity_ai_expert(state_with_skeleton, "move_right", focus="skeleton")

        # Assert - Skeleton should move. This should pass as long as any
        # of the position coordinates change.
        pos_changed = [
            isinstance(skeleton.position.x, DiscreteDistribution),
            isinstance(skeleton.position.y, DiscreteDistribution),
        ]
        assert any(pos_changed)

        # Test with cow (should not be affected)
        state_with_cow = create_simple_test_state()
        cow = CowState(
            entity_id=2,
            position=Position(x=1, y=1),
            health=5,
        )
        state_with_cow.objects.append(cow)
        state_with_cow.random_state = create_mock_rng_for_movement()

        # Act - Focus only on skeletons
        correct_entity_ai_expert(state_with_cow, "move_right", focus="skeleton")

        # Assert - Cow should not move (position remains as integer)
        assert cow.position.x == 1
        assert cow.position.y == 1

    def test_correct_entity_ai_expert_focus_all_default(self):
        """Test that entity AI expert with focus='all' affects all entities (default behavior)."""
        # Test with cow (should be affected by focus='all')
        state_with_cow = create_simple_test_state()
        cow = CowState(
            entity_id=2,
            position=Position(x=1, y=1),
            health=5,
        )
        state_with_cow.objects.append(cow)
        state_with_cow.random_state = create_mock_rng_for_movement()

        # Act - Focus on all entities (default)
        correct_entity_ai_expert(state_with_cow, "move_right", focus="all")

        # Assert - Cow should move. This should pass as long as any
        # of the position coordinates change.
        pos_changed = [
            isinstance(cow.position.x, DiscreteDistribution),
            isinstance(cow.position.y, DiscreteDistribution),
        ]
        assert any(pos_changed)

        # Test with zombie (should also be affected by focus='all')
        state_with_zombie = create_simple_test_state()
        zombie = ZombieState(
            entity_id=2,
            position=Position(x=3, y=3),
            health=10,
            cooldown=0,
        )
        state_with_zombie.objects.append(zombie)
        state_with_zombie.random_state = create_mock_rng_for_movement()

        # Act - Focus on all entities (default)
        correct_entity_ai_expert(state_with_zombie, "move_right", focus="all")

        # Assert - Zombie should move. This should pass as long as any
        # of the position coordinates change.
        pos_changed = [
            isinstance(zombie.position.x, DiscreteDistribution),
            isinstance(zombie.position.y, DiscreteDistribution),
        ]
        assert any(pos_changed)


class TestIncorrectExperts:
    """Test that incorrect experts make obviously wrong predictions."""

    def test_incorrect_movement_expert_teleports(self):
        """Test that incorrect movement expert teleports instead of moving."""
        # Arrange
        state = create_simple_test_state()
        initial_x = state.player.position.x
        initial_y = state.player.position.y

        # Act
        incorrect_player_movement_expert_teleports(state, "move_right")

        # Assert - Should teleport to random position, not move one step
        assert isinstance(state.player.position.x, DiscreteDistribution)
        assert isinstance(state.player.position.y, DiscreteDistribution)

        teleported_x = state.player.position.x.support[0]
        teleported_y = state.player.position.y.support[0]

        # Should not be the expected position (initial_x + 1, initial_y)
        assert not (teleported_x == initial_x + 1 and teleported_y == initial_y)

    def test_incorrect_combat_expert_instakills(self):
        """Test that incorrect combat expert instantly kills entities."""
        # Arrange
        state = create_simple_test_state()
        cow = CowState(entity_id=2, position=Position(x=3, y=2), health=5, name="cow")
        state.objects.append(cow)
        state.player.facing = Position(x=1, y=0)

        # Act
        incorrect_combat_damage_expert_instakills(state, "do")

        # Assert - Should instantly kill the entity
        assert isinstance(cow.health, DiscreteDistribution)
        assert cow.health.support[0] == 0

    def test_incorrect_entity_expert_self_destructs(self):
        """Test that incorrect entity expert makes all entities self-destruct."""
        # Arrange
        state = create_simple_test_state()
        cow = CowState(entity_id=2, position=Position(x=3, y=2), health=5, name="cow")
        state.objects.append(cow)

        # Act
        incorrect_entity_ai_expert_self_destructs(state, "move_right")

        # Assert - All entities should self-destruct
        assert isinstance(cow.health, DiscreteDistribution)
        assert cow.health.support[0] == 0

        # Player should not be affected
        assert not isinstance(state.player.health, DiscreteDistribution)
