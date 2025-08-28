"""
Unit tests for single-transition loss computation and optimization debugging for Crafter.

These tests validate the PoE-World inference machinery on the Crafter environment
using carefully chosen single transitions that clearly distinguish between
correct and incorrect experts.
"""

import numpy as np
import torch
import copy
from typing import List

from distant_sunburn.poe_world.core import (
    SymbolicTransition,
    DiscreteDistribution,
    ObservableId,
)
from distant_sunburn.poe_world.weight_fitter import (
    MaxLikelihoodWeightFitter,
)
from distant_sunburn.poe_world.world_model import PoEWorldModel
from distant_sunburn.poe_world.crafter.observable_extractor import ObservableExtractor
from distant_sunburn.poe_world.crafter.handwritten_experts import (
    correct_player_movement_expert,
    correct_combat_damage_expert,
    correct_entity_ai_expert,
    incorrect_player_movement_expert_teleports,
    incorrect_combat_damage_expert_instakills,
    incorrect_entity_ai_expert_self_destructs,
    CORRECT_EXPERTS,
    INCORRECT_EXPERTS,
)

from crafter.state_export import (
    WorldState,
    Position,
    PlayerState,
    CowState,
    ZombieState,
)
from crafter.functional_env import initial_state
from crafter.constants import ActionT


def create_movement_test_transition() -> SymbolicTransition[WorldState]:
    """
    Create a single transition where correct vs incorrect movement experts should be obvious.

    Setup: Player at position (2, 2), action "move_right"
    Expected: Position should go to (3, 2) (one step right)

    - Correct expert: Knows about movement, predicts position (3, 2)
    - Incorrect expert: Teleports to random position, predicts something else
    """
    # Create initial state with player at (2, 2)
    prev_state = initial_state(area=(5, 5), view=(3, 3), seed=42)

    # Manually set player position to (2, 2) for controlled test
    prev_state.player.position = Position(x=2, y=2)
    prev_state.player.facing = Position(x=1, y=0)  # Facing right

    # Action: move_right
    action: ActionT = "move_right"

    # Expected next state: position (3, 2) (moved one step right)
    next_state = copy.deepcopy(prev_state)
    next_state.player.position = Position(x=3, y=2)
    next_state.step_count = 1

    return SymbolicTransition(
        prev_metadata=prev_state, action=action, next_metadata=next_state
    )


def create_combat_test_transition() -> SymbolicTransition[WorldState]:
    """
    Create a single transition where correct vs incorrect combat experts should be obvious.

    Setup: Player at (2, 2) facing right, cow at (3, 2) with health 5, action "do"
    Expected: Cow health should go from 5 to 3 (wood sword damage)

    - Correct expert: Knows about combat damage, predicts cow health 3
    - Incorrect expert: Instantly kills cow, predicts cow health 0
    """
    # Create initial state
    prev_state = initial_state(area=(5, 5), view=(3, 3), seed=123)

    # Set up player position and facing
    prev_state.player.position = Position(x=2, y=2)
    prev_state.player.facing = Position(x=1, y=0)  # Facing right
    prev_state.player.inventory.wood_sword = 1  # Has wood sword

    # Add a cow in front of player
    cow = CowState(
        entity_id=2,
        position=Position(x=3, y=2),  # Right in front of player
        health=5,
        name="cow",
    )
    prev_state.objects.append(cow)

    # Action: do (attack)
    action: ActionT = "do"

    # Expected next state: cow health reduced from 5 to 3
    next_state = copy.deepcopy(prev_state)
    # Find the cow entity and update its health
    for obj in next_state.objects:
        if obj.name == "cow":
            obj.health = 3  # Cow takes 2 damage from wood sword
            break
    next_state.step_count = 1

    return SymbolicTransition(
        prev_metadata=prev_state, action=action, next_metadata=next_state
    )


def create_entity_ai_test_transition() -> SymbolicTransition[WorldState]:
    """
    Create a single transition where correct vs incorrect entity AI experts should be obvious.

    Setup: Player at (2, 2), zombie at (3, 2) with health 10, action "move_right"
    Expected: Zombie should pursue player (move toward player)

    - Correct expert: Knows about zombie AI, predicts zombie moves toward player
    - Incorrect expert: Makes zombie self-destruct, predicts zombie health 0
    """
    # Create initial state
    prev_state = initial_state(area=(5, 5), view=(3, 3), seed=456)

    # Set up player position
    prev_state.player.position = Position(x=2, y=2)

    # Add a zombie near player
    zombie = ZombieState(
        entity_id=2,
        position=Position(x=3, y=2),  # Right of player
        health=10,
        cooldown=0,
        name="zombie",
    )
    prev_state.objects.append(zombie)

    # Action: move_right (player moves, zombie should respond)
    action: ActionT = "move_right"

    # Expected next state: zombie moves toward player (from 3,2 to 2,2)
    next_state = copy.deepcopy(prev_state)
    next_state.player.position = Position(x=3, y=2)  # Player moved right
    next_state.objects[0].position = Position(x=2, y=2)  # Zombie moved toward player
    next_state.step_count = 1

    return SymbolicTransition(
        prev_metadata=prev_state, action=action, next_metadata=next_state
    )


class TestSingleTransitionLossComputation:
    """Test loss computation with single, clear transitions."""

    def test_movement_expert_predictions_are_different(self):
        """Verify that correct and incorrect movement experts make different predictions."""
        transition = create_movement_test_transition()

        # Test correct expert
        state_copy = copy.deepcopy(transition.prev_metadata)
        correct_player_movement_expert(state_copy, transition.action)
        correct_prediction_x = state_copy.player.position.x.support[0]  # type: ignore
        correct_prediction_y = state_copy.player.position.y.support[0]  # type: ignore

        # Test incorrect expert
        state_copy = copy.deepcopy(transition.prev_metadata)
        incorrect_player_movement_expert_teleports(state_copy, transition.action)
        incorrect_prediction_x = state_copy.player.position.x.support[0]  # type: ignore
        incorrect_prediction_y = state_copy.player.position.y.support[0]  # type: ignore

        # Predictions should be different
        assert (correct_prediction_x, correct_prediction_y) != (
            incorrect_prediction_x,
            incorrect_prediction_y,
        )

        # Correct expert should predict the ground truth
        assert correct_prediction_x == transition.next_metadata.player.position.x
        assert correct_prediction_y == transition.next_metadata.player.position.y

        # Incorrect expert should NOT predict the ground truth
        assert (incorrect_prediction_x, incorrect_prediction_y) != (
            transition.next_metadata.player.position.x,
            transition.next_metadata.player.position.y,
        )

    def test_combat_expert_predictions_are_different(self):
        """Verify that correct and incorrect combat experts make different predictions."""
        transition = create_combat_test_transition()

        # Test correct expert
        state_copy = copy.deepcopy(transition.prev_metadata)
        correct_combat_damage_expert(state_copy, transition.action)

        # Find the cow entity
        cow = None
        for obj in state_copy.objects:
            if obj.name == "cow":
                cow = obj
                break
        assert cow is not None

        correct_prediction = cow.health.support[0]  # type: ignore

        # Test incorrect expert
        state_copy = copy.deepcopy(transition.prev_metadata)
        incorrect_combat_damage_expert_instakills(state_copy, transition.action)

        # Find the cow entity
        cow = None
        for obj in state_copy.objects:
            if obj.name == "cow":
                cow = obj
                break
        assert cow is not None

        incorrect_prediction = cow.health.support[0]  # type: ignore

        # Predictions should be different
        assert correct_prediction != incorrect_prediction

        # Find the cow in the next state to get the ground truth
        cow_next = None
        for obj in transition.next_metadata.objects:
            if obj.name == "cow":
                cow_next = obj
                break
        assert cow_next is not None, "Cow entity not found in next state"

        # Correct expert should predict the ground truth
        assert correct_prediction == cow_next.health

        # Incorrect expert should NOT predict the ground truth
        assert incorrect_prediction != cow_next.health

    def test_entity_ai_expert_predictions_are_different(self):
        """Verify that correct and incorrect entity AI experts make different predictions."""
        transition = create_entity_ai_test_transition()

        # Test correct expert
        state_copy = copy.deepcopy(transition.prev_metadata)
        correct_entity_ai_expert(state_copy, transition.action)

        # Find the zombie entity
        zombie = None
        for obj in state_copy.objects:
            if obj.name == "zombie":
                zombie = obj
                break
        assert zombie is not None

        correct_prediction_health = zombie.health
        # Note: Correct expert might not change position deterministically due to randomness

        # Test incorrect expert
        state_copy = copy.deepcopy(transition.prev_metadata)
        incorrect_entity_ai_expert_self_destructs(state_copy, transition.action)

        # Find the zombie entity
        zombie = None
        for obj in state_copy.objects:
            if obj.name == "zombie":
                zombie = obj
                break
        assert zombie is not None

        incorrect_prediction = zombie.health.support[0]  # type: ignore

        # Incorrect expert should predict health 0 (self-destruct)
        assert incorrect_prediction == 0

        # Correct expert should NOT predict health 0
        assert correct_prediction_health != 0

    def test_loss_computation_ordering(self):
        """Test that loss computation correctly orders expert quality."""
        transition = create_movement_test_transition()
        experts = [
            correct_player_movement_expert,
            incorrect_player_movement_expert_teleports,
        ]

        fitter = MaxLikelihoodWeightFitter(
            observable_extractor=ObservableExtractor(),
        )
        expert_predictions = fitter._precompute_expert_predictions(
            experts, [transition]
        )

        # Test different weight combinations
        test_cases = [
            ([1.0, 0.0], "Only correct expert"),
            ([0.0, 1.0], "Only incorrect expert"),
            ([0.5, 0.5], "Equal weights"),
        ]

        losses = {}
        for weight_values, description in test_cases:
            weights = torch.tensor(weight_values, dtype=torch.float32)
            loss = fitter._compute_loss(weights, [transition], expert_predictions)
            losses[description] = loss.item()

        # Only correct expert should have lowest loss
        assert losses["Only correct expert"] < losses["Equal weights"]
        assert losses["Only correct expert"] < losses["Only incorrect expert"]

        # Only incorrect expert should have highest loss
        assert losses["Only incorrect expert"] > losses["Equal weights"]
        assert losses["Only incorrect expert"] > losses["Only correct expert"]


class TestSingleTransitionOptimization:
    """Test optimization with single, clear transitions."""

    def test_movement_transition_weight_optimization(self):
        """Test that optimizer correctly learns from single clear movement transition."""
        transition = create_movement_test_transition()
        experts = [
            correct_player_movement_expert,
            incorrect_player_movement_expert_teleports,
        ]

        fitter = MaxLikelihoodWeightFitter(
            observable_extractor=ObservableExtractor(),
            learning_rate=0.1,
            max_iterations=10,
            batch_size=1,
            l1_weight=0.0,  # No regularization for this simple test
        )

        # Fit weights
        weighted_experts = fitter.fit(experts, [transition])

        # Correct expert should have higher weight than incorrect expert
        correct_weight = weighted_experts[0].weight
        incorrect_weight = weighted_experts[1].weight

        assert correct_weight > incorrect_weight

        # The ratio should be substantial (not just barely different)
        weight_ratio = correct_weight / incorrect_weight
        assert weight_ratio > 1.5  # At least 1.5:1 ratio

    def test_combat_transition_weight_optimization(self):
        """Test that optimizer correctly learns from single clear combat transition."""
        transition = create_combat_test_transition()
        experts = [
            correct_combat_damage_expert,
            incorrect_combat_damage_expert_instakills,
        ]

        fitter = MaxLikelihoodWeightFitter(
            observable_extractor=ObservableExtractor(),
            learning_rate=0.1,
            max_iterations=10,
            batch_size=1,
            l1_weight=0.0,  # No regularization for this simple test
        )

        # Fit weights
        weighted_experts = fitter.fit(experts, [transition])

        # Correct expert should have higher weight than incorrect expert
        correct_weight = weighted_experts[0].weight
        incorrect_weight = weighted_experts[1].weight

        assert correct_weight > incorrect_weight

        # The ratio should be substantial
        weight_ratio = correct_weight / incorrect_weight
        assert weight_ratio > 1.5  # At least 1.5:1 ratio


class TestWorldModelEvaluation:
    """Test that the PoE World Model can evaluate log-probabilities correctly."""

    def test_world_model_evaluation_on_movement(self):
        """Test world model evaluation on movement transition."""
        transition = create_movement_test_transition()
        experts = [
            correct_player_movement_expert,
            incorrect_player_movement_expert_teleports,
        ]

        # Fit weights
        fitter = MaxLikelihoodWeightFitter(
            observable_extractor=ObservableExtractor(),
            max_iterations=10,
            l1_weight=0.0,
        )
        weighted_experts = fitter.fit(experts, [transition])

        # Create world model
        world_model = PoEWorldModel(
            observable_extractor=ObservableExtractor(),
            weighted_experts=weighted_experts,
        )

        # Evaluate log-probability
        log_prob = world_model.evaluate_log_probability(
            transition.prev_metadata, transition.action, transition.next_metadata
        )

        # Basic sanity checks
        assert isinstance(log_prob, (int, float))
        assert np.isfinite(log_prob), "Log-probability should be finite"

    def test_world_model_evaluation_on_combat(self):
        """Test world model evaluation on combat transition."""
        transition = create_combat_test_transition()
        experts = [
            correct_combat_damage_expert,
            incorrect_combat_damage_expert_instakills,
        ]

        # Fit weights
        fitter = MaxLikelihoodWeightFitter(
            observable_extractor=ObservableExtractor(),
            max_iterations=10,
            l1_weight=0.0,
        )
        weighted_experts = fitter.fit(experts, [transition])

        # Create world model
        world_model = PoEWorldModel(
            observable_extractor=ObservableExtractor(),
            weighted_experts=weighted_experts,
        )

        # Evaluate log-probability
        log_prob = world_model.evaluate_log_probability(
            transition.prev_metadata, transition.action, transition.next_metadata
        )

        # Basic sanity checks
        assert isinstance(log_prob, (int, float))
        assert np.isfinite(log_prob), "Log-probability should be finite"
