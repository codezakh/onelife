"""
Unit tests for single-transition loss computation and optimization debugging.

These tests capture the debugging methodology that proved critical for solving
the gradient flow issue in PoE-World weight fitting. They serve as both tests
and documentation of best practices.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from typing import List, cast
from distant_sunburn.poe_world.core import ObservableId

from distant_sunburn.poe_world.core import SymbolicTransition, RandomValues
from distant_sunburn.simple_1d_env.environment import (
    GameState,
    Player,
    Light,
    WorldConfig,
    Action,
)
from distant_sunburn.poe_world.simple_1d_env.handwritten_experts import (
    correct_movement_expert,
    incorrect_movement_expert_ignores_switch,
)
from distant_sunburn.poe_world.weight_fitter import (
    MaxLikelihoodWeightFitter,
    combine_expert_predictions_torch,
    evaluate_log_probability_torch,
)
from distant_sunburn.poe_world.simple_1d_env.observable_extractor import (
    ObservableExtractor,
)


def create_clear_transition() -> SymbolicTransition[GameState]:
    """
    Create a single transition where correct vs incorrect experts should be obvious.

    Setup: Player at position 6 (>= switch_point=6), action MOVE_RIGHT
    Expected: Position should go LEFT to 5 (due to switch zone)

    - Correct expert: Knows about switch zone, predicts position 5
    - Incorrect expert: Ignores switch zone, predicts position 7
    """

    # Create initial state: player at position 6 (right at switch point)
    prev_state = GameState(
        config=WorldConfig(width=12, switch_point=6),
        player=Player(position=6),
        lights=[Light(position=3, is_on=False), Light(position=9, is_on=True)],
        rng=random.Random(42),  # Fixed seed for deterministic test
    )

    # Action: MOVE_RIGHT (but switch zone should invert to LEFT)
    action = Action.MOVE_RIGHT

    # Expected next state: position 5 (moved left due to switch zone)
    next_state = GameState(
        config=WorldConfig(width=12, switch_point=6),
        player=Player(position=5),  # This is the ground truth
        lights=[Light(position=3, is_on=False), Light(position=9, is_on=True)],
        rng=random.Random(42),
    )

    return SymbolicTransition(
        prev_metadata=prev_state, action=action, next_metadata=next_state
    )


class TestSingleTransitionLossComputation:
    """Test loss computation with a single, clear transition."""

    def test_expert_predictions_are_different(self):
        """Verify that correct and incorrect experts make different predictions."""
        transition = create_clear_transition()

        # Test correct expert
        state_copy = copy.deepcopy(transition.prev_metadata)
        correct_movement_expert(state_copy, transition.action)
        correct_prediction = state_copy.player.position.values[0]  # type: ignore

        # Test incorrect expert
        state_copy = copy.deepcopy(transition.prev_metadata)
        incorrect_movement_expert_ignores_switch(state_copy, transition.action)
        incorrect_prediction = state_copy.player.position.values[0]  # type: ignore

        # Predictions should be different
        assert correct_prediction != incorrect_prediction

        # Correct expert should predict the ground truth
        assert correct_prediction == transition.next_metadata.player.position

        # Incorrect expert should NOT predict the ground truth
        assert incorrect_prediction != transition.next_metadata.player.position

    def test_loss_computation_ordering(self):
        """Test that loss computation correctly orders expert quality."""
        transition = create_clear_transition()
        experts = [correct_movement_expert, incorrect_movement_expert_ignores_switch]

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

    def test_gradient_flow_exists(self):
        """Test that gradients flow through loss computation."""
        transition = create_clear_transition()
        experts = [correct_movement_expert, incorrect_movement_expert_ignores_switch]

        fitter = MaxLikelihoodWeightFitter(
            observable_extractor=ObservableExtractor(),
        )
        expert_predictions = fitter._precompute_expert_predictions(
            experts, [transition]
        )

        # Create weights tensor with gradient tracking
        weights = torch.tensor([0.5, 0.5], dtype=torch.float32, requires_grad=True)

        # Compute loss and backpropagate
        loss = fitter._compute_loss(weights, [transition], expert_predictions)
        loss.backward()

        # Verify gradients exist
        assert weights.grad is not None
        assert len(weights.grad) == 2

        # Gradients should be different for different quality experts
        assert not torch.allclose(weights.grad[0], weights.grad[1], atol=1e-6)

    def test_pytorch_tensor_operations_preserve_gradients(self):
        """Test that our PyTorch-native functions preserve gradients."""
        transition = create_clear_transition()
        experts = [correct_movement_expert, incorrect_movement_expert_ignores_switch]

        fitter = MaxLikelihoodWeightFitter(
            observable_extractor=ObservableExtractor(),
        )
        expert_predictions = fitter._precompute_expert_predictions(
            experts, [transition]
        )

        # Get expert predictions for player position
        attr_predictions = [
            pred[ObservableId("player_position")] for pred in expert_predictions[0]
        ]

        # Test gradient flow through combination function
        weights = torch.tensor([0.7, 0.3], dtype=torch.float32, requires_grad=True)

        values_tensor, combined_logscores = combine_expert_predictions_torch(
            attr_predictions, weights
        )

        # Test gradient flow through evaluation function
        observed_value = transition.next_metadata.player.position
        log_prob = evaluate_log_probability_torch(
            values_tensor, combined_logscores, observed_value
        )

        # Backpropagate
        log_prob.backward()

        # Verify gradients exist and are meaningful
        assert weights.grad is not None
        assert not torch.allclose(weights.grad, torch.zeros_like(weights.grad))


class TestSingleTransitionOptimization:
    """Test optimization with a single, clear transition."""

    def test_simple_quadratic_optimization(self):
        """Test optimizer on simple quadratic function before testing actual loss."""
        # Target: minimize (x[0] - 0.8)^2 + (x[1] - 0.2)^2
        # Expected result: weights converge to [0.8, 0.2]

        weights = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        optimizer = optim.LBFGS([weights], lr=0.1, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = (weights[0] - 0.8) ** 2 + (weights[1] - 0.2) ** 2
            loss.backward()
            return loss

        # Run optimization
        for _ in range(5):
            optimizer.step(closure)

        # Should converge to target values
        assert torch.allclose(weights, torch.tensor([0.8, 0.2]), atol=1e-4)

    def test_single_transition_weight_optimization(self):
        """Test that optimizer correctly learns from single clear transition."""
        transition = create_clear_transition()
        experts = [correct_movement_expert, incorrect_movement_expert_ignores_switch]

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
        assert weight_ratio > 2.0  # At least 2:1 ratio

    def test_weight_convergence_stability(self):
        """Test that weights converge to stable values."""
        transition = create_clear_transition()
        experts = [correct_movement_expert, incorrect_movement_expert_ignores_switch]

        fitter = MaxLikelihoodWeightFitter(
            observable_extractor=ObservableExtractor(),
            learning_rate=0.1,
            max_iterations=20,
            batch_size=1,
            l1_weight=0.0,
        )

        # Run fitting multiple times with same data
        results = []
        for _ in range(3):
            weighted_experts = fitter.fit(experts, [transition])
            weights = [we.weight for we in weighted_experts]
            results.append(weights)

        # Results should be consistent (deterministic with same data/seed)
        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i], atol=1e-3)

    def test_loss_decreases_during_optimization(self):
        """Test that loss actually decreases during optimization."""
        transition = create_clear_transition()
        experts = [correct_movement_expert, incorrect_movement_expert_ignores_switch]

        fitter = MaxLikelihoodWeightFitter(
            observable_extractor=ObservableExtractor(),
        )
        expert_predictions = fitter._precompute_expert_predictions(
            experts, [transition]
        )

        # Track loss during optimization
        losses = []

        def loss_tracking_closure(weights):
            loss = fitter._compute_loss(weights, [transition], expert_predictions)
            losses.append(loss.item())
            return loss

        weights = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        optimizer = optim.LBFGS([weights], lr=0.1, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = loss_tracking_closure(weights)
            loss.backward()
            return loss

        # Run optimization
        for _ in range(5):
            optimizer.step(closure)

        # Loss should decrease
        assert len(losses) > 1
        final_loss = losses[-1]
        initial_loss = losses[0]
        assert final_loss < initial_loss


class TestDebuggingMethodology:
    """Test the debugging methodology itself."""

    def test_minimal_failing_example_principle(self):
        """Test that single transitions can expose issues that large datasets mask."""
        # This test documents the principle: if it doesn't work on 1 clear example,
        # it won't work on 1000 noisy examples

        transition = create_clear_transition()

        # Create a deliberately broken weight fitter (one that ignores expert quality)
        class BrokenFitter:
            def fit(self, experts, transitions):
                # Always returns equal weights regardless of expert quality
                return [
                    type(
                        "WeightedExpert", (), {"weight": 0.5, "expert_function": expert}
                    )()
                    for expert in experts
                ]

        broken_fitter = BrokenFitter()
        experts = [correct_movement_expert, incorrect_movement_expert_ignores_switch]

        # Even on perfect single transition, broken fitter fails
        weighted_experts = broken_fitter.fit(experts, [transition])
        weights = [we.weight for we in weighted_experts]  # type: ignore

        # This should fail (weights are equal when they shouldn't be)
        assert weights[0] == weights[1]  # Documents the bug

        # The real fitter should succeed
        real_fitter = MaxLikelihoodWeightFitter(
            observable_extractor=ObservableExtractor(),
            max_iterations=10,
            l1_weight=0.0,
        )
        real_weighted_experts = real_fitter.fit(experts, [transition])
        real_weights = [we.weight for we in real_weighted_experts]

        # Real fitter should distinguish experts
        assert real_weights[0] != real_weights[1]
        assert real_weights[0] > real_weights[1]

    def test_loss_verification_before_optimization(self):
        """Test the principle of verifying loss computation before optimization."""
        transition = create_clear_transition()
        experts = [correct_movement_expert, incorrect_movement_expert_ignores_switch]

        fitter = MaxLikelihoodWeightFitter(
            observable_extractor=ObservableExtractor(),
        )
        expert_predictions = fitter._precompute_expert_predictions(
            experts, [transition]
        )

        # Step 1: Verify loss computation manually
        # (This is what we should do BEFORE trying to optimize)

        correct_only_weights = torch.tensor([1.0, 0.0], dtype=torch.float32)
        incorrect_only_weights = torch.tensor([0.0, 1.0], dtype=torch.float32)

        correct_only_loss = fitter._compute_loss(
            correct_only_weights, [transition], expert_predictions
        )
        incorrect_only_loss = fitter._compute_loss(
            incorrect_only_weights, [transition], expert_predictions
        )

        # If this assertion fails, optimization will definitely fail
        assert correct_only_loss < incorrect_only_loss

        # Step 2: Only after verifying loss computation, test optimization
        weighted_experts = fitter.fit(experts, [transition])

        # Optimization should succeed because loss computation is correct
        assert weighted_experts[0].weight > weighted_experts[1].weight
