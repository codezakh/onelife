"""
Integration test for PoE-World inference machinery.

This test validates the complete inference pipeline:
1. Generate random data using the 1D environment
2. Split into training/testing sets
3. Fit expert weights using maximum likelihood
4. Validate that good experts get higher weights than bad experts
"""

import random
import numpy as np
import pytest
from typing import List

from distant_sunburn.poe_world.core import SymbolicTransition
from distant_sunburn.simple_1d_env.environment import (
    initial_state,
    transition_function,
    Action,
    DEFAULT_LAWS,
    GameState,
    WorldConfig,
)
from distant_sunburn.poe_world.simple_1d_env.handwritten_experts import (
    CORRECT_EXPERTS,
    INCORRECT_EXPERTS,
    ALL_EXPERTS,
)
from distant_sunburn.poe_world.simple_1d_env.weight_fitter import (
    MaxLikelihoodWeightFitter,
)
from distant_sunburn.poe_world.simple_1d_env.world_model import PoEWorldModel


def generate_random_data(
    n_transitions: int, seed: int = 42
) -> List[SymbolicTransition[GameState]]:
    """
    Generate random transitions using the 1D environment.

    Args:
        n_transitions: Number of transitions to generate
        seed: Random seed for reproducibility

    Returns:
        List of symbolic transitions
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    transitions = []
    current_state = initial_state(WorldConfig(seed=seed))

    for _ in range(n_transitions):
        # Choose random action
        action = rng.choice(list(Action))

        # Apply transition function
        next_state = transition_function(current_state, action, DEFAULT_LAWS)

        # Create symbolic transition
        transition = SymbolicTransition(
            prev_metadata=current_state, action=action, next_metadata=next_state
        )
        transitions.append(transition)

        # Update current state for next iteration
        current_state = next_state

    return transitions


def test_weight_fitting_distinguishes_good_from_bad_experts():
    """
    Integration test that validates the weight fitting pipeline.

    This test:
    1. Generates 1000 random transitions
    2. Splits into 75% training, 25% testing
    3. Fits weights to all experts (correct + incorrect)
    4. Validates that correct experts get higher weights than incorrect ones
    """
    n_transitions = 1000
    transitions = generate_random_data(n_transitions, seed=42)

    # Split into train/test (75/25)
    split_point = int(0.75 * len(transitions))
    train_transitions = transitions[:split_point]
    test_transitions = transitions[split_point:]

    print(f"Generated {len(transitions)} transitions")
    print(f"Training on {len(train_transitions)} transitions")
    print(f"Testing on {len(test_transitions)} transitions")

    fitter = MaxLikelihoodWeightFitter(
        learning_rate=0.1, max_iterations=25, batch_size=200, l1_weight=0.001
    )

    # Fit weights
    weighted_experts = fitter.fit(ALL_EXPERTS, train_transitions)

    # Extract weights for correct and incorrect experts
    correct_weights = []
    incorrect_weights = []

    for i, weighted_expert in enumerate(weighted_experts):
        weight = weighted_expert.weight
        expert_func = weighted_expert.expert_function

        print(f"Expert {i}: weight = {weight:.4f}")

        if expert_func in CORRECT_EXPERTS:
            correct_weights.append(weight)
            print(f"  -> CORRECT expert")
        elif expert_func in INCORRECT_EXPERTS:
            incorrect_weights.append(weight)
            print(f"  -> INCORRECT expert")

    # Validate that correct experts have higher average weight
    avg_correct_weight = np.mean(correct_weights)
    avg_incorrect_weight = np.mean(incorrect_weights)

    print(f"\nAverage correct expert weight: {avg_correct_weight:.4f}")
    print(f"Average incorrect expert weight: {avg_incorrect_weight:.4f}")

    # Main assertion: correct experts should have higher weights on average
    assert avg_correct_weight > avg_incorrect_weight, (
        f"Correct experts should have higher weights than incorrect ones. "
        f"Got correct={avg_correct_weight:.4f}, incorrect={avg_incorrect_weight:.4f}"
    )

    # Additional checks
    assert len(correct_weights) == len(
        CORRECT_EXPERTS
    ), "Should have weights for all correct experts"
    assert len(incorrect_weights) == len(
        INCORRECT_EXPERTS
    ), "Should have weights for all incorrect experts"

    # All weights should be non-negative
    all_weights = [we.weight for we in weighted_experts]
    assert all(w >= 0 for w in all_weights), "All weights should be non-negative"


def test_world_model_evaluation():
    """
    Test that the PoE World Model can evaluate log-probabilities correctly.
    """
    # Generate small dataset
    transitions = generate_random_data(100, seed=123)
    train_transitions = transitions[:75]
    test_transitions = transitions[75:]

    # Fit weights
    fitter = MaxLikelihoodWeightFitter(max_iterations=20)
    weighted_experts = fitter.fit(ALL_EXPERTS, train_transitions)

    # Create world model
    world_model = PoEWorldModel(weighted_experts)

    # Evaluate log-probabilities on test set
    test_log_probs = []
    for transition in test_transitions:
        log_prob = world_model.evaluate_log_probability(
            transition.prev_metadata, transition.action, transition.next_metadata
        )
        test_log_probs.append(log_prob)

    # Basic sanity checks
    assert len(test_log_probs) == len(test_transitions)
    assert all(
        isinstance(lp, (int, float)) for lp in test_log_probs
    ), "All log-probs should be numeric"

    # Log-probabilities should be finite (not -inf everywhere)
    finite_log_probs = [lp for lp in test_log_probs if np.isfinite(lp)]
    assert len(finite_log_probs) > 0, "Should have some finite log-probabilities"

    print(f"Average test log-probability: {np.mean(finite_log_probs):.4f}")


def test_world_model_sampling():
    """
    Test that the PoE World Model can sample next states.
    """
    # Use correct experts only for cleaner sampling test
    fitter = MaxLikelihoodWeightFitter(max_iterations=10)
    transitions = generate_random_data(200, seed=456)

    weighted_experts = fitter.fit(CORRECT_EXPERTS, transitions)
    world_model = PoEWorldModel(weighted_experts)

    # Sample from the initial state
    initial = initial_state(WorldConfig(seed=789))

    # Try sampling with different actions
    for action in [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.STAY]:
        sampled_state = world_model.sample_next_state(initial, action)

        # Basic sanity checks
        assert isinstance(sampled_state, GameState)
        assert 0 <= sampled_state.player.position < sampled_state.config.width
        assert len(sampled_state.lights) == len(initial.lights)

        for light in sampled_state.lights:
            assert isinstance(light.is_on, bool)

        print(
            f"Action {action}: player moved from {initial.player.position} to {sampled_state.player.position}"
        )


if __name__ == "__main__":
    # Run the tests manually if executed directly
    test_weight_fitting_distinguishes_good_from_bad_experts()
    test_world_model_evaluation()
    test_world_model_sampling()
    print("All tests passed!")
