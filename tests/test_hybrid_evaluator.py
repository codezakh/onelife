"""
Tests for the hybrid evaluation framework.

This module tests the core functionality of the hybrid evaluation framework,
including sanity checks with baseline models and integration tests.
"""

from distant_sunburn.evaluator import (
    HybridEvaluator,
    EvaluationConfig,
    Environment1DAdapter,
    TrueTransitionWorldModel,
    NullWorldModel,
)
from distant_sunburn.evaluator.simple_1d_env.components import (
    JSONPatchEditDistance,
    Semantic1DDistractorGenerator,
)
from distant_sunburn.evaluator.core import SymbolicTransition
from distant_sunburn.poe_world.benchmark_1d.environment import (
    WorldConfig,
    Action,
    initial_state,
)


def test_true_vs_null_world_model():
    """True transition function should vastly outperform null model."""

    # Setup components
    config = WorldConfig(width=12, switch_point=6)
    adapter = Environment1DAdapter(config=config, seed=42)
    environment = adapter.create_environment()

    # Create world models
    true_model = TrueTransitionWorldModel(
        environment, equal_fn=lambda x, y: x == y
    )  # Perfect model
    null_model = NullWorldModel(
        equal_fn=lambda x, y: x == y
    )  # Always predicts no change

    # Create evaluator with injected components
    evaluator = HybridEvaluator(
        config=EvaluationConfig(num_transitions=50, num_distractors=3),
        trajectory_collector=adapter.create_trajectory_collector(),
        edit_distance_calc=adapter.create_edit_distance_calculator(),
        distractor_generator=adapter.create_distractor_generator(),
    )

    # Evaluate both models
    true_results = evaluator.evaluate(true_model, environment)
    null_results = evaluator.evaluate(null_model, environment)

    # Assertions - true model should dominate
    assert true_results.mean_generative_error < null_results.mean_generative_error
    assert true_results.discriminative_accuracy > null_results.discriminative_accuracy
    assert true_results.discriminative_accuracy > 0.9  # Near perfect
    # Check that the difference in discriminative accuracy is greater than 0.5
    assert (
        true_results.discriminative_accuracy - null_results.discriminative_accuracy
    ) > 0.5


def test_edit_distance_calculation():
    """Test that edit distance calculation works."""

    calc = JSONPatchEditDistance()

    # Create two different states
    state1 = initial_state(seed=1)
    state2 = initial_state(seed=2)

    # Calculate distance
    distance = calc.compute_distance(state1, state2)

    assert distance >= 0

    # Distance to self should be 0
    self_distance = calc.compute_distance(state1, state1)
    assert self_distance == 0


def test_distractor_generation():
    """Test that distractor generation works."""

    config = WorldConfig(width=8, switch_point=4)
    generator = Semantic1DDistractorGenerator(config)

    # Create a sample transition
    state1 = initial_state(seed=1)
    state2 = initial_state(seed=2)
    transition = SymbolicTransition(state1, Action.MOVE_RIGHT, state2)

    # Generate distractors
    generator.generate_distractors(transition, [transition], num_distractors=3)


def test_baseline_world_models():
    """Test that baseline world models work correctly."""

    config = WorldConfig(width=8, switch_point=4)
    adapter = Environment1DAdapter(config=config, seed=42)
    environment = adapter.create_environment()

    # Test true transition model
    true_model = TrueTransitionWorldModel(environment, equal_fn=lambda x, y: x == y)
    state = adapter.create_environment().transition(
        initial_state(seed=1), Action.MOVE_RIGHT
    )

    # Should predict the same as environment
    pred_state = true_model.sample_next_state(initial_state(seed=1), Action.MOVE_RIGHT)
    assert pred_state.player.position == state.player.position

    # Test null model
    null_model = NullWorldModel(equal_fn=lambda x, y: x == y)
    null_pred = null_model.sample_next_state(initial_state(seed=1), Action.MOVE_RIGHT)
    # Should predict no change
    assert null_pred.player.position == initial_state(seed=1).player.position


def test_deterministic_evaluation():
    """Test that evaluation results are deterministic with same seed."""

    config = WorldConfig(width=8, switch_point=4)
    adapter = Environment1DAdapter(config=config, seed=42)
    environment = adapter.create_environment()

    true_model = TrueTransitionWorldModel(environment, equal_fn=lambda x, y: x == y)

    evaluator = HybridEvaluator(
        config=EvaluationConfig(num_transitions=20, num_distractors=2),
        trajectory_collector=adapter.create_trajectory_collector(),
        edit_distance_calc=adapter.create_edit_distance_calculator(),
        distractor_generator=adapter.create_distractor_generator(),
    )

    # Run evaluation twice with same seed
    results1 = evaluator.evaluate(true_model, environment)
    results2 = evaluator.evaluate(true_model, environment)

    # Results should be identical
    assert results1.mean_generative_error == results2.mean_generative_error
    assert results1.discriminative_accuracy == results2.discriminative_accuracy
    assert results1.total_transitions_evaluated == results2.total_transitions_evaluated


def test_evaluation_with_different_configs():
    """Test that evaluation works with different configurations."""

    config = WorldConfig(width=8, switch_point=4)
    adapter = Environment1DAdapter(config=config, seed=42)
    environment = adapter.create_environment()
    true_model = TrueTransitionWorldModel(environment, equal_fn=lambda x, y: x == y)

    # Test with different numbers of transitions
    for num_transitions in [10, 20, 50]:
        evaluator = HybridEvaluator(
            config=EvaluationConfig(num_transitions=num_transitions, num_distractors=2),
            trajectory_collector=adapter.create_trajectory_collector(),
            edit_distance_calc=adapter.create_edit_distance_calculator(),
            distractor_generator=adapter.create_distractor_generator(),
        )

        results = evaluator.evaluate(true_model, environment)
        assert results.total_transitions_evaluated == num_transitions
