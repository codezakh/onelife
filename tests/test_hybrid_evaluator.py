"""
Tests for the hybrid evaluation framework.

This module tests the core functionality of the hybrid evaluation framework,
including sanity checks with baseline models and integration tests.
"""

from distant_sunburn.evaluator import (
    Evaluator,
    EvaluationConfig,
    Environment1DAdapter,
    TrueTransitionWorldModel,
    NullWorldModel,
)
from distant_sunburn.evaluator.simple_1d_env.components import (
    Semantic1DDistractorGenerator,
    JSONPatchEditDistance,
)
from distant_sunburn.evaluator.core import SymbolicTransition, EvaluationContext
from distant_sunburn.poe_world.benchmark_1d.environment import (
    WorldConfig,
    Action,
    initial_state,
)


def test_true_vs_null_world_model():
    """True transition function should vastly outperform null model."""

    config = WorldConfig(width=12, switch_point=6)
    adapter = Environment1DAdapter(world_config=config, policy_seed=42)
    environment = adapter.create_environment()

    true_model = TrueTransitionWorldModel(
        environment, equal_fn=lambda x, y: x == y
    )  # Perfect model
    null_model = NullWorldModel(
        equal_fn=lambda x, y: x == y
    )  # Always predicts no change

    transitions = adapter.create_trajectory_collector().collect_transitions(
        environment, num_transitions=50
    )

    evaluation_context = EvaluationContext(
        config=EvaluationConfig(num_distractors=3),
        test_transitions=transitions,
        distractor_generator=adapter.create_distractor_generator(),
        edit_distance_calculator=adapter.create_edit_distance_calculator(),
    )

    evaluator = Evaluator(context=evaluation_context)

    true_results = evaluator.evaluate(true_model)
    null_results = evaluator.evaluate(null_model)

    assert true_results.mean_generative_error < null_results.mean_generative_error
    assert true_results.discriminative_accuracy > null_results.discriminative_accuracy
    assert true_results.discriminative_accuracy > 0.9  # Near perfect
    assert (
        true_results.discriminative_accuracy - null_results.discriminative_accuracy
    ) > 0.5


def test_edit_distance_calculation():
    """Test that edit distance calculation works."""

    # Create two different states
    state1 = initial_state(WorldConfig(seed=1))
    state2 = initial_state(WorldConfig(seed=2))

    calc = JSONPatchEditDistance()

    # Calculate distance
    distance = calc(state1, state2)

    assert distance >= 0

    # Distance to self should be 0
    self_distance = calc(state1, state1)
    assert self_distance == 0


def test_distractor_generation():
    """Test that distractor generation works."""

    config = WorldConfig(width=8, switch_point=4)
    generator = Semantic1DDistractorGenerator(config)

    # Create a sample transition
    state1 = initial_state(WorldConfig(seed=1))
    state2 = initial_state(WorldConfig(seed=2))
    transition = SymbolicTransition(state1, Action.MOVE_RIGHT, state2)

    # Generate distractors
    distractors = generator(transition, [transition], num_distractors=3)
    assert len(distractors) == 3


def test_baseline_world_models():
    """Test that baseline world models work correctly."""

    config = WorldConfig(width=8, switch_point=4)
    adapter = Environment1DAdapter(world_config=config, policy_seed=42)
    environment = adapter.create_environment()

    # Test true transition model
    true_model = TrueTransitionWorldModel(environment, equal_fn=lambda x, y: x == y)
    state = environment(initial_state(WorldConfig(seed=1)), Action.MOVE_RIGHT)

    # Should predict the same as environment
    pred_state = true_model.sample_next_state(
        initial_state(WorldConfig(seed=1)), Action.MOVE_RIGHT
    )
    assert pred_state.player.position == state.player.position

    # Test null model
    null_model = NullWorldModel(equal_fn=lambda x, y: x == y)
    null_pred = null_model.sample_next_state(
        initial_state(WorldConfig(seed=1)), Action.MOVE_RIGHT
    )
    # Should predict no change
    assert (
        null_pred.player.position == initial_state(WorldConfig(seed=1)).player.position
    )


def test_deterministic_evaluation():
    """Test that evaluation results are deterministic with same seed."""

    config = WorldConfig(width=8, switch_point=4)
    adapter = Environment1DAdapter(world_config=config, policy_seed=42)
    environment = adapter.create_environment()

    true_model = TrueTransitionWorldModel(environment, equal_fn=lambda x, y: x == y)

    test_transitions = adapter.create_trajectory_collector().collect_transitions(
        environment, num_transitions=20
    )

    evaluation_context = EvaluationContext(
        config=EvaluationConfig(num_distractors=2),
        test_transitions=test_transitions,
        distractor_generator=adapter.create_distractor_generator(),
        edit_distance_calculator=adapter.create_edit_distance_calculator(),
    )

    evaluator = Evaluator(context=evaluation_context)

    # Run evaluation twice with same seed
    results1 = evaluator.evaluate(true_model)
    results2 = evaluator.evaluate(true_model)

    # Results should be identical
    assert results1.mean_generative_error == results2.mean_generative_error
    assert results1.discriminative_accuracy == results2.discriminative_accuracy
    assert results1.total_transitions_evaluated == results2.total_transitions_evaluated
