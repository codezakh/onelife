"""
Tests the following for a simple 1D environment:

1. We can learn a world model using PoE-World with data from a policy.
2. That we can evaluate the world model using the evaluator.
3. That the learned world model outperforms a null model, but not a perfect model.
"""

import random
from typing import List

import numpy as np

import distant_sunburn.simple_1d_env.environment
from distant_sunburn.evaluator import (
    EvaluationConfig,
    Evaluator,
    NullWorldModel,
    TrueTransitionWorldModel,
)
from distant_sunburn.evaluator.simple_1d_env.factory import OneDEvaluationFactory
from distant_sunburn.log_utils import change_log_level
from distant_sunburn.poe_world.core import SymbolicTransition
from distant_sunburn.poe_world.simple_1d_env.handwritten_experts import (
    ALL_EXPERTS,
)
from distant_sunburn.poe_world.simple_1d_env.weight_fitter import (
    MaxLikelihoodWeightFitter,
)
from distant_sunburn.poe_world.simple_1d_env.world_model import PoEWorldModel
from distant_sunburn.simple_1d_env.environment import (
    DEFAULT_LAWS,
    Action,
    GameState,
    WorldConfig,
    initial_state,
    transition_function,
)


def generate_random_data(
    world_config: WorldConfig, n_transitions: int, policy_seed: int = 42
) -> List[SymbolicTransition[GameState]]:
    """
    Generate random transitions using the 1D environment.

    Args:
        n_transitions: Number of transitions to generate
        seed: Random seed for reproducibility

    Returns:
        List of symbolic transitions
    """

    rng = random.Random(policy_seed)
    np.random.seed(policy_seed)

    transitions = []
    current_state = initial_state(world_config)

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


def test():
    world_config = WorldConfig()

    # Just to suppress logging from the simple 1d environment.
    with change_log_level(
        {
            "INFO": [distant_sunburn.simple_1d_env.environment],
        }
    ):
        # First we generate some data from a random policy and fit the world model.
        transitions = generate_random_data(
            world_config, n_transitions=750, policy_seed=42
        )
        fitter = MaxLikelihoodWeightFitter(
            learning_rate=0.1, max_iterations=25, batch_size=200, l1_weight=0.001
        )

        weighted_experts = fitter.fit(ALL_EXPERTS, transitions)
        learned_world_model = PoEWorldModel(weighted_experts)

        # Now we create an evaluation factory for evaluating the world model.
        evaluation_factory = OneDEvaluationFactory(
            world_config=world_config, policy_seed=42
        )
        evaluation_context = evaluation_factory.create_context(
            config=EvaluationConfig(num_distractors=3), num_transitions=50
        )

        evaluator = Evaluator(evaluation_context)
        learned_wm_perf = evaluator.evaluate(learned_world_model)

        true_model = TrueTransitionWorldModel(
            evaluation_factory.environment, equal_fn=lambda x, y: x == y
        )  # Perfect model
        null_model = NullWorldModel(
            equal_fn=lambda x, y: x == y
        )  # Always predicts no change

        true_wm_perf = evaluator.evaluate(true_model)
        null_wm_perf = evaluator.evaluate(null_model)

        assert (
            learned_wm_perf.mean_generative_error < null_wm_perf.mean_generative_error
        )
        assert (
            learned_wm_perf.discriminative_accuracy
            > null_wm_perf.discriminative_accuracy
        )

        assert (
            learned_wm_perf.mean_generative_error > true_wm_perf.mean_generative_error
        )
        assert (
            learned_wm_perf.discriminative_accuracy
            < true_wm_perf.discriminative_accuracy
        )
