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

from typing import Callable
from loguru import logger
from distant_sunburn.log_utils import change_log_level
from distant_sunburn.evaluator import (
    Evaluator,
    EvaluationConfig,
    TrueTransitionWorldModel,
    NullWorldModel,
)
from distant_sunburn.evaluator.simple_1d_env.factory import OneDEvaluationFactory


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
    import distant_sunburn.simple_1d_env.environment

    with change_log_level(
        {
            "INFO": [distant_sunburn.simple_1d_env.environment],
        }
    ):
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


import pytest


@pytest.mark.skip(reason="in progress")
def test():
    world_config = WorldConfig()

    # First we generate some data from a random policy and fit the world model.
    transitions = generate_random_data(world_config, n_transitions=750, policy_seed=42)
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
    results = evaluator.evaluate(learned_world_model)

    assert results.mean_generative_error < 0.1
    assert results.discriminative_accuracy > 0.9
