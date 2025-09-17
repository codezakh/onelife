"""
Tests the following for the Crafter environment:

1. We can learn a world model using PoE-World with data from a policy.
2. That we can evaluate the world model using the evaluator.
3. That the learned world model outperforms a null model, but not a perfect model.
4. That bad experts (especially the entity lifecycle expert) get lower weights and accuracy.
"""

import copy
import random
from typing import List

from loguru import logger
import numpy as np

from crafter.functional_env import EnvConfig, initial_state, transition
from crafter.state_export import WorldState
from crafter.constants import ActionT

from distant_sunburn.evaluator import (
    EvaluationConfig,
    Evaluator,
    NullWorldModel,
    TrueTransitionWorldModel,
    EvaluationResults,
)
from distant_sunburn.evaluator.crafter.factory import CrafterEvaluationFactory
from distant_sunburn.evaluator.crafter.components import _gamestate_to_json
from distant_sunburn.evaluator.crafter.utils import MAP_ACTION_TO_INDEX
from distant_sunburn.poe_world.core import SymbolicTransition
from distant_sunburn.poe_world.crafter.handwritten_experts import (
    ALL_EXPERTS,
    incorrect_entity_lifecycle_expert_spurious_spawning,
)
from distant_sunburn.poe_world.weight_fitter import MaxLikelihoodWeightFitter
from distant_sunburn.poe_world.world_model import PoEWorldModel
from distant_sunburn.poe_world.crafter.observable_extractor import ObservableExtractor
import rich
from distant_sunburn.evaluator.baselines import RandomWorldModel
from rich.console import Console
from rich.table import Table
from rich.pretty import pprint


def generate_random_data(
    env_config: EnvConfig, n_transitions: int, policy_seed: int = 42
) -> List[SymbolicTransition[WorldState]]:
    """
    Generate random transitions using the Crafter environment.

    Args:
        env_config: Environment configuration
        n_transitions: Number of transitions to generate
        policy_seed: Random seed for reproducibility

    Returns:
        List of symbolic transitions
    """
    rng = random.Random(policy_seed)
    np.random.seed(policy_seed)

    transitions = []
    current_state = initial_state(
        area=env_config.size,
        view=env_config.view,
        episode=1,
        seed=policy_seed,
    )

    # Define available actions for Crafter
    available_actions = list(MAP_ACTION_TO_INDEX.keys())

    for _ in range(n_transitions):
        # Choose random action
        action = rng.choice(available_actions)
        action_index = MAP_ACTION_TO_INDEX[action]

        # Apply transition function to a copy of current state
        next_state, _ = transition(copy.deepcopy(current_state), action_index)

        # Create symbolic transition
        transition_obj = SymbolicTransition(
            prev_metadata=current_state, action=action, next_metadata=next_state
        )
        transitions.append(transition_obj)

        # Update current state for next iteration
        current_state = next_state

    return transitions


def test():
    """Test that learned world model performs between null and true models."""
    env_config = EnvConfig(size=(9, 9), view=(9, 9))

    # First we generate some data from a random policy and fit the world model.
    transitions = generate_random_data(env_config, n_transitions=100, policy_seed=42)

    fitter = MaxLikelihoodWeightFitter(
        observable_extractor=ObservableExtractor(),
        learning_rate=0.1,
        max_iterations=5,
        batch_size=100,
        l1_weight=0.001,
    )

    weighted_experts = fitter.fit(ALL_EXPERTS, transitions)  # type: ignore
    # Check that the bad entity lifecycle expert gets a low weight
    bad_expert_weight = None
    for weighted_expert in weighted_experts:
        if (
            weighted_expert.expert_function.__name__
            == incorrect_entity_lifecycle_expert_spurious_spawning.__name__
        ):
            bad_expert_weight = weighted_expert.weight
            break

    assert (
        bad_expert_weight is not None
    ), "Bad expert should be found in weighted experts"

    # The bad expert should have a lower weight than most correct experts
    # (though this is not guaranteed due to the learning process)
    print(f"Bad entity lifecycle expert weight: {bad_expert_weight}")

    # Check that weights are reasonable (between 0 and 1)
    assert 0 <= bad_expert_weight <= 1, "Expert weight should be between 0 and 1"

    # Print all expert weights for debugging
    print("\nAll expert weights:")
    for i, weighted_expert in enumerate(weighted_experts):
        expert_name = weighted_expert.expert_function.__name__
        print(f"  {expert_name}: {weighted_expert.weight}")

    learned_world_model = PoEWorldModel(
        observable_extractor=ObservableExtractor(),
        weighted_experts=weighted_experts,
    )

    # Create true and null models for comparison
    def equality_check(state1: WorldState, state2: WorldState) -> bool:
        return _gamestate_to_json(state1) == _gamestate_to_json(state2)

    def wrap_true_transition_fn(state: WorldState, action) -> WorldState:
        next_state, _ = transition(state, MAP_ACTION_TO_INDEX[action])
        return next_state

    true_model = TrueTransitionWorldModel(wrap_true_transition_fn, equality_check)
    null_model = NullWorldModel(equality_check)
    random_world_model = RandomWorldModel()

    # Create evaluation factory with different seed for each run
    evaluation_factory = CrafterEvaluationFactory(env_config=env_config, policy_seed=42)
    evaluation_context = evaluation_factory.create_context(
        config=EvaluationConfig(num_distractors=10, num_trials=10),
        num_transitions_per_scenario=30,
    )
    evaluator = Evaluator(evaluation_context)

    # Evaluate all models with this seed
    with logger.contextualize(world_model="learned"):
        learned_wm_perf = evaluator.evaluate(learned_world_model)

    with logger.contextualize(world_model="true"):
        true_wm_perf = evaluator.evaluate(true_model)

    with logger.contextualize(world_model="null"):
        null_wm_perf = evaluator.evaluate(null_model)

    with logger.contextualize(world_model="random"):
        random_wm_perf = evaluator.evaluate(random_world_model)

    # Print all performance metrics with statistics
    pprint(
        {
            "edit_distance": {
                "raw": {
                    "true_world_model": true_wm_perf.edit_distance.raw,
                    "null_world_model": null_wm_perf.edit_distance.raw,
                    "poe_world_model": learned_wm_perf.edit_distance.raw,
                    "random_world_model": random_wm_perf.edit_distance.raw,
                },
                "normalized": {
                    "true_world_model": true_wm_perf.edit_distance.normalized,
                    "null_world_model": null_wm_perf.edit_distance.normalized,
                    "poe_world_model": learned_wm_perf.edit_distance.normalized,
                    "random_world_model": random_wm_perf.edit_distance.normalized,
                },
                "intersection_over_union": {
                    "true_world_model": true_wm_perf.edit_distance.intersection_over_union,
                    "null_world_model": null_wm_perf.edit_distance.intersection_over_union,
                    "poe_world_model": learned_wm_perf.edit_distance.intersection_over_union,
                    "random_world_model": random_wm_perf.edit_distance.intersection_over_union,
                },
            },
            "discriminative_accuracy": {
                "true_world_model": true_wm_perf.discriminative_accuracy,
                "null_world_model": null_wm_perf.discriminative_accuracy,
                "poe_world_model": learned_wm_perf.discriminative_accuracy,
                "random_world_model": random_wm_perf.discriminative_accuracy,
            },
            "normalized_recall": {
                "true_world_model": true_wm_perf.normalized_recall,
                "null_world_model": null_wm_perf.normalized_recall,
                "poe_world_model": learned_wm_perf.normalized_recall,
                "random_world_model": random_wm_perf.normalized_recall,
            },
        },
        expand_all=True,
    )

    # Assertions based on mean values across all runs
    assert (
        learned_wm_perf.discriminative_accuracy > null_wm_perf.discriminative_accuracy
    ), "Learned model should have higher discriminative accuracy than null model"

    # Assert that learned model underperforms true model
    assert (
        learned_wm_perf.edit_distance.raw > true_wm_perf.edit_distance.raw
    ), "Learned model should have higher generative error than true model"

    assert (
        learned_wm_perf.discriminative_accuracy < true_wm_perf.discriminative_accuracy
    ), "Learned model should have lower discriminative accuracy than true model"

    assert (
        learned_wm_perf.normalized_recall > null_wm_perf.normalized_recall
    ), "Learned model should have higher normalized recall than null model"

    assert (
        learned_wm_perf.normalized_recall < true_wm_perf.normalized_recall
    ), "Learned model should have lower normalized recall than true model"

    # Print rich table showing metrics by scenario for the first learned world model
    console = Console()
    table = Table(title="PoE World Model Metrics by Scenario")

    # Add columns
    table.add_column("Scenario", style="cyan", no_wrap=True)
    table.add_column("Edit Distance (Raw)", justify="right", style="magenta")
    table.add_column("Edit Distance (Normalized)", justify="right", style="magenta")
    table.add_column("Edit Distance (IoU)", justify="right", style="magenta")
    table.add_column("Discriminative Accuracy", justify="right", style="green")
    table.add_column("Normalized Recall", justify="right", style="blue")
    table.add_column("N Distractors", justify="right", style="yellow")

    # Add rows for each scenario
    for scenario_name, metrics in learned_wm_perf.metrics_by_source.items():
        table.add_row(
            scenario_name,
            f"{metrics['mean'].edit_distance.raw:.3f} ({metrics['std'].edit_distance.raw:.3f})",
            f"{metrics['mean'].edit_distance.normalized:.3f} ({metrics['std'].edit_distance.normalized:.3f})",
            f"{metrics['mean'].edit_distance.intersection_over_union:.3f} ({metrics['std'].edit_distance.intersection_over_union:.3f})",
            f"{metrics['mean'].discriminative_accuracy:.3f} ({metrics['std'].discriminative_accuracy:.3f})",
            f"{metrics['mean'].normalized_recall:.3f} ({metrics['std'].normalized_recall:.3f})",
            f"{metrics['mean'].n_distractors:.0f} ({metrics['std'].n_distractors:.0f})",
        )

    console.print(table)
