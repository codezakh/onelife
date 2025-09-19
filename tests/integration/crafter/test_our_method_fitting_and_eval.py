import copy
import random
from typing import List

import numpy as np
from crafter.functional_env import EnvConfig, initial_state, transition
from crafter.state_export import WorldState
from loguru import logger

from distant_sunburn.evaluator import (
    EvaluationConfig,
    Evaluator,
    NullWorldModel,
    TrueTransitionWorldModel,
)
from distant_sunburn.evaluator.baselines import RandomWorldModel
from distant_sunburn.evaluator.crafter.components import _gamestate_to_json
from distant_sunburn.evaluator.crafter.factory import CrafterEvaluationFactory
from distant_sunburn.evaluator.crafter.utils import MAP_ACTION_TO_INDEX
from distant_sunburn.our_method.core import SymbolicTransition
from distant_sunburn.our_method.crafter.handwritten_laws import ALL_EXPERTS
from distant_sunburn.our_method.crafter.observable_extractor import ObservableExtractor
from distant_sunburn.our_method.optimization import MaxLikelihoodWeightFitter
from distant_sunburn.our_method.world_modeling import LawMixture
from rich.console import Console
from rich.table import Table
from rich.text import Text


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
            prev_state=current_state, action=action, next_state=next_state
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
    learned_world_model = LawMixture(
        observable_extractor=ObservableExtractor(),
        weighted_laws=weighted_experts,
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

    eval_seed = 42
    # Create evaluation factory with different seed for each run
    evaluation_factory = CrafterEvaluationFactory(
        env_config=env_config, policy_seed=eval_seed
    )
    evaluation_context = evaluation_factory.create_context(
        config=EvaluationConfig(num_distractors=10, num_trials=5),
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

    # Print all expert weights for debugging
    print("\nAll expert weights:")
    for i, weighted_expert in enumerate(weighted_experts):
        expert_name = weighted_expert.law.__name__
        print(f"  {expert_name}: {weighted_expert.weight}")

    # Print all performance metrics with statistics in a rich table
    console = Console()
    metrics_table = Table(title="World Model Performance Comparison")

    # Add columns
    metrics_table.add_column("Model", style="cyan", no_wrap=True)
    metrics_table.add_column("Edit Distance (Raw)", justify="right", style="magenta")
    metrics_table.add_column(
        "Edit Distance (Normalized)", justify="right", style="magenta"
    )
    metrics_table.add_column("Edit Distance (IoU)", justify="right", style="magenta")
    metrics_table.add_column("Discriminative Accuracy", justify="right", style="green")
    metrics_table.add_column("Normalized Recall", justify="right", style="blue")
    metrics_table.add_column("Reciprocal Rank", justify="right", style="blue")

    # Add rows for each model with styling: gray out true model and bold best non-true
    models = [
        ("True World Model", true_wm_perf, True),
        ("Null World Model", null_wm_perf, False),
        ("Our World Model", learned_wm_perf, False),
        ("Random World Model", random_wm_perf, False),
    ]

    # Choose best non-true model by highest discriminative accuracy
    non_true_models = [(n, p) for (n, p, is_true) in models if not is_true]
    best_non_true_name = None
    if non_true_models:
        best_non_true_name = max(
            non_true_models, key=lambda x: x[1].discriminative_accuracy
        )[0]

    for model_name, performance, is_true in models:
        styled_name = Text(model_name)
        if is_true:
            styled_name.stylize("grey50")
        elif model_name == best_non_true_name:
            styled_name.stylize("bold")

        metrics_table.add_row(
            styled_name,
            f"{performance.edit_distance.raw:.3f} ({performance.edit_distance_std.raw:.3f})",
            f"{performance.edit_distance.normalized:.3f} ({performance.edit_distance_std.normalized:.3f})",
            f"{performance.edit_distance.intersection_over_union:.3f} ({performance.edit_distance_std.intersection_over_union:.3f})",
            f"{performance.discriminative_accuracy:.3f} ({performance.discriminative_accuracy_std:.3f})",
            f"{performance.normalized_recall:.3f} ({performance.normalized_recall_std:.3f})",
            f"{performance.reciprocal_rank:.3f} ({performance.reciprocal_rank_std:.3f})",
        )

    console.print(metrics_table)

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

    console = Console()
    table = Table(title="Learned World Model Metrics by Scenario")

    # Add columns
    table.add_column("Scenario", style="cyan", no_wrap=True)
    table.add_column("Edit Distance (Raw)", justify="right", style="magenta")
    table.add_column("Edit Distance (Normalized)", justify="right", style="magenta")
    table.add_column("Edit Distance (IoU)", justify="right", style="magenta")
    table.add_column("Discriminative Accuracy", justify="right", style="green")
    table.add_column("Normalized Recall", justify="right", style="blue")
    table.add_column("Reciprocal Rank", justify="right", style="blue")
    table.add_column("N Distractors", justify="right", style="yellow")

    # Percentile-based colorization for scenario rows; require non-empty metrics
    by_source = learned_wm_perf.metrics_by_source
    assert by_source, "Expected non-empty metrics_by_source for scenario table"

    mean_raw = np.array([m["mean"].edit_distance.raw for m in by_source.values()])
    mean_norm = np.array(
        [m["mean"].edit_distance.normalized for m in by_source.values()]
    )
    mean_iou = np.array(
        [m["mean"].edit_distance.intersection_over_union for m in by_source.values()]
    )
    mean_acc = np.array([m["mean"].discriminative_accuracy for m in by_source.values()])
    mean_recall = np.array([m["mean"].normalized_recall for m in by_source.values()])
    mean_rr = np.array([m["mean"].reciprocal_rank for m in by_source.values()])

    def thresholds(arr: np.ndarray) -> tuple[float, float]:
        return (float(np.nanpercentile(arr, 33)), float(np.nanpercentile(arr, 66)))

    raw_t = thresholds(mean_raw)
    norm_t = thresholds(mean_norm)
    iou_t = thresholds(mean_iou)
    acc_t = thresholds(mean_acc)
    recall_t = thresholds(mean_recall)
    rr_t = thresholds(mean_rr)

    def color_for(value: float, t: tuple[float, float], higher_is_better: bool) -> str:
        low, high = t
        if higher_is_better:
            if value <= low:
                return "red3"
            if value >= high:
                return "green3"
            return "yellow3"
        else:
            if value <= low:
                return "green3"
            if value >= high:
                return "red3"
            return "yellow3"

    for scenario_name, metrics in by_source.items():
        mean = metrics["mean"]
        std = metrics["std"]
        cells = [
            Text(scenario_name, style="cyan"),
            Text(
                f"{mean.edit_distance.raw:.3f} ({std.edit_distance.raw:.3f})",
                style=color_for(mean.edit_distance.raw, raw_t, higher_is_better=False),
            ),
            Text(
                f"{mean.edit_distance.normalized:.3f} ({std.edit_distance.normalized:.3f})",
                style=color_for(
                    mean.edit_distance.normalized, norm_t, higher_is_better=False
                ),
            ),
            Text(
                f"{mean.edit_distance.intersection_over_union:.3f} ({std.edit_distance.intersection_over_union:.3f})",
                style=color_for(
                    mean.edit_distance.intersection_over_union, iou_t, True
                ),
            ),
            Text(
                f"{mean.discriminative_accuracy:.3f} ({std.discriminative_accuracy:.3f})",
                style=color_for(mean.discriminative_accuracy, acc_t, True),
            ),
            Text(
                f"{mean.normalized_recall:.3f} ({std.normalized_recall:.3f})",
                style=color_for(mean.normalized_recall, recall_t, True),
            ),
            Text(
                f"{mean.reciprocal_rank:.3f} ({std.reciprocal_rank:.3f})",
                style=color_for(mean.reciprocal_rank, rr_t, True),
            ),
            Text(
                f"{mean.n_distractors:.0f} ({std.n_distractors:.0f})",
            ),
        ]
        table.add_row(*cells)

    console.print(table)
