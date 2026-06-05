import argparse
import ast
import sys
from pathlib import Path
from typing import Any, Dict, List, cast

import cloudpickle
import numpy as np
from crafter_oo.functional_env import EnvConfig, transition
from crafter_oo.state_export import WorldState

import onelife
import crafter_oo
from onelife.balrog_evaluator import TrajectoryStep
from onelife.evaluator import (
    EvaluationConfig,
    EvaluationResults,
    Evaluator,
    NullWorldModel,
    TrueTransitionWorldModel,
)
from onelife.evaluator.baselines import RandomWorldModel
from onelife.evaluator.crafter.components import _gamestate_to_json
from onelife.evaluator.crafter.factory import CrafterEvaluationFactory
from onelife.evaluator.crafter.utils import MAP_ACTION_TO_INDEX
from onelife.io_utils import PydanticJSONLinesReader
from onelife.local_code_execution import ExecWithLimitedNamespace
from onelife.our_method.core import (
    LawFunctionWrapper,
    LawProtocol,
    SymbolicTransition,
)
from onelife.our_method.crafter.observable_extractor import ObservableExtractor
from onelife.our_method.optimization import MaxLikelihoodWeightFitter
from onelife.our_method.world_modeling import LawMixture
from onelife.poe_world.core import DiscreteDistribution
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.text import Text
from tqdm.auto import tqdm


from onelife.law_synthesis.core import LawInduction
from onelife.our_method.action_remapping import (
    remap_slug_actions_to_balrog_actions,
)
from onelife.pipeline_utils import PydanticJsonLinesFileTarget

# Map the old package names onto the current ones so cloudpickle can load
# world-model pickles written before the distant_sunburn -> onelife and
# crafter -> crafter_oo renames.
sys.modules["distant_sunburn"] = onelife
sys.modules["crafter"] = crafter_oo


def convert_trajectory_steps_to_transitions(
    trajectory_steps: List[TrajectoryStep[WorldState]],
) -> List[SymbolicTransition[WorldState]]:
    """
    Convert trajectory steps to SymbolicTransition objects.
    Follows the same pattern as regroup_trajectory_steps in experiment 20.

    Args:
        trajectory_steps: List of trajectory steps from experiment 19

    Returns:
        List of symbolic transitions (s_t, a_t, s_{t+1})
    """
    transitions: List[SymbolicTransition[WorldState]] = []

    # Follow the validated pattern from experiment 20: start from index 1
    for i in range(1, len(trajectory_steps)):
        transitions.append(
            SymbolicTransition(
                prev_state=trajectory_steps[i - 1].info,  # Previous state
                action=trajectory_steps[i].action,  # Action that led to next state
                next_state=trajectory_steps[i].info,  # Current state (result of action)
            )
        )

    return transitions


def extract_class_name_from_code(code: str) -> str:
    """
    Extract the class name from Python code using AST.

    Args:
        code: Python source code containing a class definition

    Returns:
        The name of the first class found in the code

    Raises:
        ValueError: If no class definition is found
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                return node.name
        raise ValueError("No class definition found in code")
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax in code: {e}")


def materialize_law(law_induction: LawInduction) -> LawProtocol[WorldState]:
    """
    Materialize a law from its code representation into a LawProtocol instance.

    Args:
        law_induction: The law induction containing the code

    Returns:
        A LawProtocol instance that can be used for world modeling
    """
    # Extract class name from the law code
    class_name = extract_class_name_from_code(law_induction.law_code)

    # Create execution namespace with required symbols
    # Import all the necessary classes from state_export and poe_world
    from crafter_oo.state_export import (
        Achievements,
        ArrowState,
        BaseObjectState,
        ChunkState,
        CowState,
        FenceState,
        Inventory,
        PlantState,
        PlayerState,
        Position,
        SkeletonState,
        WorldState,
        ZombieState,
    )

    # Create the execution namespace
    execution_namespace = {
        # Core classes
        "WorldState": WorldState,
        "PlayerState": PlayerState,
        "CowState": CowState,
        "ZombieState": ZombieState,
        "SkeletonState": SkeletonState,
        "ArrowState": ArrowState,
        "PlantState": PlantState,
        "FenceState": FenceState,
        "Position": Position,
        "Inventory": Inventory,
        "Achievements": Achievements,
        "BaseObjectState": BaseObjectState,
        "ChunkState": ChunkState,
        # DiscreteDistribution for probabilistic predictions
        "DiscreteDistribution": DiscreteDistribution,
        # Standard library
        "np": np,
        "Any": Any,
        "List": List,
        "Dict": Dict,
        "Optional": type(None),
        "Union": type(None),
        "Tuple": tuple,
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
    }

    # Create executor with the namespace
    executor = ExecWithLimitedNamespace(
        allowed_names=set(execution_namespace.keys()),
        inherited_scope=execution_namespace,
    )

    # Execute the law code
    executor(law_induction.law_code)

    # Extract the class from the namespace
    if class_name not in executor.namespace:
        raise ValueError(f"Class {class_name} not found in executed namespace")

    law_class = executor.namespace[class_name]

    # Create an instance of the law
    law_instance = law_class()

    # Wrap it in LawFunctionWrapper
    wrapped_law = LawFunctionWrapper(
        law=law_instance,
        source_code=law_induction.law_code,
        action_remapper=remap_slug_actions_to_balrog_actions,
    )

    return wrapped_law


def gather_induced_laws(laws_dir: Path) -> List[LawInduction]:
    """Gather all induced laws from the synthesis step's output directory."""
    laws: List[LawInduction] = []
    logger.info(f"Gathering induced laws from {laws_dir}")
    law_blobs = list(laws_dir.glob("laws/*.jsonl"))
    logger.info(f"Found {len(law_blobs)} law blobs")
    for law_blob in tqdm(law_blobs, desc="Gathering induced laws"):
        reader = PydanticJSONLinesReader(law_blob, LawInduction)
        laws.extend(list(reader))
    return laws


IGNORE_LAWS = {"ProbabilisticArrowMovement", "ZombieSpawnEvent"}  # Errors out


def main(
    trajectory_path: Path,
    laws_dir: Path,
    output_dir: Path,
):
    """Fit world-model law weights from induced laws, then evaluate."""
    logger.info("Starting law optimization and evaluation")

    output_dir.mkdir(parents=True, exist_ok=True)

    induced_laws: List[LawInduction] = gather_induced_laws(laws_dir)
    logger.info(f"Loaded {len(induced_laws)} induced laws")

    # Materialize laws into LawProtocol instances
    materialized_laws: List[LawProtocol[WorldState]] = []
    for i, law_induction in enumerate(induced_laws):
        try:
            logger.info(
                f"Materializing law {i+1}/{len(induced_laws)}: {law_induction.natural_language_law[:50]}..."
            )
            materialized_law = materialize_law(law_induction)
            if materialized_law.__name__ in IGNORE_LAWS:
                logger.warning(f"Ignoring law: {materialized_law.__name__}")
                continue
            materialized_laws.append(materialized_law)
            logger.info(f"Successfully materialized law: {materialized_law.__name__}")
        except Exception as e:
            logger.error(f"Failed to materialize law {i+1}: {e}")
            logger.error(f"Law code: {law_induction.law_code}")
            continue

    if not materialized_laws:
        raise RuntimeError("No laws were successfully materialized")

    logger.info(f"Successfully materialized {len(materialized_laws)} laws")

    trajectory_steps_target = PydanticJsonLinesFileTarget(
        file_path=trajectory_path,
        model=TrajectoryStep[WorldState],
    )
    trajectory_steps: List[TrajectoryStep[WorldState]] = trajectory_steps_target.load()
    logger.info(f"Loaded {len(trajectory_steps)} trajectory steps")

    # Check if fitted world model already exists
    world_model_path = output_dir / "fitted_world_model.pkl"

    if world_model_path.exists():
        logger.info(f"Loading existing fitted world model from: {world_model_path}")
        with open(world_model_path, "rb") as f:
            learned_world_model = cast(LawMixture[WorldState, Any], cloudpickle.load(f))
            weighted_laws = learned_world_model.laws
        logger.info("Successfully loaded fitted world model from pickle")
    else:
        logger.info("No existing fitted world model found, fitting new model...")

        # Convert trajectory steps to SymbolicTransition objects
        transitions = convert_trajectory_steps_to_transitions(trajectory_steps)
        logger.info(f"Converted to {len(transitions)} training transitions")

        # Fit the world model using the MaxLikelihoodWeightFitter
        fitter = MaxLikelihoodWeightFitter(
            observable_extractor=ObservableExtractor(),
            learning_rate=0.1,
            max_iterations=3,
            batch_size=1000,
            l1_weight=0.001,
        )

        # Fit weights for the materialized laws
        logger.info("Fitting weights for induced laws...")
        weighted_laws = fitter.fit(materialized_laws, transitions)
        logger.info(f"Fitted weights for {len(weighted_laws)} laws")

        # Print law weights for debugging
        logger.info("Law weights after fitting:")
        for i, weighted_law in enumerate(weighted_laws):
            logger.info(f"  {weighted_law.law.__name__}: {weighted_law.weight:.4f}")

        # Create the world model with fitted weights
        learned_world_model = LawMixture(
            observable_extractor=ObservableExtractor(),
            weighted_laws=weighted_laws,
        )

        # Save the fitted world model using Cloud Pickle
        with open(world_model_path, "wb") as f:
            cloudpickle.dump(learned_world_model, f)
        logger.info(f"Saved fitted world model to: {world_model_path}")

    # Create comparison models
    def equality_check(state1: WorldState, state2: WorldState) -> bool:
        return _gamestate_to_json(state1) == _gamestate_to_json(state2)

    def wrap_true_transition_fn(state: WorldState, action) -> WorldState:
        next_state, _ = transition(state, MAP_ACTION_TO_INDEX[action])
        return next_state

    true_model = TrueTransitionWorldModel(wrap_true_transition_fn, equality_check)
    null_model = NullWorldModel(equality_check)
    random_world_model = RandomWorldModel()

    # Create evaluation context using the same environment config as the trajectory data
    # Extract env config from the first trajectory step
    first_step = trajectory_steps[0]
    env_config = EnvConfig(size=first_step.info.size, view=first_step.info.view)

    eval_seed = 42
    evaluation_factory = CrafterEvaluationFactory(
        env_config=env_config, policy_seed=eval_seed
    )
    # Check if evaluation results already exist
    evaluation_results_path = output_dir / "evaluation_results.pkl"

    if evaluation_results_path.exists():
        logger.info(
            f"Loading existing evaluation results from: {evaluation_results_path}"
        )
        with open(evaluation_results_path, "rb") as f:
            evaluation_results = cast(Dict[str, Any], cloudpickle.load(f))

        # Extract individual performance results
        learned_wm_perf = cast(
            EvaluationResults, evaluation_results["learned_world_model_performance"]
        )
        true_wm_perf = cast(
            EvaluationResults, evaluation_results["true_world_model_performance"]
        )
        null_wm_perf = cast(
            EvaluationResults, evaluation_results["null_world_model_performance"]
        )
        random_wm_perf = cast(
            EvaluationResults, evaluation_results["random_world_model_performance"]
        )

        logger.info("Successfully loaded evaluation results from pickle")
    else:
        logger.info("No existing evaluation results found, running evaluations...")

        evaluation_context = evaluation_factory.create_context(
            config=EvaluationConfig(num_distractors=10, num_trials=5),
            num_transitions_per_scenario=30,
        )
        evaluator = Evaluator(evaluation_context)

        # Evaluate all models
        with logger.contextualize(world_model="learned"):
            learned_wm_perf = evaluator.evaluate(learned_world_model)

        with logger.contextualize(world_model="true"):
            true_wm_perf = evaluator.evaluate(true_model)

        with logger.contextualize(world_model="null"):
            null_wm_perf = evaluator.evaluate(null_model)

        with logger.contextualize(world_model="random"):
            random_wm_perf = evaluator.evaluate(random_world_model)

        # Save evaluation results using Cloud Pickle
        evaluation_results = {
            "learned_world_model_performance": learned_wm_perf,
            "true_world_model_performance": true_wm_perf,
            "null_world_model_performance": null_wm_perf,
            "random_world_model_performance": random_wm_perf,
            "evaluation_config": {
                "num_distractors": 10,
                "num_trials": 5,
                "num_transitions_per_scenario": 30,
                "eval_seed": eval_seed,
            },
        }

        with open(evaluation_results_path, "wb") as f:
            cloudpickle.dump(evaluation_results, f)
        logger.info(f"Saved evaluation results to: {evaluation_results_path}")

    # Print results
    console = Console()
    metrics_table = Table(title="World Model Performance Comparison (Induced Laws)")

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

    # Add rows for each model
    models = [
        ("True World Model", true_wm_perf, True),
        ("Null World Model", null_wm_perf, False),
        ("Induced Laws Model", learned_wm_perf, False),
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

    # Print per-scenario breakdown for the learned model
    scenario_table = Table(title="Induced Laws Model Metrics by Scenario")

    # Add columns
    scenario_table.add_column("Scenario", style="cyan", no_wrap=True)
    scenario_table.add_column("Edit Distance (Raw)", justify="right", style="magenta")
    scenario_table.add_column(
        "Edit Distance (Normalized)", justify="right", style="magenta"
    )
    scenario_table.add_column("Edit Distance (IoU)", justify="right", style="magenta")
    scenario_table.add_column("Discriminative Accuracy", justify="right", style="green")
    scenario_table.add_column("Normalized Recall", justify="right", style="blue")
    scenario_table.add_column("Reciprocal Rank", justify="right", style="blue")
    scenario_table.add_column("N Distractors", justify="right", style="yellow")

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
        scenario_table.add_row(*cells)

    console.print(scenario_table)

    # Print law information with weights
    console.print(
        f"\nSuccessfully used {len(weighted_laws)} induced laws with fitted weights:"
    )
    for i, weighted_law in enumerate(weighted_laws):
        console.print(
            f"  {i+1}. {weighted_law.law.__name__}: {weighted_law.weight:.4f}"
        )

    # Print all expert weights for debugging (similar to test file)
    console.print("\nAll law weights:")
    for i, weighted_law in enumerate(weighted_laws):
        law_name = weighted_law.law.__name__
        console.print(f"  {law_name}: {weighted_law.weight:.4f}")

    logger.info("Experiment 21 completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit induced-law weights and evaluate the world model."
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path(
            "runbook_output/01_exploration/crafter/open_ended/open_ended_run_00.jsonl"
        ),
        help="Path to the exploration trajectory produced by step 1.",
    )
    parser.add_argument(
        "--laws-dir",
        type=Path,
        default=Path("runbook_output/02_synthesis"),
        help="Step 2 output directory (the one containing laws/*.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runbook_output/03_optimize_and_evaluate"),
        help="Directory to write the fitted world model and evaluation results.",
    )
    args = parser.parse_args()

    main(
        trajectory_path=args.trajectory,
        laws_dir=args.laws_dir,
        output_dir=args.output_dir,
    )
