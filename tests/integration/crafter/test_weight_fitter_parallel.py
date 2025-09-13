"""
Integration test: compare baseline vs parallelized weight fitting.

This test measures:
1) Runtime speed between non-parallel and parallel implementations
2) Similarity of final losses (computed via common bucketed loss)
3) Similarity of learned expert weights
4) Similarity of evaluation results produced by the learned world models

We use the Crafter environment as a realistic workload, but the fitter path
is extractor-agnostic.
"""

from __future__ import annotations

import copy
import time
from typing import List

import numpy as np
import pytest

from crafter.functional_env import EnvConfig, initial_state, transition
from crafter.state_export import WorldState

from distant_sunburn.evaluator import (
    EvaluationConfig,
    Evaluator,
)
from distant_sunburn.evaluator.crafter.factory import CrafterEvaluationFactory
from distant_sunburn.evaluator.crafter.utils import MAP_ACTION_TO_INDEX
from distant_sunburn.poe_world.core import SymbolicTransition
from distant_sunburn.poe_world.crafter.handwritten_experts import ALL_EXPERTS
from distant_sunburn.poe_world.weight_fitter import MaxLikelihoodWeightFitter
from distant_sunburn.poe_world.world_model import PoEWorldModel
from distant_sunburn.poe_world.crafter.observable_extractor import ObservableExtractor


def _generate_random_data(
    env_config: EnvConfig, n_transitions: int, policy_seed: int = 123
) -> List[SymbolicTransition[WorldState]]:
    """
    Generate a short sequence of random Crafter transitions for testing.
    """
    rng = np.random.RandomState(policy_seed)

    transitions = []
    current_state = initial_state(
        area=env_config.size,
        view=env_config.view,
        episode=1,
        seed=policy_seed,
    )
    available_actions = list(MAP_ACTION_TO_INDEX.keys())

    for _ in range(n_transitions):
        action = available_actions[int(rng.randint(0, len(available_actions)))]
        action_index = MAP_ACTION_TO_INDEX[action]
        next_state, _ = transition(copy.deepcopy(current_state), action_index)
        transitions.append(
            SymbolicTransition(
                prev_metadata=current_state, action=action, next_metadata=next_state
            )
        )
        current_state = next_state

    return transitions


@pytest.mark.flaky(retries=3, delay=0.0)
def test_weight_fitter_parallel_speed_and_equivalence():
    env_config = EnvConfig(size=(9, 9), view=(9, 9))
    transitions = _generate_random_data(env_config, n_transitions=1000, policy_seed=321)
    extractor = ObservableExtractor()

    # Common fitter knobs
    learning_rate = 0.1
    max_iterations = 10
    batch_size = 150
    l1_weight = 0.001

    # Baseline fit
    np.random.seed(777)  # ensure same batch sampling across runs
    fitter_base = MaxLikelihoodWeightFitter(
        observable_extractor=extractor,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        batch_size=batch_size,
        l1_weight=l1_weight,
        use_parallel_loss=False,
    )
    start_base = time.perf_counter()
    weighted_experts_base = fitter_base.fit(ALL_EXPERTS, transitions)  # type: ignore[arg-type]
    dur_base = time.perf_counter() - start_base

    # Parallel fit
    np.random.seed(777)  # same sampling
    fitter_par = MaxLikelihoodWeightFitter(
        observable_extractor=extractor,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        batch_size=batch_size,
        l1_weight=l1_weight,
        use_parallel_loss=True,
    )
    start_par = time.perf_counter()
    weighted_experts_par = fitter_par.fit(ALL_EXPERTS, transitions)  # type: ignore[arg-type]
    dur_par = time.perf_counter() - start_par

    # Compare speed (allow parallel to be at least not slower by a large margin)
    # It's okay if CI noise makes it similar; assert it's not > 2.5x slower.
    assert (
        dur_par <= 2.5 * dur_base
    ), f"Parallel path unexpectedly slow: base={dur_base:.3f}s par={dur_par:.3f}s"

    # Build common predictions and buckets for loss evaluation (same dataset, same extractor)
    preds = fitter_base._precompute_expert_predictions(ALL_EXPERTS, transitions)  # type: ignore[arg-type]
    buckets = fitter_base.build_loss_buckets(transitions, preds)

    # Convert learned weights to tensors in expert order
    import torch

    weights_base = torch.tensor(
        [w.weight for w in weighted_experts_base], dtype=torch.float32
    )
    weights_par = torch.tensor(
        [w.weight for w in weighted_experts_par], dtype=torch.float32
    )

    # Final losses via the same bucketed computation for fairness
    loss_base = fitter_base.compute_buckets_loss(weights_base, buckets).item()
    loss_par = fitter_base.compute_buckets_loss(weights_par, buckets).item()

    # Loss similarity (relative tolerance)
    denom = max(1.0, abs(loss_base))
    rel_diff = abs(loss_base - loss_par) / denom
    assert (
        rel_diff < 0.05
    ), f"Losses differ too much: base={loss_base:.4f} par={loss_par:.4f} (rel {rel_diff:.3%})"

    # Weight similarity
    mae = float(
        np.mean(np.abs(weights_base.detach().numpy() - weights_par.detach().numpy()))
    )
    assert mae < 0.1, f"Weights differ too much (MAE={mae:.4f})"

    # Evaluation similarity (average over multiple runs to reduce randomness)
    wm_base = PoEWorldModel(
        observable_extractor=extractor, weighted_experts=weighted_experts_base
    )
    wm_par = PoEWorldModel(
        observable_extractor=extractor, weighted_experts=weighted_experts_par
    )

    seeds = [101, 202, 303, 404, 505]
    accs_base, accs_par = [], []
    ed_raw_base_list, ed_raw_par_list = [], []
    ed_norm_base_list, ed_norm_par_list = [], []

    for s in seeds:
        evaluation_factory = CrafterEvaluationFactory(
            env_config=env_config, policy_seed=s
        )
        evaluation_context = evaluation_factory.create_context(
            config=EvaluationConfig(num_distractors=5), num_transitions_per_scenario=20
        )
        evaluator = Evaluator(evaluation_context)

        perf_base = evaluator.evaluate(wm_base)
        perf_par = evaluator.evaluate(wm_par)

        accs_base.append(perf_base.discriminative_accuracy)
        accs_par.append(perf_par.discriminative_accuracy)
        ed_raw_base_list.append(perf_base.edit_distance.raw)
        ed_raw_par_list.append(perf_par.edit_distance.raw)
        ed_norm_base_list.append(perf_base.edit_distance.normalized)
        ed_norm_par_list.append(perf_par.edit_distance.normalized)

    acc_base = float(np.mean(accs_base))
    acc_par = float(np.mean(accs_par))
    acc_diff = abs(acc_base - acc_par)
    ed_raw_base = float(np.mean(ed_raw_base_list))
    ed_raw_par = float(np.mean(ed_raw_par_list))
    ed_raw_diff = abs(ed_raw_base - ed_raw_par)
    ed_norm_base = float(np.mean(ed_norm_base_list))
    ed_norm_par = float(np.mean(ed_norm_par_list))
    ed_norm_diff = abs(ed_norm_base - ed_norm_par)

    # Debug table printout (rich only)
    from rich.console import Console
    from rich.table import Table

    console = Console()

    meta = Table(title="Weight Fitter Parallel vs Baseline - Summary (Averages)")
    meta.add_column("Metric", justify="left")
    meta.add_column("Baseline", justify="right")
    meta.add_column("Parallel", justify="right")
    meta.add_column("Abs Diff", justify="right")
    meta.add_column("Rel Diff", justify="right")

    def rel(a: float, b: float) -> float:
        denom = max(1.0, abs(a))
        return abs(a - b) / denom

    meta.add_row(
        "Runtime (s)",
        f"{dur_base:.3f}",
        f"{dur_par:.3f}",
        f"{abs(dur_base - dur_par):.3f}",
        f"{rel(dur_base, dur_par):.2%}",
    )
    meta.add_row(
        "Final Loss",
        f"{loss_base:.4f}",
        f"{loss_par:.4f}",
        f"{abs(loss_base - loss_par):.4f}",
        f"{rel(loss_base, loss_par):.2%}",
    )
    meta.add_row(
        "Disc. Accuracy",
        f"{acc_base:.3f}",
        f"{acc_par:.3f}",
        f"{acc_diff:.3f}",
        f"{rel(acc_base, acc_par):.2%}",
    )
    meta.add_row(
        "EditDist Raw",
        f"{ed_raw_base:.3f}",
        f"{ed_raw_par:.3f}",
        f"{ed_raw_diff:.3f}",
        f"{rel(ed_raw_base, ed_raw_par):.2%}",
    )
    meta.add_row(
        "EditDist Norm",
        f"{ed_norm_base:.3f}",
        f"{ed_norm_par:.3f}",
        f"{ed_norm_diff:.3f}",
        f"{rel(ed_norm_base, ed_norm_par):.2%}",
    )

    console.print(meta)

    wt = Table(title="Expert Weights Comparison")
    wt.add_column("Expert", justify="left")
    wt.add_column("Baseline", justify="right")
    wt.add_column("Parallel", justify="right")
    wt.add_column("Abs Diff", justify="right")

    for w_b, w_p in zip(weighted_experts_base, weighted_experts_par):
        name = w_b.expert_function.__name__
        wt.add_row(
            name,
            f"{w_b.weight:.4f}",
            f"{w_p.weight:.4f}",
            f"{abs(w_b.weight - w_p.weight):.4f}",
        )

    console.print(wt)

    assert (
        acc_diff
        <= 0.05  # Discriminative accuracy is quite random, because distractors are random
    ), f"Discriminative accuracy differs too much: base={acc_base:.3f} par={acc_par:.3f}"
    assert (
        ed_norm_diff <= 0.05
    ), f"Normalized edit distance differs too much: base={ed_norm_base:.3f} par={ed_norm_par:.3f}"
    # Raw edit distance is scale-dependent; allow a looser bound
    assert (
        ed_raw_diff <= 5.0
    ), f"Raw edit distance differs too much: base={ed_raw_base:.3f} par={ed_raw_par:.3f}"
