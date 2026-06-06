"""Tests for pickle-free EvaluationResults serialization (onelife.evaluator.serde).

Round-trip equivalence judged by value (frozen dataclasses compare by field):
  1. a fully-populated EvaluationResults survives object -> jsonable -> object,
  2. the jsonable form is real JSON (survives json.dumps/loads),
  3. numpy scalars (np.float64 fields, np.int64 count) round-trip to plain types.
"""

import json
from typing import cast

import numpy as np

from onelife.evaluator.core import EditDistance, EvaluationMetrics, EvaluationResults
from onelife.evaluator.serde import (
    evaluation_results_from_jsonable,
    evaluation_results_to_jsonable,
)


def _edit_distance(scale: float = 1.0) -> EditDistance:
    return EditDistance(
        raw=1.5 * scale,
        normalized=0.12 * scale,
        total_elements=10.0,
        intersection_over_union=0.87 * scale,
    )


def _metrics(scale: float = 1.0) -> EvaluationMetrics:
    return EvaluationMetrics(
        edit_distance=_edit_distance(scale),
        discriminative_accuracy=0.8 * scale,
        normalized_recall=0.7 * scale,
        reciprocal_rank=0.6 * scale,
        raw_rank=2.0,
        n_distractors=10.0,
    )


def _results(total_transitions: int = 30) -> EvaluationResults:
    return EvaluationResults(
        edit_distance=_edit_distance(),
        edit_distance_std=_edit_distance(0.1),
        discriminative_accuracy=0.82,
        discriminative_accuracy_std=0.05,
        normalized_recall=0.71,
        normalized_recall_std=0.04,
        reciprocal_rank=0.63,
        reciprocal_rank_std=0.03,
        total_transitions_evaluated=total_transitions,
        metrics_by_source={
            "collect_wood": {"mean": _metrics(), "std": _metrics(0.1)},
            "place_table": {"mean": _metrics(0.9), "std": _metrics(0.2)},
        },
        raw_metrics_by_source={
            "collect_wood": [_metrics(), _metrics(0.5)],
            "place_table": [_metrics(0.9)],
        },
    )


def test_roundtrip_preserves_full_results():
    original = _results()
    rebuilt = evaluation_results_from_jsonable(evaluation_results_to_jsonable(original))
    assert rebuilt == original


def test_jsonable_form_is_real_json():
    original = _results()
    blob = json.dumps(evaluation_results_to_jsonable(original))
    rebuilt = evaluation_results_from_jsonable(json.loads(blob))
    assert rebuilt == original


def test_numpy_scalars_roundtrip_to_plain_types():
    # The real evaluator yields numpy scalars; np.int64 in particular is NOT an
    # int subclass, so this is the case most likely to break serialization.
    original = EvaluationResults(
        edit_distance=EditDistance(
            raw=np.float64(1.5),
            normalized=np.float64(0.12),
            total_elements=np.float64(10.0),
            intersection_over_union=np.float64(0.87),
        ),
        edit_distance_std=_edit_distance(0.1),
        discriminative_accuracy=np.float64(0.82),
        discriminative_accuracy_std=0.05,
        normalized_recall=0.71,
        normalized_recall_std=0.04,
        reciprocal_rank=0.63,
        reciprocal_rank_std=0.03,
        total_transitions_evaluated=cast(int, np.int64(30)),
        metrics_by_source={"collect_wood": {"mean": _metrics(), "std": _metrics(0.1)}},
        raw_metrics_by_source={"collect_wood": [_metrics()]},
    )
    rebuilt = evaluation_results_from_jsonable(evaluation_results_to_jsonable(original))

    assert rebuilt.discriminative_accuracy == 0.82
    assert rebuilt.total_transitions_evaluated == 30
    assert isinstance(rebuilt.total_transitions_evaluated, int)
    assert isinstance(rebuilt.edit_distance.raw, float)
