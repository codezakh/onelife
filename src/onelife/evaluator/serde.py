"""Pickle-free JSON serialization for evaluator ``EvaluationResults``.

``EvaluationResults`` and its nested ``EditDistance`` / ``EvaluationMetrics`` are
plain frozen dataclasses whose every leaf is numeric, so a pydantic
``TypeAdapter`` round-trips them to/from JSON-compatible structures with no
changes to the dataclasses themselves.
"""

from typing import Any, cast

import numpy as np
from pydantic import TypeAdapter

from onelife.evaluator.core import EvaluationResults

_RESULTS_ADAPTER: TypeAdapter[EvaluationResults] = TypeAdapter(EvaluationResults)


def _numpy_fallback(value: Any) -> Any:
    """Coerce numpy scalars the evaluator produces (e.g. ``np.int64`` counts) to
    plain Python numbers; ``np.int64`` is not an ``int`` subclass so pydantic's
    JSON serializer rejects it otherwise."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize value of type {type(value)!r}")


def evaluation_results_to_jsonable(results: EvaluationResults) -> dict[str, Any]:
    """Convert ``EvaluationResults`` to a JSON-compatible dict."""
    return cast(
        dict[str, Any],
        _RESULTS_ADAPTER.dump_python(results, mode="json", fallback=_numpy_fallback),
    )


def evaluation_results_from_jsonable(data: dict[str, Any]) -> EvaluationResults:
    """Rebuild ``EvaluationResults`` from a dict produced by
    :func:`evaluation_results_to_jsonable`."""
    return _RESULTS_ADAPTER.validate_python(data)
