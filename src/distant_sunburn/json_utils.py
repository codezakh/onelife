import json
from typing import Any, NamedTuple
import jsonpatch
import numpy as np
from pydantic import BaseModel
from typing import TypeVar, cast


def flatten_json_to_pathmap(data: dict | list) -> dict[str, Any]:
    """
    Flattens a nested dictionary or list into a dictionary of JSON-Patch style paths
    using an iterative, stack-based approach.
    """
    flat_paths: dict[str, Any] = {}

    # The stack will hold tuples of (path, object_to_process)
    # Start with the root object and an empty path string.
    stack = [("", data)]

    while stack:
        path, obj = stack.pop()

        if isinstance(obj, dict):
            if not obj:
                # We will record an empty dict as a terminal value
                flat_paths[path] = obj
                continue
            # Add child items to the stack to be processed later.
            # We iterate in reverse to maintain a more natural (or consistent)
            # processing order since a stack is LIFO.
            for key, value in reversed(list(obj.items())):
                stack.append((f"{path}/{key}", value))

        elif isinstance(obj, list):
            if not obj:
                # We will record an empty list as a terminal value
                flat_paths[path] = obj
                continue
            # Add child items to the stack in reverse order.
            for i, value in reversed(list(enumerate(obj))):
                stack.append((f"{path}/{i}", value))

        else:
            # This is a terminal value, so we record its path.
            # The path will be non-empty because we start with "" for the root.
            flat_paths[path] = obj

    return flat_paths


class PatchOperation(NamedTuple):
    """Represents a JSON Patch operation for equality comparison.

    This named tuple captures the essential information needed to compare
    patch operations for intersection over union calculations.
    """

    path: str
    operation: str
    value: Any


def compute_patch_intersection_over_union(
    patch1: jsonpatch.JsonPatch,
    patch2: jsonpatch.JsonPatch,
) -> float:
    """
    Compute the intersection over union (IoU) of two JSON patches.

    This function compares two JsonPatch objects by converting their operations
    into comparable PatchOperation objects and computing the IoU metric.

    Args:
        patch1: First JsonPatch object
        patch2: Second JsonPatch object

    Returns:
        float: IoU value between 0.0 and 1.0, where:
               - 1.0 means the patches are identical
               - 0.0 means the patches have no operations in common
               - Values in between indicate partial overlap

    Example:
        >>> source = {"a": 1, "b": 2}
        >>> patch1 = jsonpatch.make_patch(source, {"a": 3, "b": 2})
        >>> patch2 = jsonpatch.make_patch(source, {"a": 3, "b": 4})
        >>> iou = compute_patch_intersection_over_union(patch1, patch2, source)
        >>> print(f"IoU: {iou:.2f}")  # Should be 0.5 (1 common operation out of 2 total)
    """
    # Convert patches to sets of hashable operation keys for comparison.
    #
    # Why not store raw values? In RFC 6902 JSON Patch, the "value" field for
    # add/replace/test can be any JSON value, including objects and arrays.
    # Python dicts/lists are unhashable, so attempting to put them directly in
    # a set (e.g., within a NamedTuple) raises "TypeError: unhashable type: 'dict'".
    #
    # To support correct equality and hashing, we canonicalize each operation
    # into a tuple of primitives:
    #   (op, path, canonical_json_value_or_None, from_path_or_None)
    # where the value (when present) is serialized with sorted keys and compact
    # separators. This makes two semantically equal JSON values compare equal
    # as strings, and remains hashable.
    #
    # We also include the optional "from" field for copy/move operations to
    # distinguish otherwise identical operations with different sources.

    def _operation_key(operation: dict) -> tuple[str, str, str | None, str | None]:
        op_type = operation["op"]
        path = operation["path"]
        value_repr: str | None
        # Only serialize if the key exists. This preserves distinction between
        # missing value vs explicit null value (the latter becomes 'null').
        if "value" in operation:
            value_repr = json.dumps(
                operation["value"], sort_keys=True, separators=(",", ":")
            )
        else:
            value_repr = None
        from_path = operation.get("from", None)
        return (op_type, path, value_repr, from_path)

    operations1 = {_operation_key(op) for op in patch1.patch}
    operations2 = {_operation_key(op) for op in patch2.patch}

    # Compute intersection and union
    intersection = operations1 & operations2
    union = operations1 | operations2

    # Calculate IoU
    if len(union) == 0:
        return 1.0  # Both patches are empty, so they're identical

    iou = len(intersection) / len(union)
    return iou


def _coerce_numpy_to_builtins(obj: Any) -> Any:
    """
    Recursively convert numpy scalar/array types into Python builtins.

    - np.generic (np.int64, np.float64, np.bool_) -> .item()
    - np.ndarray -> .tolist()
    - Containers (dict, list, tuple, set) -> coerced recursively, preserving type
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _coerce_numpy_to_builtins(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numpy_to_builtins(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_coerce_numpy_to_builtins(v) for v in obj)
    if isinstance(obj, set):
        return {_coerce_numpy_to_builtins(v) for v in obj}
    return obj


TModel = TypeVar("TModel", bound=BaseModel)


def sanitize_model_numpy_types(model: TModel) -> TModel:
    """
    Return a new instance of the same Pydantic model with numpy types coerced
    to Python builtins by dumping in Python mode, coercing, then re-validating.

    This is non-invasive (doesn't mutate the original model in place) and
    ensures the resulting model is fully JSON-serializable under Pydantic v2.
    """
    data = model.model_dump(mode="python")
    data = _coerce_numpy_to_builtins(data)
    return cast(TModel, model.__class__.model_validate(data))
