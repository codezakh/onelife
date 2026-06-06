"""Pickle-free serialization for Crafter ``LawMixture`` world models.

Learned-content-wise a fitted ``LawMixture`` is just one weight per law: each
law's behavior lives in its source code, which ``LawFunctionWrapper`` already
carries. This module dumps ``(source_code, weight)`` per law plus the observable
extractor's full config to a single JSON file, and rebuilds the model by
re-executing each law's source in the Crafter class namespace.

It lives under ``our_method/crafter`` because reconstruction is Crafter-bound:
the exec namespace, the action remapper, and the observable extractor are all
Crafter-specific. The dump loop only needs ``WorldModelProtocol.laws``, but the
extractor config lives on the concrete ``LawMixture``, so ``save`` takes that type.

The extractor config is serialized field-for-field (every dataclass field on
``ObservableExtractorConfig``, numpy arrays included) rather than a hand-picked
subset, so adding a config field never silently drops state on round-trip.
"""

import ast
import dataclasses
from pathlib import Path
from typing import Any, cast

import numpy as np
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
from pydantic import BaseModel

from onelife.local_code_execution import ExecWithLimitedNamespace
from onelife.our_method.action_remapping import remap_slug_actions_to_balrog_actions
from onelife.our_method.core import LawFunctionWrapper, LawProtocol, WeightedLaw
from onelife.our_method.crafter.observable_extractor import (
    ObservableExtractor,
    ObservableExtractorConfig,
)
from onelife.our_method.world_modeling import LawMixture
from onelife.poe_world.core import DiscreteDistribution

SUPPORTED_FORMAT_VERSION = 1


class SerializedLaw(BaseModel):
    name: str
    weight: float
    is_fitted: bool
    source_code: str


class SerializedLawMixture(BaseModel):
    format_version: int = SUPPORTED_FORMAT_VERSION
    # A dict, not a fixed-field model: enumerating the extractor fields here would
    # reintroduce the silent-drop footgun. _dump/_load_extractor own the schema.
    observable_extractor: dict[str, Any]
    laws: list[SerializedLaw]


def _dump_extractor(extractor: ObservableExtractor) -> dict[str, Any]:
    """Read every ObservableExtractorConfig field off the extractor instance.

    The extractor mirrors each config field onto a same-named attribute, so the
    config's field list is the authoritative schema. numpy arrays become lists.
    """
    out: dict[str, Any] = {}
    for f in dataclasses.fields(ObservableExtractorConfig):
        value = getattr(extractor, f.name)
        out[f.name] = value.tolist() if isinstance(value, np.ndarray) else value
    return out


def _load_extractor(data: dict[str, Any]) -> ObservableExtractor:
    """Rebuild an ObservableExtractor, falling back to current defaults.

    A field absent from ``data`` (an older file, or one written before the field
    existed) takes its current default, mirroring the unpickling shim's behavior.
    """
    defaults = ObservableExtractorConfig()
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(defaults):
        default_value = getattr(defaults, f.name)
        if f.name not in data:
            kwargs[f.name] = default_value
        elif isinstance(default_value, np.ndarray):
            kwargs[f.name] = np.array(data[f.name])
        else:
            kwargs[f.name] = data[f.name]
    return ObservableExtractor(ObservableExtractorConfig(**kwargs))


def _class_name(source_code: str) -> str:
    for node in ast.walk(ast.parse(source_code)):
        if isinstance(node, ast.ClassDef):
            return node.name
    raise ValueError("No class definition found in law source code")


def _materialize_law(source_code: str) -> LawProtocol[WorldState]:
    # Mirrors runbook step 3's materialize_law namespace so reconstructed laws
    # behave identically to freshly-synthesized ones.
    namespace: dict[str, Any] = {
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
        "DiscreteDistribution": DiscreteDistribution,
        "np": np,
        "Any": Any,
        "List": list,
        "Dict": dict,
        "Optional": type(None),
        "Union": type(None),
        "Tuple": tuple,
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
    }
    executor = ExecWithLimitedNamespace(
        allowed_names=set(namespace), inherited_scope=namespace
    )
    executor(source_code)
    law_class = executor.namespace[_class_name(source_code)]
    return LawFunctionWrapper(
        law=law_class(),
        source_code=source_code,
        action_remapper=remap_slug_actions_to_balrog_actions,
    )


def save_law_mixture(model: LawMixture[WorldState, Any], path: str | Path) -> None:
    """Serialize a fitted LawMixture to a single JSON file (no pickle)."""
    serialized = SerializedLawMixture(
        # serde is Crafter-bound, so the extractor is always the concrete one.
        observable_extractor=_dump_extractor(
            cast(ObservableExtractor, model.observable_extractor)
        ),
        laws=[
            SerializedLaw(
                name=weighted_law.law.__name__,
                weight=weighted_law.weight,
                is_fitted=weighted_law.is_fitted,
                source_code=weighted_law.law.__source_code__,
            )
            for weighted_law in model.laws
        ],
    )
    Path(path).write_text(serialized.model_dump_json(indent=2))


def load_law_mixture(path: str | Path) -> LawMixture[WorldState, Any]:
    """Rebuild a LawMixture from a file written by ``save_law_mixture``."""
    serialized = SerializedLawMixture.model_validate_json(Path(path).read_text())
    if serialized.format_version != SUPPORTED_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported serialized LawMixture format_version "
            f"{serialized.format_version}; this code supports "
            f"{SUPPORTED_FORMAT_VERSION}."
        )
    weighted_laws = [
        WeightedLaw(
            law=_materialize_law(law.source_code),
            weight=law.weight,
            is_fitted=law.is_fitted,
        )
        for law in serialized.laws
    ]
    return LawMixture(
        observable_extractor=_load_extractor(serialized.observable_extractor),
        weighted_laws=weighted_laws,
    )
