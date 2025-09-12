"""
PoEWorldLearner: orchestrates per-object-type ObjectModelOrchestrators and
composes a global PoEWorldModel from their experts.

This follows the PoE-World architecture in spirit while operating on our
generic core protocols (WorldModelProtocol, ObservableExtractorProtocol,
SymbolicTransition).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, Iterable, List, Optional, TypeVar

from loguru import logger

from .core import (
    ObservableExtractorProtocol,
    SymbolicTransition,
    WeightedExpert,
    WorldModelProtocol,
)
from .world_model import PoEWorldModel
from .object_model_learner import (
    ObjectModelOrchestrator,
    ObjectTypeModel,
)
from typing import Sequence


SymbolicStateT = TypeVar("SymbolicStateT")
ActionT = TypeVar("ActionT")


@dataclass
class PoEWorldLearnerConfig:
    """Configuration for PoEWorldLearner.

    Additional knobs (e.g., per-type update cadence) can be added later.
    """

    fast_update_default: bool = True


class PoEWorldLearner(Generic[SymbolicStateT, ActionT]):
    """
    Multi-object-type learner that composes a global world model from
    per-type ObjectModelOrchestrators.

    Responsibilities:
    - Route transitions to orchestrators via add_datapoint
    - Trigger learning cycles (full and fast) on orchestrators
    - Compose a PoEWorldModel from all experts
    - Expose a WorldModelProtocol surface (read-only inference)
    - Snapshot/load orchestrators (delegated)
    """

    def __init__(
        self,
        object_type_to_orchestrator: Dict[
            str, ObjectModelOrchestrator[SymbolicStateT, ActionT]
        ],
        observable_extractor: ObservableExtractorProtocol[SymbolicStateT],
        config: Optional[PoEWorldLearnerConfig] = None,
    ) -> None:
        self._object_type_to_orchestrator = object_type_to_orchestrator
        self._observable_extractor = observable_extractor
        self._config = config or PoEWorldLearnerConfig()

        self._current_model: Optional[WorldModelProtocol[SymbolicStateT]] = None

        logger.info(
            f"Initialized PoEWorldLearner with {len(self._object_type_to_orchestrator)} object types"
        )

    # ----------------------------- Public API ----------------------------- #

    def synthesize_world_model(
        self, transitions: Sequence[SymbolicTransition[SymbolicStateT]]
    ) -> WorldModelProtocol[SymbolicStateT]:
        """Run full learning across all orchestrators and compose a model.

        This mirrors the offline initial synthesis pass in PoE-World.
        """
        logger.info(
            f"Synthesis: routing {len(transitions)} transitions to {len(self._object_type_to_orchestrator)} orchestrators"
        )

        # Route transitions and run full inference
        for obj_type, orchestrator in self._object_type_to_orchestrator.items():
            for t in transitions:
                orchestrator.add_datapoint(t)
            logger.info(f"Running full inference for object type '{obj_type}'")
            orchestrator.infer_moe()

        # Compose model from orchestrators' experts
        self._current_model = self._compose_world_model()
        logger.info("Synthesis completed and composed model built")
        return self._current_model

    def update_world_model(
        self,
        transitions: List[SymbolicTransition[SymbolicStateT]],
        fast: Optional[bool] = None,
    ) -> WorldModelProtocol[SymbolicStateT]:
        """Run an online update and re-compose the global model.

        If fast is True, runs fast updates (fit only new experts). Otherwise,
        runs full updates.
        """
        fast_mode = self._config.fast_update_default if fast is None else fast
        logger.info(
            f"Update: routing {len(transitions)} transitions (fast={fast_mode})"
        )

        for obj_type, orchestrator in self._object_type_to_orchestrator.items():
            for t in transitions:
                orchestrator.add_datapoint(t)
            if fast_mode:
                logger.info(f"Running fast inference for object type '{obj_type}'")
                orchestrator.fast_infer_moe()
            else:
                logger.info(f"Running full inference for object type '{obj_type}'")
                orchestrator.infer_moe()

        self._current_model = self._compose_world_model()
        logger.info("Update completed and composed model rebuilt")
        return self._current_model

    def save_snapshot(self) -> None:
        """Save orchestrator snapshots.

        Note: This delegates to the orchestrators' checkpoint mechanism.
        """
        for obj_type, orchestrator in self._object_type_to_orchestrator.items():
            try:
                orchestrator._save_checkpoint(None)  # See note in design doc
                logger.info(f"Saved orchestrator snapshot for '{obj_type}'")
            except Exception as e:
                logger.warning(f"Failed to save snapshot for '{obj_type}': {e}")

    def load_snapshot(self) -> None:
        """Load orchestrator snapshots if present."""
        for obj_type, orchestrator in self._object_type_to_orchestrator.items():
            try:
                if orchestrator._load_checkpoint(None):
                    logger.info(f"Loaded orchestrator snapshot for '{obj_type}'")
            except Exception as e:
                logger.warning(f"Failed to load snapshot for '{obj_type}': {e}")

        # Rebuild composed model after loading
        self._current_model = self._compose_world_model()

    def get_model(self) -> WorldModelProtocol[SymbolicStateT]:
        """Return the current composed model.

        Raises if called before synthesis/update has been run.
        """
        if self._current_model is None:
            raise RuntimeError(
                "World model not yet built. Run synthesize_world_model or update_world_model first."
            )
        return self._current_model

    # ---------------------------- Internal API --------------------------- #

    def _compose_world_model(self) -> PoEWorldModel[SymbolicStateT, ActionT]:
        """Aggregate experts from all orchestrators and build a PoEWorldModel."""
        all_experts: List[WeightedExpert] = []
        for obj_type, orchestrator in self._object_type_to_orchestrator.items():
            obj_model: ObjectTypeModel[SymbolicStateT, ActionT] = (
                orchestrator.get_model()
            )
            # Concatenate non-creation and creation experts
            experts = obj_model.non_creation_experts + obj_model.creation_experts
            logger.debug(f"Collecting {len(experts)} experts from '{obj_type}'")
            all_experts.extend(experts)

        logger.info(f"Composed model with {len(all_experts)} total experts")
        return PoEWorldModel(self._observable_extractor, all_experts)
