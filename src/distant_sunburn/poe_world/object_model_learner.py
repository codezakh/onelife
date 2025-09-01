"""
Generic Object Model Orchestrator for PoE-World.

This module implements the core learning orchestration from the external
poe-world implementation, providing a generic framework for object-specific
world model learning.

ARCHITECTURE OVERVIEW:
This module corresponds to the ObjModelLearner class in external poe-world.
The key insight is that the orchestrator manages the learning process for
a specific object type by:

1. Owning two expert managers (non-creation and creation experts)
2. Accumulating transitions over time
3. Running the learning loop that identifies surprising transitions
4. Synthesizing new experts and adding them to the appropriate manager
5. Fitting weights to all experts using accumulated data
6. Pruning useless experts
7. Returning a composed ObjectTypeModel

The structure mirrors external poe-world:
- ObjectModelOrchestrator (this class) = ObjModelLearner
- ObjectTypeModel = ObjTypeModel (data structure containing learned experts)
- ExpertManagerProtocol = MoEObjModel (manages experts and their weights)
"""

import asyncio
from typing import List, Optional, Protocol, TypeVar, Generic, Dict, Any
from dataclasses import dataclass
from loguru import logger

from .core import SymbolicTransition, WeightedExpert

SymbolicStateT = TypeVar("SymbolicStateT")
ActionT = TypeVar("ActionT")


class ExpertManagerProtocol(Protocol[SymbolicStateT, ActionT]):
    """
    Protocol for expert managers that handle a collection of experts for a specific behavior type.

    This corresponds to MoEObjModel in external poe-world. Each expert manager handles
    either non-creation experts (for existing objects) or creation experts (for new objects).
    """

    def add_experts(self, experts: List[WeightedExpert]) -> None:
        """Add new experts to this manager."""
        ...

    def fit_weights(
        self,
        transitions: List[SymbolicTransition[SymbolicStateT]],
        fast_mode: bool = False,
    ) -> None:
        """Fit expert weights using the given transitions."""
        ...

    def prune_experts(self) -> None:
        """Remove experts with low/zero weights."""
        ...

    def evaluate_log_probability(
        self, state: SymbolicStateT, action: ActionT, next_state: SymbolicStateT
    ) -> float:
        """Evaluate log probability of a transition under this manager's experts."""
        ...

    def get_experts(self) -> List[WeightedExpert]:
        """Get all experts managed by this manager."""
        ...

    def save(self, checkpoint_path: str) -> None:
        """Save manager state to checkpoint."""
        ...

    def load(self, checkpoint_path: str) -> bool:
        """Load manager state from checkpoint. Returns True if successful."""
        ...


class ExpertSynthesizerProtocol(Protocol[SymbolicStateT, ActionT]):
    """
    Protocol for expert synthesizers that can generate new experts from transitions.

    This corresponds to the synthesizer modules in external poe-world that generate
    Python code from observed transitions.
    """

    async def synthesize_experts(
        self, transitions: List[SymbolicTransition[SymbolicStateT]], object_type: str
    ) -> List[WeightedExpert]:
        """Synthesize expert programs from state transitions."""
        ...


@dataclass
class ObjectTypeModel(Generic[SymbolicStateT, ActionT]):
    """
    Data structure containing learned experts for a specific object type.

    This corresponds to ObjTypeModel in external poe-world. It contains:
    - object_type: The type of object this model handles (e.g., "player", "ball")
    - non_creation_experts: Experts for predicting how existing objects change
    - creation_experts: Experts for predicting when new objects appear

    Note: Constraints are omitted for simplicity, matching our design decision.
    """

    object_type: str
    non_creation_experts: List[WeightedExpert]
    creation_experts: List[WeightedExpert]


@dataclass
class LearningConfig:
    """Configuration for the object model learning process."""

    batch_size: int = 10
    save_freq: int = 100
    surprise_threshold: float = -2.0
    fast_update_frequency: int = 5
    max_experts_per_object_type: int = 100


class ObjectModelOrchestrator(Generic[SymbolicStateT, ActionT]):
    """
    Generic Object Model Orchestrator that implements the core learning loop from poe-world.

    This class corresponds to ObjModelLearner in external poe-world. It orchestrates
    the learning process for a specific object type by:

    1. Owning two expert managers (non-creation and creation experts)
    2. Owning two expert synthesizers (non-creation and creation experts)
    3. Accumulating transitions over time via add_datapoint()
    4. Running the learning loop in infer_moe() that:
       - Identifies surprising transitions that need explanation
       - Synthesizes new experts using appropriate synthesizers
       - Adds experts to the appropriate manager (non-creation vs creation)
       - Fits weights to all experts using accumulated data
       - Prunes useless experts
    5. Managing checkpointing for incremental learning
    6. Returning a composed ObjectTypeModel containing all learned experts

    The actual expert management, expert synthesis, and weight fitting are injected
    as dependencies, making this class generic and reusable.
    """

    def __init__(
        self,
        object_type: str,
        non_creation_expert_manager: ExpertManagerProtocol[SymbolicStateT, ActionT],
        creation_expert_manager: ExpertManagerProtocol[SymbolicStateT, ActionT],
        non_creation_synthesizer: ExpertSynthesizerProtocol[SymbolicStateT, ActionT],
        creation_synthesizer: ExpertSynthesizerProtocol[SymbolicStateT, ActionT],
        config: LearningConfig,
        checkpoint_dir: str = "checkpoints",
    ):
        """
        Initialize the object model orchestrator.

        Args:
            object_type: Type of object this orchestrator is responsible for
            non_creation_expert_manager: Manager for experts that predict existing object changes
            creation_expert_manager: Manager for experts that predict new object appearance
            non_creation_synthesizer: Synthesizer for experts that handle existing object changes
            creation_synthesizer: Synthesizer for experts that handle new object appearance
            config: Learning configuration parameters
        """
        self.object_type = object_type
        self.non_creation_expert_manager = non_creation_expert_manager
        self.creation_expert_manager = creation_expert_manager
        self.non_creation_synthesizer = non_creation_synthesizer
        self.creation_synthesizer = creation_synthesizer
        self.config = config
        self.checkpoint_dir = checkpoint_dir

        # State management (similar to external implementation)
        self.transitions: List[SymbolicTransition[SymbolicStateT]] = []
        self.processed_obs_count = 0

        logger.info(f"Initialized ObjectModelOrchestrator for {object_type}")

    def add_datapoint(self, transition: SymbolicTransition[SymbolicStateT]) -> None:
        """
        Add a single transition to the accumulated dataset.

        This method accumulates transitions over time, similar to the external
        implementation's add_datapoint method.
        """
        self.transitions.append(transition)
        logger.debug(
            f"Added datapoint for {self.object_type}, total: {len(self.transitions)}"
        )

    def infer_moe(self) -> ObjectTypeModel[SymbolicStateT, ActionT]:
        """
        Main inference method that processes all accumulated observations.

        This implements the core learning loop from the external implementation:
        1. Load from checkpoint if available
        2. Process observations in batches
        3. Identify surprising transitions
        4. Synthesize new experts using appropriate synthesizers
        5. Add experts to appropriate managers (non-creation vs creation)
        6. Fit weights to all experts
        7. Prune useless experts
        8. Save checkpoint

        Returns:
            ObjectTypeModel containing all learned experts for this object type
        """
        logger.info(
            f"Starting inference for {self.object_type} with {len(self.transitions)} transitions"
        )

        # Try to load from final checkpoint
        if self._load_checkpoint(None):
            logger.info(f"Loaded final checkpoint for {self.object_type}")
            return self._get_object_type_model()

        # Try to load from recent checkpoints
        for checkpoint in range(
            len(self.transitions) // self.config.save_freq * self.config.save_freq,
            0,
            -self.config.save_freq,
        ):
            if self._load_checkpoint(checkpoint):
                logger.info(f"Loaded checkpoint {checkpoint} for {self.object_type}")
                break

        # Main learning loop
        logger.info(f"Starting from checkpoint {self.processed_obs_count}")
        while self.processed_obs_count < len(self.transitions):
            # Process in batches
            batch_end = min(
                self.processed_obs_count + self.config.batch_size, len(self.transitions)
            )
            batch_indices = list(range(self.processed_obs_count, batch_end))

            # Find surprising transitions in this batch
            surprising_indices = self._find_surprising_transitions(batch_indices)

            if surprising_indices:
                logger.info(
                    f"Found {len(surprising_indices)} surprising transitions in batch"
                )

                # Synthesize experts for surprising transitions using both synthesizers
                # Each synthesizer can decide independently whether it can handle the transition
                non_creation_experts, creation_experts = asyncio.run(
                    self._synthesize_for_surprising_transitions(surprising_indices)
                )

                # Add experts to appropriate managers if any were synthesized
                if non_creation_experts:
                    logger.info(
                        f"Synthesized {len(non_creation_experts)} non-creation experts"
                    )
                    self.non_creation_expert_manager.add_experts(non_creation_experts)
                if creation_experts:
                    logger.info(f"Synthesized {len(creation_experts)} creation experts")
                    self.creation_expert_manager.add_experts(creation_experts)

                # Fit weights using all accumulated data if we added new experts
                if non_creation_experts or creation_experts:
                    self.non_creation_expert_manager.fit_weights(
                        self.transitions, fast_mode=False
                    )
                    self.creation_expert_manager.fit_weights(
                        self.transitions, fast_mode=False
                    )

                    # Prune useless experts
                    self.non_creation_expert_manager.prune_experts()
                    self.creation_expert_manager.prune_experts()
            else:
                logger.debug("No surprising transitions found in batch")

            # Update checkpoint
            self.processed_obs_count = batch_end

            # Save checkpoint periodically
            if self.processed_obs_count % self.config.save_freq == 0:
                self._save_checkpoint(self.processed_obs_count)

        # Final weight fitting and pruning
        logger.info("Performing final weight fitting and pruning")
        self.non_creation_expert_manager.fit_weights(self.transitions, fast_mode=False)
        self.creation_expert_manager.fit_weights(self.transitions, fast_mode=False)
        self.non_creation_expert_manager.prune_experts()
        self.creation_expert_manager.prune_experts()

        # Save final checkpoint
        self._save_checkpoint(None)

        return self._get_object_type_model()

    def fast_infer_moe(self) -> ObjectTypeModel[SymbolicStateT, ActionT]:
        """
        Fast inference method for quick updates during agent execution.

        This method only fits weights for newly added experts, making it
        much faster than the full infer_moe() method.
        """
        logger.info(f"Starting fast inference for {self.object_type}")

        # Find surprising transitions in new data only
        new_indices = list(range(self.processed_obs_count, len(self.transitions)))
        surprising_indices = self._find_surprising_transitions(new_indices)

        if surprising_indices:
            logger.info(f"Found {len(surprising_indices)} surprising transitions")

            # Synthesize experts for surprising transitions using both synthesizers
            non_creation_experts, creation_experts = asyncio.run(
                self._synthesize_for_surprising_transitions(surprising_indices)
            )

            # Add experts to appropriate managers if any were synthesized
            if non_creation_experts:
                logger.info(
                    f"Synthesized {len(non_creation_experts)} non-creation experts"
                )
                self.non_creation_expert_manager.add_experts(non_creation_experts)
            if creation_experts:
                logger.info(f"Synthesized {len(creation_experts)} creation experts")
                self.creation_expert_manager.add_experts(creation_experts)

            # Fast weight fitting (only new experts) if we added new experts
            if non_creation_experts or creation_experts:
                self.non_creation_expert_manager.fit_weights(
                    self.transitions, fast_mode=True
                )
                self.creation_expert_manager.fit_weights(
                    self.transitions, fast_mode=True
                )

        # Update checkpoint
        self.processed_obs_count = len(self.transitions)

        return self._get_object_type_model()

    def _find_surprising_transitions(self, indices: List[int]) -> List[int]:
        """
        Find indices of transitions that are surprising (low probability under current models).

        This method evaluates each transition using both expert managers and considers
        a transition surprising if neither manager can explain it well (log probability
        below threshold). This allows both creation and non-creation experts to
        contribute to explaining transitions.

        Args:
            indices: List of transition indices to check

        Returns:
            List of indices for surprising transitions
        """
        surprising_indices = []

        for idx in indices:
            if idx >= len(self.transitions):
                continue

            transition = self.transitions[idx]

            try:
                # Evaluate log probability using both expert managers
                # Each manager only considers its own experts (creation vs non-creation)
                non_creation_log_prob = (
                    self.non_creation_expert_manager.evaluate_log_probability(
                        state=transition.prev_metadata,
                        action=transition.action,
                        next_state=transition.next_metadata,
                    )
                )
                creation_log_prob = (
                    self.creation_expert_manager.evaluate_log_probability(
                        state=transition.prev_metadata,
                        action=transition.action,
                        next_state=transition.next_metadata,
                    )
                )

                # Use the maximum log probability (best explanation from either manager)
                # This means if either creation or non-creation experts can explain
                # the transition well, it's not considered surprising
                log_prob = max(non_creation_log_prob, creation_log_prob)

                # If log probability is below threshold, transition is surprising
                if log_prob < self.config.surprise_threshold:
                    surprising_indices.append(idx)
                    logger.debug(
                        f"Surprising transition at index {idx}: "
                        f"log_prob={log_prob:.3f} < {self.config.surprise_threshold}"
                    )

            except Exception as e:
                logger.warning(f"Failed to evaluate transition {idx}: {e}")
                # If we can't evaluate, treat as surprising to be safe
                surprising_indices.append(idx)

        return surprising_indices

    async def _synthesize_for_surprising_transitions(
        self, indices: List[int]
    ) -> tuple[List[WeightedExpert], List[WeightedExpert]]:
        """
        Synthesize experts for the given surprising transition indices.

        This method uses both synthesizers to attempt to explain each surprising
        transition. Each synthesizer can independently decide whether it can
        handle a transition and return appropriate experts. This design allows
        for natural separation of concerns without requiring pre-classification
        of transitions.

        Args:
            indices: List of transition indices to synthesize experts for

        Returns:
            Tuple of (non_creation_experts, creation_experts)
        """
        non_creation_experts = []
        creation_experts = []

        for idx in indices:
            if idx >= len(self.transitions):
                continue

            transition = self.transitions[idx]

            try:
                # Synthesize experts using both synthesizers
                # Each synthesizer can return empty list if it can't handle the transition
                non_creation_batch = (
                    await self.non_creation_synthesizer.synthesize_experts(
                        transitions=[transition], object_type=self.object_type
                    )
                )
                creation_batch = await self.creation_synthesizer.synthesize_experts(
                    transitions=[transition], object_type=self.object_type
                )

                non_creation_experts.extend(non_creation_batch)
                creation_experts.extend(creation_batch)

                logger.debug(
                    f"Synthesized {len(non_creation_batch)} non-creation and "
                    f"{len(creation_batch)} creation experts for transition {idx}"
                )

            except Exception as e:
                logger.error(f"Failed to synthesize experts for transition {idx}: {e}")
                continue

        return non_creation_experts, creation_experts

    def _get_object_type_model(self) -> ObjectTypeModel[SymbolicStateT, ActionT]:
        """
        Get the current object type model containing all learned experts.

        Returns:
            ObjectTypeModel containing non-creation and creation experts
        """
        non_creation_experts = self.non_creation_expert_manager.get_experts()
        creation_experts = self.creation_expert_manager.get_experts()

        return ObjectTypeModel(
            object_type=self.object_type,
            non_creation_experts=non_creation_experts,
            creation_experts=creation_experts,
        )

    def _save_checkpoint(self, checkpoint: Optional[int]) -> None:
        """
        Save current state to checkpoint.

        Args:
            checkpoint: Checkpoint number, or None for final checkpoint
        """
        checkpoint_path = self._get_checkpoint_path(checkpoint)

        try:
            self.non_creation_expert_manager.save(checkpoint_path + "_non_creation")
            self.creation_expert_manager.save(checkpoint_path + "_creation")
            logger.info(f"Saved checkpoint {checkpoint} to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint}: {e}")

    def _load_checkpoint(self, checkpoint: Optional[int]) -> bool:
        """
        Load state from checkpoint.

        Args:
            checkpoint: Checkpoint number, or None for final checkpoint

        Returns:
            True if load successful, False otherwise
        """
        checkpoint_path = self._get_checkpoint_path(checkpoint)

        try:
            non_creation_success = self.non_creation_expert_manager.load(
                checkpoint_path + "_non_creation"
            )
            creation_success = self.creation_expert_manager.load(
                checkpoint_path + "_creation"
            )
            success = non_creation_success and creation_success
            if success:
                logger.info(f"Loaded checkpoint {checkpoint} from {checkpoint_path}")
            return success
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint}: {e}")
            return False

    def _get_checkpoint_path(self, checkpoint: Optional[int]) -> str:
        """
        Get the file path for a checkpoint.

        Args:
            checkpoint: Checkpoint number, or None for final checkpoint

        Returns:
            Checkpoint file path
        """
        checkpoint_str = "final" if checkpoint is None else str(checkpoint)
        return f"{self.checkpoint_dir}/{self.object_type}/{checkpoint_str}.pkl"

    def get_model(self) -> ObjectTypeModel[SymbolicStateT, ActionT]:
        """Get the current object type model."""
        return self._get_object_type_model()
