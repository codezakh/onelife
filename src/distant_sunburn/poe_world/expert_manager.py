"""
Expert Manager implementation for PoE-World.

This module implements the ExpertManagerProtocol by wrapping the existing
MaxLikelihoodWeightFitter and PoEWorldModel components. It provides the
interface needed by ObjectModelOrchestrator to manage experts and their weights.
"""

import copy
import os
from typing import Generic, List, TypeVar

import torch
from loguru import logger

from .core import SymbolicTransition, WeightedExpert
from .object_model_learner import ExpertManagerProtocol
from .weight_fitter import MaxLikelihoodWeightFitter
from .world_model import PoEWorldModel

SymbolicStateT = TypeVar("SymbolicStateT")
ActionT = TypeVar("ActionT")


class ExpertManager(Generic[SymbolicStateT, ActionT]):
    """
    Expert Manager that implements ExpertManagerProtocol by coordinating
    MaxLikelihoodWeightFitter and PoEWorldModel components.

    This class wraps the existing weight fitting and world model components
    to provide the interface required by ObjectModelOrchestrator. It handles:
    - Expert addition and management
    - Weight fitting with support for fast mode
    - Expert pruning based on weight thresholds
    - Checkpointing using safetensors
    - Log probability evaluation

    The implementation follows the design outlined in the issue document,
    using immutable updates and tracking expert fitting state.
    """

    def __init__(
        self,
        observable_extractor,
        weight_fitter: MaxLikelihoodWeightFitter[SymbolicStateT],
        weight_threshold: float = 0.01,
    ):
        """
        Initialize the expert manager.

        Args:
            observable_extractor: Extractor for observable attributes from states
            weight_fitter: Weight fitting component for learning expert weights
            weight_threshold: Threshold for pruning experts (default: 0.01)
        """
        self.observable_extractor = observable_extractor
        self.weight_fitter = weight_fitter
        self.weight_threshold = weight_threshold

        # Initialize world model with empty expert list
        self.world_model = PoEWorldModel(observable_extractor, [])

        # Track which experts have been fitted
        self._fitted_experts = set()

        logger.info(
            f"Initialized ExpertManager with weight_threshold={weight_threshold}"
        )

    def add_experts(self, experts: List[WeightedExpert]) -> None:
        """
        Add new experts to this manager.

        Args:
            experts: List of weighted experts to add
        """
        # Create new world model with additional experts
        current_experts = self.world_model.experts
        new_experts = current_experts + experts

        self.world_model = PoEWorldModel(self.observable_extractor, new_experts)

        # Mark new experts as unfitted
        for expert in experts:
            # Use expert function as identifier for tracking
            expert_id = id(expert.expert_function)
            if expert_id not in self._fitted_experts:
                logger.debug(f"Added unfitted expert: {expert_id}")

        logger.info(
            f"Added {len(experts)} experts, total: {len(self.world_model.experts)}"
        )

    def fit_weights(
        self,
        transitions: List[SymbolicTransition[SymbolicStateT]],
        fast_mode: bool = False,
    ) -> None:
        """
        Fit expert weights using the given transitions.

        Args:
            transitions: Training data as symbolic transitions
            fast_mode: If True, only fit weights for newly added experts
        """
        if not transitions:
            logger.warning("No transitions provided for weight fitting")
            return

        if fast_mode:
            # Fast mode: Only fit weights for newly added experts
            # This approach avoids complex masking by passing only new experts to the fitter

            # Identify new experts (those added since last fit)
            new_experts = []
            for expert in self.world_model.experts:
                expert_id = id(expert.expert_function)
                if expert_id not in self._fitted_experts:
                    new_experts.append(expert)

            if new_experts:
                logger.info(
                    f"Fast mode: Fitting weights for {len(new_experts)} new experts"
                )

                # Fit only new experts using existing weight fitter
                new_expert_functions = [
                    expert.expert_function for expert in new_experts
                ]
                new_weighted_experts = self.weight_fitter.fit(
                    new_expert_functions, transitions
                )

                # Update weights for new experts while preserving existing weights
                self._update_weights_for_new_experts(new_weighted_experts)

                # Mark new experts as fitted
                for expert in new_experts:
                    expert_id = id(expert.expert_function)
                    self._fitted_experts.add(expert_id)
                    logger.debug(f"Marked expert {expert_id} as fitted")
            else:
                logger.debug("Fast mode: No new experts to fit")
        else:
            # Full mode: Fit all experts (current behavior)
            logger.info(
                f"Full mode: Fitting weights for all {len(self.world_model.experts)} experts"
            )

            all_expert_functions = [
                expert.expert_function for expert in self.world_model.experts
            ]
            all_weighted_experts = self.weight_fitter.fit(
                all_expert_functions, transitions
            )

            # Replace world model with new weighted experts
            self.world_model = PoEWorldModel(
                self.observable_extractor, all_weighted_experts
            )

            # Mark all experts as fitted
            for expert in self.world_model.experts:
                expert_id = id(expert.expert_function)
                self._fitted_experts.add(expert_id)
                logger.debug(f"Marked expert {expert_id} as fitted")

    def prune_experts(self) -> None:
        """
        Remove experts with weights below the configured threshold.

        This method removes experts that have learned weights below the
        weight_threshold, helping to keep the expert collection focused
        on the most useful predictors.
        """
        if not self.world_model.experts:
            return

        remaining_experts = [
            expert
            for expert in self.world_model.experts
            if expert.weight >= self.weight_threshold
        ]

        pruned_count = len(self.world_model.experts) - len(remaining_experts)

        if pruned_count > 0:
            self.world_model = PoEWorldModel(
                self.observable_extractor, remaining_experts
            )

            # Update fitted tracking to only include remaining experts
            self._fitted_experts = {
                id(expert.expert_function) for expert in remaining_experts
            }

            logger.info(
                f"Pruned {pruned_count} experts below threshold {self.weight_threshold}"
            )
        else:
            logger.debug("No experts pruned - all above threshold")

    def evaluate_log_probability(
        self, state: SymbolicStateT, action: ActionT, next_state: SymbolicStateT
    ) -> float:
        """
        Evaluate log probability of a transition under this manager's experts.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state

        Returns:
            Log probability of the transition
        """
        # NOTE: No error handling implemented - errors from underlying components
        # will propagate up to the caller
        return self.world_model.evaluate_log_probability(state, action, next_state)

    def get_experts(self) -> List[WeightedExpert]:
        """
        Get all experts managed by this manager.

        Returns:
            List of weighted experts
        """
        return self.world_model.experts

    def save(self, checkpoint_path: str) -> None:
        """
        Save manager state to checkpoint using safetensors.

        Args:
            checkpoint_path: Path to save the checkpoint
        """
        # NOTE: No error handling implemented - errors from file operations
        # or serialization will propagate up to the caller

        # Ensure directory exists
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # Prepare checkpoint data
        checkpoint_data = {
            "weight_threshold": torch.tensor([self.weight_threshold]),
            "expert_weights": torch.tensor(
                [expert.weight for expert in self.world_model.experts]
            ),
            "fitted_expert_ids": torch.tensor(list(self._fitted_experts)),
        }

        # Save using safetensors
        from safetensors.torch import save_file

        save_file(checkpoint_data, checkpoint_path)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load(self, checkpoint_path: str) -> bool:
        """
        Load manager state from checkpoint.

        Args:
            checkpoint_path: Path to load the checkpoint from

        Returns:
            True if load successful, False otherwise
        """
        # NOTE: No error handling implemented - errors from file operations
        # or deserialization will propagate up to the caller

        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint file not found: {checkpoint_path}")
            return False

        try:
            # Load using safetensors
            from safetensors import safe_open

            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                # Load basic data
                weight_threshold = f.get_tensor("weight_threshold").item()
                expert_weights = f.get_tensor("expert_weights").numpy()
                fitted_expert_ids = f.get_tensor("fitted_expert_ids").numpy()

            # Update manager state
            self.weight_threshold = weight_threshold
            self._fitted_experts = set(fitted_expert_ids.tolist())

            # Update expert weights in world model
            if len(expert_weights) == len(self.world_model.experts):
                updated_experts = []
                for i, expert in enumerate(self.world_model.experts):
                    updated_expert = WeightedExpert(
                        expert_function=expert.expert_function,
                        weight=float(expert_weights[i]),
                    )
                    updated_experts.append(updated_expert)

                self.world_model = PoEWorldModel(
                    self.observable_extractor, updated_experts
                )

                logger.info(f"Loaded checkpoint from {checkpoint_path}")
                return True
            else:
                logger.error(
                    f"Checkpoint expert count mismatch: expected {len(self.world_model.experts)}, got {len(expert_weights)}"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return False

    def _update_weights_for_new_experts(
        self, new_weighted_experts: List[WeightedExpert]
    ) -> None:
        """
        Update weights for new experts while preserving existing expert weights.

        This helper method is used during fast mode to merge new expert weights
        with existing ones without refitting all experts.

        Args:
            new_weighted_experts: Newly fitted weighted experts
        """
        # Create mapping from expert function to new weight
        new_weights = {id(we.expert_function): we.weight for we in new_weighted_experts}

        # Update existing experts with new weights where available
        updated_experts = []
        for expert in self.world_model.experts:
            expert_id = id(expert.expert_function)
            if expert_id in new_weights:
                # Use new weight
                updated_expert = WeightedExpert(
                    expert_function=expert.expert_function,
                    weight=new_weights[expert_id],
                )
            else:
                # Keep existing weight
                updated_expert = expert
            updated_experts.append(updated_expert)

        # Update world model
        self.world_model = PoEWorldModel(self.observable_extractor, updated_experts)

        logger.debug(f"Updated weights for {len(new_weighted_experts)} new experts")


# Type annotation to indicate this implements the protocol
# Note: We can't use @implements decorator here due to generic type issues
ExpertManager.__implements__ = ExpertManagerProtocol
