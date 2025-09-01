"""
Expert Manager implementation for PoE-World.

This module implements the ExpertManagerProtocol by wrapping the existing
MaxLikelihoodWeightFitter and PoEWorldModel components. It provides the
interface needed by ObjectModelOrchestrator to manage experts and their weights.
"""

import os
from typing import Generic, List, TypeVar

import torch
from loguru import logger

from .core import SymbolicTransition, WeightedExpert, ObservableExtractorProtocol
from .weight_fitter import MaxLikelihoodWeightFitter
from .world_model import PoEWorldModel

SymbolicStateT = TypeVar("SymbolicStateT")
ActionT = TypeVar("ActionT")


class ExpertManager(Generic[SymbolicStateT, ActionT]):
    """
    Coordinates a a world model and an optimizer for the world model to fit a
    and manage a set of experts.
    """

    def __init__(
        self,
        observable_extractor: ObservableExtractorProtocol[SymbolicStateT],
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
        self.world_model = PoEWorldModel[SymbolicStateT, ActionT](
            observable_extractor, []
        )

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
            # Find unfitted experts and their indices
            unfitted_experts_with_indices = [
                (i, expert)
                for i, expert in enumerate(self.world_model.experts)
                if not expert.is_fitted
            ]

            if unfitted_experts_with_indices:
                indices, unfitted_experts = zip(*unfitted_experts_with_indices)
                expert_functions = [
                    expert.expert_function for expert in unfitted_experts
                ]

                logger.info(
                    f"Fast mode: Fitting weights for {len(unfitted_experts)} new experts"
                )

                # Fit the unfitted experts
                fitted_experts = self.weight_fitter.fit(expert_functions, transitions)

                # Validate that the returned list has the expected length
                if len(fitted_experts) != len(expert_functions):
                    raise ValueError(
                        f"Weight fitter returned {len(fitted_experts)} experts but expected {len(expert_functions)}. "
                        "The returned list should maintain the same order and length as the input experts list."
                    )

                # Update the experts at the specific indices
                updated_experts = list(self.world_model.experts)  # Create a copy
                for idx, fitted_expert in zip(indices, fitted_experts):
                    updated_experts[idx] = fitted_expert

                self.world_model = PoEWorldModel(
                    self.observable_extractor, updated_experts
                )
            else:
                logger.debug("Fast mode: No new experts to fit")
        else:
            # Full mode: Fit all experts
            logger.info(
                f"Full mode: Fitting weights for all {len(self.world_model.experts)} experts"
            )

            all_expert_functions = [
                expert.expert_function for expert in self.world_model.experts
            ]
            all_fitted_experts = self.weight_fitter.fit(
                all_expert_functions, transitions
            )

            # Validate that the returned list has the expected length
            if len(all_fitted_experts) != len(all_expert_functions):
                raise ValueError(
                    f"Weight fitter returned {len(all_fitted_experts)} experts but expected {len(all_expert_functions)}. "
                    "The returned list should maintain the same order and length as the input experts list."
                )

            # Replace all experts (same order)
            self.world_model = PoEWorldModel(
                self.observable_extractor, all_fitted_experts
            )

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
            "expert_is_fitted": torch.tensor(
                [expert.is_fitted for expert in self.world_model.experts]
            ),
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

                # Load is_fitted status - this field is required
                try:
                    expert_is_fitted = f.get_tensor("expert_is_fitted").numpy()
                except KeyError:
                    raise ValueError(
                        f"Checkpoint {checkpoint_path} is missing required 'expert_is_fitted' field. "
                        "This indicates an incompatible checkpoint format."
                    )

            # Validate tensor lengths match
            if len(expert_weights) != len(expert_is_fitted):
                raise ValueError(
                    f"Checkpoint tensor length mismatch: expert_weights has {len(expert_weights)} "
                    f"elements but expert_is_fitted has {len(expert_is_fitted)} elements"
                )

            # Update manager state
            self.weight_threshold = weight_threshold

            # Update expert weights in world model
            if len(expert_weights) == len(self.world_model.experts):
                updated_experts = []
                for i, expert in enumerate(self.world_model.experts):
                    updated_expert = WeightedExpert(
                        expert_function=expert.expert_function,
                        weight=float(expert_weights[i]),
                        is_fitted=bool(expert_is_fitted[i]),
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
