"""
PoE World Model implementation for the 1D benchmark environment.

This module implements the core world model that uses weighted experts to make
probabilistic predictions about state transitions.
"""

import copy
from typing import Dict, Generic, TypeVar

import torch
from loguru import logger

from distant_sunburn.poe_world.core import ObservableExtractorProtocol

from ..typing_utils import implements
from .core import (
    RandomValues,
    WeightedExpert,
    WorldModelProtocol,
    ObservableId,
)
from .weight_fitter import (
    combine_expert_predictions_for_attr,
)

SymbolicStateT = TypeVar("SymbolicStateT")
ActionT = TypeVar("ActionT")


class PoEWorldModel(Generic[SymbolicStateT, ActionT]):
    """
    Product of Experts World Model for the 1D benchmark environment.

    This model uses weighted experts to make probabilistic predictions about
    state transitions. It implements the core PoE combination logic described
    in the PRD and supplementary material.
    """

    def __init__(
        self,
        observable_extractor: ObservableExtractorProtocol[SymbolicStateT],
        weighted_experts: list[WeightedExpert] | None = None,
    ):
        self._experts = weighted_experts or []

        self.observable_extractor = observable_extractor

        logger.debug(f"Initialized PoEWorldModel with {len(self._experts)} experts")

    @property
    def experts(self) -> list[WeightedExpert]:
        """Get the list of weighted experts."""
        return self._experts

    def with_new_experts(
        self, new_experts: list[WeightedExpert]
    ) -> "PoEWorldModel[SymbolicStateT, ActionT]":
        """Create a new world model with the given experts."""
        return PoEWorldModel(
            weighted_experts=new_experts, observable_extractor=self.observable_extractor
        )

    def sample_next_state(
        self, current_state: SymbolicStateT, action: ActionT
    ) -> SymbolicStateT:
        """
        Sample a next state using the weighted experts.

        Args:
            current_state: Current game state
            action: Action being taken

        Returns:
            Sampled next state
        """
        if not self._experts:
            # No experts - return current state unchanged
            return copy.deepcopy(current_state)

        # Get expert predictions
        expert_predictions = self._get_expert_predictions(current_state, action)

        # Extract weights as tensor
        weights = torch.tensor(
            [expert.weight for expert in self._experts], dtype=torch.float32
        )

        # Create new state by sampling from combined distributions
        new_state = copy.deepcopy(current_state)

        new_state = self.observable_extractor.apply_expert_predictions(
            new_state, expert_predictions, weights
        )

        return new_state

    def evaluate_log_probability(
        self, state: SymbolicStateT, action: ActionT, next_state: SymbolicStateT
    ) -> float:
        """
        Evaluate the log-probability of a transition under this model.

        Args:
            transition: The transition to evaluate

        Returns:
            Log-probability of the transition
        """
        if not self._experts:
            # No experts - return uniform probability (log(1) = 0)
            return 0.0

        # Get expert predictions
        expert_predictions = self._get_expert_predictions(state, action)

        # Extract weights as tensor
        weights = torch.tensor(
            [expert.weight for expert in self._experts], dtype=torch.float32
        )

        # Get observed values from next state
        observed_values = self.observable_extractor.get_observed_values(next_state)

        total_log_prob = 0.0

        # Evaluate log-probability for each attribute
        for attr_name, observed_value in observed_values.items():
            if attr_name in expert_predictions:
                attr_predictions = expert_predictions[attr_name]
                combined_dist = combine_expert_predictions_for_attr(
                    attr_predictions, weights
                )
                log_prob = combined_dist.evaluate_log_probability(observed_value)
                total_log_prob += log_prob

        return total_log_prob

    def _get_expert_predictions(
        self, state: SymbolicStateT, action: ActionT
    ) -> Dict[ObservableId, list[RandomValues]]:
        """
        Get predictions from all experts for the given state and action.

        Returns:
            Dictionary mapping attribute names to lists of expert predictions
        """
        predictions_from_all_experts: dict[ObservableId, list[RandomValues]] = {}

        for expert in self._experts:
            # Deep copy state and run expert
            state_copy = copy.deepcopy(state)
            expert.expert_function(state_copy, action)

            # Extract predictions for each attribute
            attr_predictions = self.observable_extractor.extract_attribute_predictions(
                state_copy
            )

            # Group by attribute name
            for attr_name, attr_prediction in attr_predictions.items():
                if attr_name not in predictions_from_all_experts:
                    predictions_from_all_experts[attr_name] = []
                predictions_from_all_experts[attr_name].append(attr_prediction)

        return predictions_from_all_experts


implements(WorldModelProtocol)(PoEWorldModel)
