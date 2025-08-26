"""
PoE World Model implementation for the 1D benchmark environment.

This module implements the core world model that uses weighted experts to make
probabilistic predictions about state transitions.
"""

import copy
import numpy as np
import torch
from typing import Dict
from loguru import logger

from .core import (
    RandomValues,
    WeightedExpert,
    SymbolicTransition,
    WorldModelProtocol,
)
from ..simple_1d_env.environment import GameState, Action
from .weight_fitter import (
    expand_to_full_domain,
    combine_expert_predictions,
    ObservableExtractor,
)
from ..typing_utils import implements
from typing import TypeVar, Generic
from typing import Protocol, Any


SymbolicStateT = TypeVar("SymbolicStateT")
ActionT = TypeVar("ActionT")


class ObservableExtractorProtocol(Protocol[SymbolicStateT]):
    def extract_attribute_predictions(
        self, state: SymbolicStateT
    ) -> Dict[str, Any]: ...

    def get_observed_values(self, state: SymbolicStateT) -> Dict[str, Any]: ...

    def apply_expert_predictions(
        self,
        new_state: SymbolicStateT,
        expert_predictions: Dict[str, Any],
        weights: torch.Tensor,
    ) -> SymbolicStateT: ...


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

        # Define domain for the 1D environment
        # self.position_domain = np.arange(0, 12)  # [0, 1, 2, ..., 11]
        # self.bool_domain = np.array([0, 1])  # [False, True]

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

        # # Sample player position
        # if "player_position" in expert_predictions:
        #     player_preds = expert_predictions["player_position"]
        #     combined_dist = combine_expert_predictions(player_preds, weights)
        #     new_state.player.position = combined_dist.sample()

        # # Sample light states
        # for i, light in enumerate(new_state.lights):
        #     attr_name = f"light_{i}_is_on"
        #     if attr_name in expert_predictions:
        #         light_preds = expert_predictions[attr_name]
        #         combined_dist = combine_expert_predictions(light_preds, weights)
        #         new_state.lights[i].is_on = bool(combined_dist.sample())

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
                combined_dist = combine_expert_predictions(attr_predictions, weights)
                log_prob = combined_dist.evaluate_log_probability(observed_value)
                total_log_prob += log_prob

        return total_log_prob

    def _get_expert_predictions(
        self, state: SymbolicStateT, action: ActionT
    ) -> Dict[str, list[RandomValues]]:
        """
        Get predictions from all experts for the given state and action.

        Returns:
            Dictionary mapping attribute names to lists of expert predictions
        """
        all_predictions = {}

        for expert in self._experts:
            # Deep copy state and run expert
            state_copy = copy.deepcopy(state)
            expert.expert_function(state_copy, action)

            # Extract predictions for each attribute
            predictions = self.observable_extractor.extract_attribute_predictions(
                state_copy
            )

            # Group by attribute name
            for attr_name, prediction in predictions.items():
                if attr_name not in all_predictions:
                    all_predictions[attr_name] = []
                all_predictions[attr_name].append(prediction)

        return all_predictions

    # def _extract_attribute_predictions(
    #     self, state: GameState
    # ) -> Dict[str, RandomValues]:
    #     """
    #     Extract RandomValues predictions from a state after expert execution.

    #     Returns:
    #         Dictionary mapping attribute names to their predictions
    #     """
    #     predictions = {}

    #     # Extract player position
    #     if isinstance(state.player.position, RandomValues):
    #         predictions["player_position"] = expand_to_full_domain(
    #             state.player.position, self.position_domain
    #         )
    #     else:
    #         # Expert didn't modify this attribute - create uniform distribution
    #         predictions["player_position"] = RandomValues(
    #             values=self.position_domain,
    #             logscores=np.zeros(len(self.position_domain), dtype=np.float32),
    #         )

    #     # Extract light states
    #     for i, light in enumerate(state.lights):
    #         attr_name = f"light_{i}_is_on"
    #         if isinstance(light.is_on, RandomValues):
    #             predictions[attr_name] = expand_to_full_domain(
    #                 light.is_on, self.bool_domain
    #             )
    #         else:
    #             # Expert didn't modify this attribute - create uniform distribution
    #             predictions[attr_name] = RandomValues(
    #                 values=self.bool_domain,
    #                 logscores=np.zeros(len(self.bool_domain), dtype=np.float32),
    #             )

    #     return predictions

    # def _get_observed_values(self, state: GameState) -> Dict[str, int]:
    #     """Extract ground truth observed values from a state."""
    #     observed = {}

    #     # Player position
    #     observed["player_position"] = state.player.position

    #     # Light states
    #     for i, light in enumerate(state.lights):
    #         observed[f"light_{i}_is_on"] = int(light.is_on)

    #     return observed


implements(WorldModelProtocol)(PoEWorldModel)
