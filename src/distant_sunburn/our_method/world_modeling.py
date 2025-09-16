import copy
import itertools
from loguru import logger
from collections import defaultdict
from typing import Dict, Generic, TypeVar

import torch


from .core import ObservableExtractorProtocol

from ..typing_utils import implements
from ..poe_world.core import (
    DiscreteDistribution,
    ObservableId,
)
from .optimization import (
    combine_expert_predictions_for_attr,
)
from .core import WeightedLaw, WorldModelProtocol
from typing import TypeAlias


ExpertIndex: TypeAlias = int


# Constants for log probability values
LOG_IMPOSSIBLE_VALUE = -1000.0  # Very low probability for impossible transitions

SymbolicStateT = TypeVar("SymbolicStateT")
ActionT = TypeVar("ActionT")


class LawMixture(Generic[SymbolicStateT, ActionT]):
    def __init__(
        self,
        observable_extractor: ObservableExtractorProtocol[SymbolicStateT],
        weighted_laws: list[WeightedLaw[SymbolicStateT]] | None = None,
    ):
        self._laws = weighted_laws or []
        self.observable_extractor = observable_extractor

    @property
    def laws(self) -> list[WeightedLaw[SymbolicStateT]]:
        """Get the list of weighted laws."""
        return self._laws

    def with_new_laws(
        self, new_laws: list[WeightedLaw[SymbolicStateT]]
    ) -> "LawMixture[SymbolicStateT, ActionT]":
        """Create a new world model with the given laws."""
        return LawMixture(
            weighted_laws=new_laws, observable_extractor=self.observable_extractor
        )

    def sample_next_state(
        self, current_state: SymbolicStateT, action: ActionT
    ) -> SymbolicStateT:
        """
        Sample a next state using the weighted laws.

        Args:
            current_state: Current game state
            action: Action being taken

        Returns:
            Sampled next state
        """
        if not self._laws:
            # No experts - return current state unchanged
            return copy.deepcopy(current_state)

        with logger.contextualize(action=action):

            # Get expert predictions
            expert_predictions = self._get_law_predictions(current_state, action)

            active_law_indices = {
                _ for _ in itertools.chain.from_iterable(expert_predictions.values())
            }

            active_laws = [self._laws[_] for _ in active_law_indices]

            logger.debug(
                f"{len(active_laws)} active laws",
                active_laws=[law.law.__name__ for law in active_laws],
            )

            with logger.contextualize(
                active_laws=[law.law.__name__ for law in active_laws]
            ):

                # Extract weights as tensor
                # weights = torch.tensor(
                #     [
                #         law.weight
                #         for idx, law in enumerate(self._laws)
                #         if idx in active_law_indices
                #     ],
                #     dtype=torch.float32,
                # )

                weights = torch.tensor(
                    [_.weight for _ in self._laws],
                    dtype=torch.float32,
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
            state: Current state
            action: Action taken
            next_state: Next state

        Returns:
            Log-probability of the transition
        """
        if not self._laws:
            return LOG_IMPOSSIBLE_VALUE

        # Get expert predictions
        law_predictions = self._get_law_predictions(state, action)

        # Extract weights as tensor
        weights = torch.tensor(
            [_.weight for _ in self._laws],
            dtype=torch.float32,
        )

        # Get observed values from next state
        observed_values = self.observable_extractor.get_observed_outcomes(next_state)

        total_log_prob = 0.0

        # Evaluate log-probability for each attribute
        for attr_name, observed_value in observed_values.items():
            if attr_name in law_predictions:
                law_id_to_law_preds = law_predictions[attr_name]
                active_indices = list(law_id_to_law_preds.keys())
                active_weights = weights[active_indices]
                raw_preds = list(law_id_to_law_preds.values())
                combined_dist = combine_expert_predictions_for_attr(
                    raw_preds, active_weights
                )
                log_prob = combined_dist.evaluate_log_probability(observed_value)
                total_log_prob += log_prob

        return total_log_prob

    def _get_law_predictions(
        self, state: SymbolicStateT, action: ActionT
    ) -> dict[ObservableId, dict[ExpertIndex, DiscreteDistribution]]:
        """
        Get predictions from all experts for the given state and action.

        Returns:
            Tuple containing:
            - List of indices of active laws (who made a prediction)
            - Dictionary mapping attribute names to lists of law predictions
                made by active laws
        """
        preds_from_active_laws: dict[
            ObservableId, dict[ExpertIndex, DiscreteDistribution]
        ] = defaultdict(dict)

        for law_idx, weighted_law in enumerate(self._laws):
            # Deep copy state and run expert
            state_copy = copy.deepcopy(state)

            if not weighted_law.law.precondition(state_copy, action):
                continue

            weighted_law.law.effect(state_copy, action)

            # Extract predictions for each attribute
            attr_predictions = self.observable_extractor.extract_attribute_predictions(
                state_copy
            )

            # Group by attribute name
            for attr_name, attr_prediction in attr_predictions.items():
                preds_from_active_laws[attr_name][law_idx] = attr_prediction

        return preds_from_active_laws


implements(WorldModelProtocol)(LawMixture)
