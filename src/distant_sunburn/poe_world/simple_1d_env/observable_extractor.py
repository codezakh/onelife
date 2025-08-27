import numpy as np
import torch

from distant_sunburn.poe_world.core import DiscreteDistribution
from distant_sunburn.poe_world.weight_fitter import (
    combine_expert_predictions_for_attr,
    # expand_to_full_domain,
)
from distant_sunburn.simple_1d_env.environment import GameState
from ..core import ObservableId, ObservableExtractorProtocol
from ...typing_utils import implements


class ObservableExtractor:
    def __init__(self):
        self.position_domain = np.arange(0, 12)  # [0, 1, 2, ..., 11]
        self.bool_domain = np.array([0, 1])  # [False, True]

    def extract_attribute_predictions(
        self, state: GameState
    ) -> dict[ObservableId, DiscreteDistribution]:
        """
        Extract RandomValues predictions from a state after expert execution.

        Returns:
            Dictionary mapping attribute names to their domains and predictions
        """
        predictions = {}

        # Extract player position
        if isinstance(state.player.position, DiscreteDistribution):
            predictions["player_position"] = state.player.position.expand_support(
                self.position_domain
            )
        else:
            # Expert didn't modify this attribute - create uniform distribution
            predictions["player_position"] = DiscreteDistribution.from_uniform(
                self.position_domain
            )

        # Extract light states
        for i, light in enumerate(state.lights):
            attr_name = f"light_{i}_is_on"
            if isinstance(light.is_on, DiscreteDistribution):
                predictions[attr_name] = light.is_on.expand_support(self.bool_domain)
            else:
                # Expert didn't modify this attribute - create uniform distribution
                predictions[attr_name] = DiscreteDistribution.from_uniform(
                    self.bool_domain
                )

        return predictions

    def get_observed_outcomes(self, state: GameState) -> dict[ObservableId, int]:
        """Extract ground truth observed values from a state."""
        observed = {}

        # Player position
        observed["player_position"] = state.player.position

        # Light states
        for i, light in enumerate(state.lights):
            observed[f"light_{i}_is_on"] = int(light.is_on)

        return observed

    @staticmethod
    def apply_expert_predictions(
        new_state: GameState,
        expert_predictions: dict[ObservableId, list[DiscreteDistribution]],
        weights: torch.Tensor,
    ) -> GameState:
        # Sample player position
        if "player_position" in expert_predictions:
            player_preds = expert_predictions[ObservableId("player_position")]
            combined_dist = combine_expert_predictions_for_attr(player_preds, weights)
            new_state.player.position = combined_dist.sample()

        # Sample light states
        for i, light in enumerate(new_state.lights):
            attr_name = f"light_{i}_is_on"
            if attr_name in expert_predictions:
                light_preds = expert_predictions[ObservableId(attr_name)]
                combined_dist = combine_expert_predictions_for_attr(
                    light_preds, weights
                )
                new_state.lights[i].is_on = bool(combined_dist.sample())

        return new_state


implements(ObservableExtractorProtocol[GameState])(ObservableExtractor)
