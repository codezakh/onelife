from ...typing_utils import implements
from crafter.state_export import WorldState
from ...poe_world.core import (
    ObservableExtractorProtocol,
    ObservableId,
    DiscreteDistribution,
)
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from ..optimization import combine_expert_predictions_for_attr
from loguru import logger


# Define fixed domains for position and health attributes
# Using reasonable defaults without reading from game state
@dataclass(frozen=True)
class ObservableExtractorConfig:
    position_domain: np.ndarray = field(default_factory=lambda: np.arange(0, 101))
    health_domain: np.ndarray = field(default_factory=lambda: np.arange(0, 101))
    entity_types: list[str] = field(
        default_factory=lambda: ["cow", "zombie", "skeleton", "plant", "arrow", "fence"]
    )
    entity_count_domain: np.ndarray = field(default_factory=lambda: np.arange(0, 50))
    entity_existence_domain: np.ndarray = field(
        default_factory=lambda: np.array([0, 1])
    )


class ObservableExtractor:
    def __init__(self, config: Optional[ObservableExtractorConfig] = None):
        config = config or ObservableExtractorConfig()
        self.position_domain = config.position_domain
        self.health_domain = config.health_domain

        # Define entity types available in Crafter
        self.entity_types = config.entity_types

        # Define domain for entity counts (reasonable range for Crafter)
        self.entity_count_domain = config.entity_count_domain

        # Define domain for entity existence (0 = deleted, 1 = exists)
        self.entity_existence_domain = config.entity_existence_domain

    def extract_attribute_predictions(
        self, state: WorldState
    ) -> dict[ObservableId, DiscreteDistribution]:
        """
        Extract probabilistic predictions from a state after expert execution.
        """
        predictions: dict[ObservableId, DiscreteDistribution] = {}

        # Extract player position
        if hasattr(state.player.position, "x") and isinstance(
            state.player.position.x, DiscreteDistribution
        ):
            predictions[ObservableId("player_position_x")] = (
                state.player.position.x.expand_support(self.position_domain)
            )

        if hasattr(state.player.position, "y") and isinstance(
            state.player.position.y, DiscreteDistribution
        ):
            predictions[ObservableId("player_position_y")] = (
                state.player.position.y.expand_support(self.position_domain)
            )

        # Extract player health
        if isinstance(state.player.health, DiscreteDistribution):
            predictions[ObservableId("player_health")] = (
                state.player.health.expand_support(self.health_domain)
            )

        # Extract entity lifecycle observables
        # Track entity existence (per entity ID) - whether each specific entity exists
        for entity in state.objects:
            if entity.entity_id == state.player.entity_id:
                continue  # Skip player, is always present
            entity_id = entity.entity_id
            predictions[ObservableId(f"entity_exists_{entity_id}")] = (
                DiscreteDistribution([1]).expand_support(self.entity_existence_domain)
            )

        # Track entity counts (per entity type) - total count of each entity type
        entity_counts = {entity_type: 0 for entity_type in self.entity_types}
        for entity in state.objects:
            if entity.entity_id == state.player.entity_id:
                continue  # Skip player, is always present
            entity_counts[entity.name] += 1
        for entity_type in self.entity_types:
            predictions[ObservableId(f"entity_count_{entity_type}")] = (
                DiscreteDistribution([entity_counts[entity_type]]).expand_support(
                    self.entity_count_domain
                )
            )

        # Extract entity positions and health
        for entity in state.objects:
            if entity.entity_id == state.player.entity_id:
                continue  # Skip player as we already handled it

            # Entity position x
            attr_name = f"entity_{entity.entity_id}_position_x"
            if hasattr(entity.position, "x") and isinstance(
                entity.position.x, DiscreteDistribution
            ):
                predictions[ObservableId(attr_name)] = entity.position.x.expand_support(
                    self.position_domain
                )

            # Entity position y
            attr_name = f"entity_{entity.entity_id}_position_y"
            if hasattr(entity.position, "y") and isinstance(
                entity.position.y, DiscreteDistribution
            ):
                predictions[ObservableId(attr_name)] = entity.position.y.expand_support(
                    self.position_domain
                )

            # Entity health
            attr_name = f"entity_{entity.entity_id}_health"
            if isinstance(entity.health, DiscreteDistribution):
                predictions[ObservableId(attr_name)] = entity.health.expand_support(
                    self.health_domain
                )

        return predictions

    def get_observed_outcomes(self, state: WorldState) -> dict[ObservableId, int]:
        """
        Extract ground truth observed values from a state.
        """
        observed: dict[ObservableId, int] = {}

        # Player position
        observed[ObservableId("player_position_x")] = state.player.position.x
        observed[ObservableId("player_position_y")] = state.player.position.y

        # Player health
        observed[ObservableId("player_health")] = state.player.health

        # Extract entity lifecycle observables
        # Count entities by type
        entity_counts = {entity_type: 0 for entity_type in self.entity_types}
        for entity in state.objects:
            if entity.entity_id == state.player.entity_id:
                continue  # Skip player, is always present
            entity_type = entity.name
            entity_counts[entity_type] += 1

        # Record entity counts
        for entity_type in self.entity_types:
            observed[ObservableId(f"entity_count_{entity_type}")] = entity_counts.get(
                entity_type, 0
            )

        # Record entity existence (all entities in state exist)
        for entity in state.objects:
            if entity.entity_id == state.player.entity_id:
                continue  # Skip player, is always present
            entity_id = entity.entity_id
            observed[ObservableId(f"entity_exists_{entity_id}")] = 1

        # Entity positions and health
        for entity in state.objects:
            if entity.entity_id == state.player.entity_id:
                continue  # Skip player as we already handled it

            observed[ObservableId(f"entity_{entity.entity_id}_position_x")] = (
                entity.position.x
            )
            observed[ObservableId(f"entity_{entity.entity_id}_position_y")] = (
                entity.position.y
            )
            observed[ObservableId(f"entity_{entity.entity_id}_health")] = entity.health

        return observed

    @staticmethod
    def apply_expert_predictions(
        new_state: WorldState,
        expert_predictions: dict[ObservableId, list[DiscreteDistribution]],
        weights: torch.Tensor,
    ) -> WorldState:
        """
        Apply combined expert predictions to create a new state.

        Note: For entity lifecycle observables, we don't need to create/delete
        entities here because experts predict by modifying state directly
        (calling create_object(), setting deleted attribute, etc.).
        Our observable extractor just observes what experts did to the state.
        """

        # Sample player position
        if "player_position_x" in expert_predictions:
            with logger.contextualize(attr_name="player_position_x"):
                player_x_preds = expert_predictions[ObservableId("player_position_x")]
                combined_dist = combine_expert_predictions_for_attr(
                    player_x_preds, weights
                )
                new_state.player.position.x = combined_dist.sample()

        if "player_position_y" in expert_predictions:
            with logger.contextualize(attr_name="player_position_y"):
                player_y_preds = expert_predictions[ObservableId("player_position_y")]
                combined_dist = combine_expert_predictions_for_attr(
                    player_y_preds, weights
                )
                new_state.player.position.y = combined_dist.sample()

        # Sample player health
        if "player_health" in expert_predictions:
            with logger.contextualize(attr_name="player_health"):
                player_health_preds = expert_predictions[ObservableId("player_health")]
                combined_dist = combine_expert_predictions_for_attr(
                    player_health_preds, weights
                )
                new_state.player.health = combined_dist.sample()

        # Sample entity positions and health
        for entity in new_state.objects:
            if entity.entity_id == new_state.player.entity_id:
                continue  # Skip player as we already handled it

            with logger.contextualize(
                entity_name=entity.name, entity_id=entity.entity_id
            ):

                # Entity position x
                attr_name = f"entity_{entity.entity_id}_position_x"
                if attr_name in expert_predictions:
                    with logger.contextualize(attr_name=attr_name):
                        entity_x_preds = expert_predictions[ObservableId(attr_name)]
                        combined_dist = combine_expert_predictions_for_attr(
                            entity_x_preds, weights
                        )
                        entity.position.x = combined_dist.sample()

                # Entity position y
                attr_name = f"entity_{entity.entity_id}_position_y"
                if attr_name in expert_predictions:
                    with logger.contextualize(attr_name=attr_name):
                        entity_y_preds = expert_predictions[ObservableId(attr_name)]
                        combined_dist = combine_expert_predictions_for_attr(
                            entity_y_preds, weights
                        )
                        entity.position.y = combined_dist.sample()

                # Entity health
                attr_name = f"entity_{entity.entity_id}_health"
                if attr_name in expert_predictions:
                    with logger.contextualize(attr_name=attr_name):
                        entity_health_preds = expert_predictions[
                            ObservableId(attr_name)
                        ]
                        combined_dist = combine_expert_predictions_for_attr(
                            entity_health_preds, weights
                        )
                        entity.health = combined_dist.sample()

        return new_state


implements(ObservableExtractorProtocol[WorldState])(ObservableExtractor)
