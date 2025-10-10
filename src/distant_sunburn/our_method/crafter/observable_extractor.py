from ...typing_utils import implements
from crafter.state_export import WorldState
from ...poe_world.core import (
    ObservableId,
    DiscreteDistribution,
)
from ..core import ObservableExtractorProtocol
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from ..world_modeling import combine_active_expert_predictions_for_attr
from typing import Mapping, TypeAlias
from loguru import logger
from crafter.state_export import Inventory
from crafter.constants import MaterialT, materials
from icecream import ic

# Setting the threshold to 0.01 means that
# we will _always_ sample even if
# the experts have put a logprob as high as 0.0 (100% probability)
# on a single value.
LOGP_DETERMINISTIC_THRESHOLD_NEVER = 0.01


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
    inventory_domain: np.ndarray = field(default_factory=lambda: np.arange(0, 101))
    logp_deterministic_threshold: float = field(
        default=LOGP_DETERMINISTIC_THRESHOLD_NEVER
    )
    noise_logscore: float = field(default=-10.0)
    top_k: Optional[int] = field(default=None)


ExpertIndex: TypeAlias = int


class InventoryAttrSetter:
    def __init__(self, inventory: Inventory):
        self.inventory = inventory

    def set_health(self, v: int):
        self.inventory.health = v

    def set_food(self, v: int):
        self.inventory.food = v

    def set_drink(self, v: int):
        self.inventory.drink = v

    def set_energy(self, v: int):
        self.inventory.energy = v

    def set_sapling(self, v: int):
        self.inventory.sapling = v

    def set_wood(self, v: int):
        self.inventory.wood = v

    def set_stone(self, v: int):
        self.inventory.stone = v

    def set_coal(self, v: int):
        self.inventory.coal = v

    def set_iron(self, v: int):
        self.inventory.iron = v

    def set_diamond(self, v: int):
        self.inventory.diamond = v

    def set_wood_pickaxe(self, v: int):
        self.inventory.wood_pickaxe = v

    def set_stone_pickaxe(self, v: int):
        self.inventory.stone_pickaxe = v

    def set_iron_pickaxe(self, v: int):
        self.inventory.iron_pickaxe = v

    def set_wood_sword(self, v: int):
        self.inventory.wood_sword = v

    def set_stone_sword(self, v: int):
        self.inventory.stone_sword = v

    def set_iron_sword(self, v: int):
        self.inventory.iron_sword = v


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

        # Define domain for inventory (reasonable range for Crafter)
        self.inventory_domain = config.inventory_domain

        self.logp_deterministic_threshold = config.logp_deterministic_threshold
        self.noise_logscore = config.noise_logscore
        self.top_k = config.top_k

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
                state.player.position.x.expand_support(
                    self.position_domain, noise_logscore=self.noise_logscore
                )
            )

        if hasattr(state.player.position, "y") and isinstance(
            state.player.position.y, DiscreteDistribution
        ):
            predictions[ObservableId("player_position_y")] = (
                state.player.position.y.expand_support(
                    self.position_domain, noise_logscore=self.noise_logscore
                )
            )

        # Extract player health
        if isinstance(state.player.health, DiscreteDistribution):
            predictions[ObservableId("player_health")] = (
                state.player.health.expand_support(
                    self.health_domain, noise_logscore=self.noise_logscore
                )
            )

        # Extract entity lifecycle observables
        # Track entity existence (per entity ID) - whether each specific entity exists
        for entity in state.objects:
            if entity.entity_id == state.player.entity_id:
                continue  # Skip player, is always present
            entity_id = entity.entity_id
            predictions[
                ObservableId(f"entity_exists_{entity_id}")
            ] = DiscreteDistribution([1]).expand_support(
                self.entity_existence_domain, noise_logscore=self.noise_logscore
            )

        # Track entity counts (per entity type) - total count of each entity type
        entity_counts = {entity_type: 0 for entity_type in self.entity_types}
        for entity in state.objects:
            if entity.entity_id == state.player.entity_id:
                continue  # Skip player, is always present
            entity_counts[entity.name] += 1
        for entity_type in self.entity_types:
            predictions[
                ObservableId(f"entity_count_{entity_type}")
            ] = DiscreteDistribution([entity_counts[entity_type]]).expand_support(
                self.entity_count_domain, noise_logscore=self.noise_logscore
            )

        # Extract inventory observables
        if isinstance(state.player.inventory.health, DiscreteDistribution):
            predictions[ObservableId("player_inventory_health")] = (
                state.player.inventory.health.expand_support(
                    self.inventory_domain, noise_logscore=self.noise_logscore
                )
            )

        if isinstance(state.player.inventory.food, DiscreteDistribution):
            predictions[ObservableId("player_inventory_food")] = (
                state.player.inventory.food.expand_support(
                    self.inventory_domain, noise_logscore=self.noise_logscore
                )
            )

        if isinstance(state.player.inventory.drink, DiscreteDistribution):
            predictions[ObservableId("player_inventory_drink")] = (
                state.player.inventory.drink.expand_support(
                    self.inventory_domain, noise_logscore=self.noise_logscore
                )
            )

        if isinstance(state.player.inventory.energy, DiscreteDistribution):
            predictions[ObservableId("player_inventory_energy")] = (
                state.player.inventory.energy.expand_support(
                    self.inventory_domain, noise_logscore=self.noise_logscore
                )
            )

        if isinstance(state.player.inventory.sapling, DiscreteDistribution):
            predictions[ObservableId("player_inventory_sapling")] = (
                state.player.inventory.sapling.expand_support(
                    self.inventory_domain, noise_logscore=self.noise_logscore
                )
            )

        if isinstance(state.player.inventory.wood, DiscreteDistribution):
            predictions[ObservableId("player_inventory_wood")] = (
                state.player.inventory.wood.expand_support(
                    self.inventory_domain, noise_logscore=self.noise_logscore
                )
            )

        if isinstance(state.player.inventory.stone, DiscreteDistribution):
            predictions[ObservableId("player_inventory_stone")] = (
                state.player.inventory.stone.expand_support(
                    self.inventory_domain, noise_logscore=self.noise_logscore
                )
            )

        if isinstance(state.player.inventory.coal, DiscreteDistribution):
            predictions[ObservableId("player_inventory_coal")] = (
                state.player.inventory.coal.expand_support(
                    self.inventory_domain, noise_logscore=self.noise_logscore
                )
            )

        if isinstance(state.player.inventory.iron, DiscreteDistribution):
            predictions[ObservableId("player_inventory_iron")] = (
                state.player.inventory.iron.expand_support(
                    self.inventory_domain, noise_logscore=self.noise_logscore
                )
            )

        if isinstance(state.player.inventory.diamond, DiscreteDistribution):
            predictions[ObservableId("player_inventory_diamond")] = (
                state.player.inventory.diamond.expand_support(
                    self.inventory_domain, noise_logscore=self.noise_logscore
                )
            )

        if isinstance(state.player.inventory.wood_pickaxe, DiscreteDistribution):
            predictions[ObservableId("player_inventory_wood_pickaxe")] = (
                state.player.inventory.wood_pickaxe.expand_support(
                    self.inventory_domain, noise_logscore=self.noise_logscore
                )
            )

        if isinstance(state.player.inventory.stone_pickaxe, DiscreteDistribution):
            predictions[ObservableId("player_inventory_stone_pickaxe")] = (
                state.player.inventory.stone_pickaxe.expand_support(
                    self.inventory_domain, noise_logscore=self.noise_logscore
                )
            )

        if isinstance(state.player.inventory.iron_pickaxe, DiscreteDistribution):
            predictions[ObservableId("player_inventory_iron_pickaxe")] = (
                state.player.inventory.iron_pickaxe.expand_support(
                    self.inventory_domain, noise_logscore=self.noise_logscore
                )
            )

        if isinstance(state.player.inventory.wood_sword, DiscreteDistribution):
            predictions[ObservableId("player_inventory_wood_sword")] = (
                state.player.inventory.wood_sword.expand_support(
                    self.inventory_domain, noise_logscore=self.noise_logscore
                )
            )

        if isinstance(state.player.inventory.stone_sword, DiscreteDistribution):
            predictions[ObservableId("player_inventory_stone_sword")] = (
                state.player.inventory.stone_sword.expand_support(
                    self.inventory_domain, noise_logscore=self.noise_logscore
                )
            )

        if isinstance(state.player.inventory.iron_sword, DiscreteDistribution):
            predictions[ObservableId("player_inventory_iron_sword")] = (
                state.player.inventory.iron_sword.expand_support(
                    self.inventory_domain, noise_logscore=self.noise_logscore
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
                    self.position_domain, noise_logscore=self.noise_logscore
                )

            # Entity position y
            attr_name = f"entity_{entity.entity_id}_position_y"
            if hasattr(entity.position, "y") and isinstance(
                entity.position.y, DiscreteDistribution
            ):
                predictions[ObservableId(attr_name)] = entity.position.y.expand_support(
                    self.position_domain, noise_logscore=self.noise_logscore
                )

            # Entity health
            attr_name = f"entity_{entity.entity_id}_health"
            if isinstance(entity.health, DiscreteDistribution):
                predictions[ObservableId(attr_name)] = entity.health.expand_support(
                    self.health_domain, noise_logscore=self.noise_logscore
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

        # Extract inventory observables
        observed[ObservableId("player_inventory_health")] = (
            state.player.inventory.health
        )
        observed[ObservableId("player_inventory_food")] = state.player.inventory.food
        observed[ObservableId("player_inventory_drink")] = state.player.inventory.drink
        observed[ObservableId("player_inventory_energy")] = (
            state.player.inventory.energy
        )
        observed[ObservableId("player_inventory_sapling")] = (
            state.player.inventory.sapling
        )
        observed[ObservableId("player_inventory_wood")] = state.player.inventory.wood
        observed[ObservableId("player_inventory_stone")] = state.player.inventory.stone
        observed[ObservableId("player_inventory_coal")] = state.player.inventory.coal
        observed[ObservableId("player_inventory_iron")] = state.player.inventory.iron
        observed[ObservableId("player_inventory_diamond")] = (
            state.player.inventory.diamond
        )
        observed[ObservableId("player_inventory_wood_pickaxe")] = (
            state.player.inventory.wood_pickaxe
        )
        observed[ObservableId("player_inventory_stone_pickaxe")] = (
            state.player.inventory.stone_pickaxe
        )
        observed[ObservableId("player_inventory_iron_pickaxe")] = (
            state.player.inventory.iron_pickaxe
        )
        observed[ObservableId("player_inventory_wood_sword")] = (
            state.player.inventory.wood_sword
        )
        observed[ObservableId("player_inventory_stone_sword")] = (
            state.player.inventory.stone_sword
        )
        observed[ObservableId("player_inventory_iron_sword")] = (
            state.player.inventory.iron_sword
        )

        return observed

    def apply_expert_predictions(
        self,
        new_state: WorldState,
        expert_predictions: Mapping[
            ObservableId, Mapping[ExpertIndex, DiscreteDistribution]
        ],
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
                combined_dist = combine_active_expert_predictions_for_attr(
                    player_x_preds, weights
                )
                new_state.player.position.x = combined_dist.sample(
                    self.logp_deterministic_threshold, self.top_k
                )

        if "player_position_y" in expert_predictions:
            with logger.contextualize(attr_name="player_position_y"):
                player_y_preds = expert_predictions[ObservableId("player_position_y")]
                combined_dist = combine_active_expert_predictions_for_attr(
                    player_y_preds, weights
                )
                new_state.player.position.y = combined_dist.sample(
                    self.logp_deterministic_threshold, self.top_k
                )

        # Sample player health
        if "player_health" in expert_predictions:
            with logger.contextualize(attr_name="player_health"):
                player_health_preds = expert_predictions[ObservableId("player_health")]
                combined_dist = combine_active_expert_predictions_for_attr(
                    player_health_preds, weights
                )
                new_state.player.health = combined_dist.sample(
                    self.logp_deterministic_threshold, self.top_k
                )

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
                        combined_dist = combine_active_expert_predictions_for_attr(
                            entity_x_preds, weights
                        )
                        entity.position.x = combined_dist.sample(
                            self.logp_deterministic_threshold, self.top_k
                        )

                # Entity position y
                attr_name = f"entity_{entity.entity_id}_position_y"
                if attr_name in expert_predictions:
                    with logger.contextualize(attr_name=attr_name):
                        entity_y_preds = expert_predictions[ObservableId(attr_name)]
                        combined_dist = combine_active_expert_predictions_for_attr(
                            entity_y_preds, weights
                        )
                        entity.position.y = combined_dist.sample(
                            self.logp_deterministic_threshold, self.top_k
                        )

                # Entity health
                attr_name = f"entity_{entity.entity_id}_health"
                if attr_name in expert_predictions:
                    with logger.contextualize(attr_name=attr_name):
                        entity_health_preds = expert_predictions[
                            ObservableId(attr_name)
                        ]
                        combined_dist = combine_active_expert_predictions_for_attr(
                            entity_health_preds, weights
                        )
                        entity.health = combined_dist.sample(
                            self.logp_deterministic_threshold, self.top_k
                        )

        inventory_attr_setter = InventoryAttrSetter(new_state.player.inventory)

        inventory_observables = [
            ("player_inventory_health", inventory_attr_setter.set_health),
            ("player_inventory_food", inventory_attr_setter.set_food),
            ("player_inventory_drink", inventory_attr_setter.set_drink),
            ("player_inventory_energy", inventory_attr_setter.set_energy),
            ("player_inventory_sapling", inventory_attr_setter.set_sapling),
            ("player_inventory_wood", inventory_attr_setter.set_wood),
            ("player_inventory_stone", inventory_attr_setter.set_stone),
            ("player_inventory_coal", inventory_attr_setter.set_coal),
            ("player_inventory_iron", inventory_attr_setter.set_iron),
            ("player_inventory_diamond", inventory_attr_setter.set_diamond),
            ("player_inventory_wood_pickaxe", inventory_attr_setter.set_wood_pickaxe),
            ("player_inventory_stone_pickaxe", inventory_attr_setter.set_stone_pickaxe),
            ("player_inventory_iron_pickaxe", inventory_attr_setter.set_iron_pickaxe),
            ("player_inventory_wood_sword", inventory_attr_setter.set_wood_sword),
            ("player_inventory_stone_sword", inventory_attr_setter.set_stone_sword),
            ("player_inventory_iron_sword", inventory_attr_setter.set_iron_sword),
        ]
        for attr_name, attr_setter in inventory_observables:
            if attr_name in expert_predictions:
                with logger.contextualize(attr_name=attr_name):
                    attr_preds = expert_predictions[ObservableId(attr_name)]
                    combined_dist = combine_active_expert_predictions_for_attr(
                        attr_preds, weights
                    )
                    attr_setter(
                        combined_dist.sample(
                            self.logp_deterministic_threshold, self.top_k
                        )
                    )

        return new_state


implements(ObservableExtractorProtocol[WorldState])(ObservableExtractor)
