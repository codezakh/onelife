from dataclasses import dataclass

from crafter_oo.state_reconstruction import (
    ArrowState,
    CowState,
    FenceState,
    PlantState,
    PlayerState,
    SkeletonState,
    WorldState,
    ZombieState,
)
from .transition import Transition


from typing import Protocol


class ChangeDetector(Protocol):
    """Protocol for change detection in state transitions."""

    aspect: str

    def has_changes(self, transition: "Transition") -> bool:
        """Check if there are changes in this aspect of the state."""
        ...


@dataclass
class PlayerInventoryChangeDetector:
    """Detects changes in player inventory."""

    aspect: str = "player_inventory"

    def has_changes(self, transition: "Transition") -> bool:
        """Check if player inventory changed."""
        if transition.state.player is None or transition.next_state.player is None:
            return False

        return (
            transition.state.player.inventory != transition.next_state.player.inventory
        )


@dataclass
class PlayerPositionChangeDetector:
    """Detects changes in player position."""

    aspect: str = "player_position"

    def has_changes(self, transition: "Transition") -> bool:
        """Check if player position changed."""
        if transition.state.player is None or transition.next_state.player is None:
            return False

        return transition.state.player.position != transition.next_state.player.position


@dataclass
class MapTilesChangeDetector:
    """Detects changes in map tiles within local view."""

    aspect: str = "map_tiles"

    def has_changes(self, transition: "Transition") -> bool:
        """Check if map tiles changed within local view."""
        if transition.state.player is None:
            return False

        # Get local view bounds
        player_x, player_y = (
            transition.state.player.position.x,
            transition.state.player.position.y,
        )
        view_x, view_y = transition.state.view

        start_x = max(0, player_x - view_x)
        end_x = min(len(transition.state.materials), player_x + view_x + 1)
        start_y = max(0, player_y - view_y)
        end_y = min(
            len(transition.state.materials[0]) if transition.state.materials else 0,
            player_y + view_y + 1,
        )

        # Check if any materials changed in the local view
        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                if (
                    x < len(transition.state.materials)
                    and y < len(transition.state.materials[x])
                    and x < len(transition.next_state.materials)
                    and y < len(transition.next_state.materials[x])
                ):
                    if (
                        transition.state.materials[x][y]
                        != transition.next_state.materials[x][y]
                    ):
                        return True

        return False


def filter_objects_to_local_view(state: WorldState) -> list:
    """
    Filter objects to only those visible in the local view around the player.

    Args:
        state: The world state containing objects, player position, and view range

    Returns:
        List of objects that are within the local view bounds
    """
    if state.player is None:
        raise ValueError("Local view is undefined with no player in the state.")

    # Get player position and view range
    player_x, player_y = state.player.position.x, state.player.position.y
    view_x, view_y = state.view

    # Calculate bounds for the local view - ensure player is centered
    start_x = max(0, player_x - view_x)
    end_x = min(len(state.materials), player_x + view_x + 1)
    start_y = max(0, player_y - view_y)
    end_y = min(
        len(state.materials[0]) if state.materials else 0, player_y + view_y + 1
    )

    # Filter objects to only those within the local view bounds
    visible_objects: list[
        ZombieState
        | CowState
        | ArrowState
        | PlantState
        | FenceState
        | PlayerState
        | SkeletonState
    ] = []
    for obj in state.objects:
        if hasattr(obj, "position") and obj.position:
            obj_x, obj_y = obj.position.x, obj.position.y
            # Check if object is within the local view bounds
            if start_x <= obj_x < end_x and start_y <= obj_y < end_y:
                visible_objects.append(obj)

    return visible_objects


@dataclass
class EntityTypeChangeDetector:
    """Detects changes in entities of a specific type within local view."""

    aspect: str
    entity_type: type

    def has_changes(self, transition: "Transition") -> bool:
        """Check if entities of this type changed within local view."""
        # Get objects in local view for both states
        state_objects = filter_objects_to_local_view(transition.state)
        next_state_objects = filter_objects_to_local_view(transition.next_state)

        # Filter to only entities of the target type
        state_entities = [
            obj for obj in state_objects if isinstance(obj, self.entity_type)
        ]
        next_state_entities = [
            obj for obj in next_state_objects if isinstance(obj, self.entity_type)
        ]

        # Create lookup by entity_id
        state_by_id = {obj.entity_id: obj for obj in state_entities}
        next_state_by_id = {obj.entity_id: obj for obj in next_state_entities}

        # Check for changes in existing entities
        for entity_id, state_entity in state_by_id.items():
            if entity_id in next_state_by_id:
                if state_entity != next_state_by_id[entity_id]:
                    return True
            else:
                # Entity disappeared
                return True

        # Check for new entities
        for entity_id in next_state_by_id:
            if entity_id not in state_by_id:
                return True

        return False


@dataclass
class ZombieHealthChangeDetector:
    """Detects changes in zombie health specifically."""

    aspect: str = "zombie_health"

    def has_changes(self, transition: "Transition") -> bool:
        """Check if any zombies have health changes."""
        # Get all zombies from both states (not just local view, since health changes could be important globally)
        state_zombies = [
            obj for obj in transition.state.objects if isinstance(obj, ZombieState)
        ]
        next_state_zombies = [
            obj for obj in transition.next_state.objects if isinstance(obj, ZombieState)
        ]

        # Create lookup by entity_id
        state_by_id = {obj.entity_id: obj for obj in state_zombies}
        next_state_by_id = {obj.entity_id: obj for obj in next_state_zombies}

        # Check for health changes in existing zombies
        for entity_id, state_zombie in state_by_id.items():
            if entity_id in next_state_by_id:
                next_zombie = next_state_by_id[entity_id]
                if hasattr(state_zombie, "health") and hasattr(next_zombie, "health"):
                    if state_zombie.health != next_zombie.health:
                        return True

        return False


@dataclass
class PlayerHealthChangeDetector:
    """Detects changes in player health specifically."""

    aspect: str = "player_health"

    def has_changes(self, transition: "Transition") -> bool:
        """Check if player health changed."""
        if transition.state.player is None or transition.next_state.player is None:
            return False

        return transition.state.player.health != transition.next_state.player.health


def find_transitions_with_attribute_changes(
    all_transitions: list[Transition],
) -> dict[str, dict[str, list[int]]]:
    """
    Find transitions that show changes to various entity attributes.

    Returns a nested dict: entity_type -> attribute -> list of transition indices
    where that attribute changed for that entity type.
    """
    attribute_changes: dict[str, dict[str, list[int]]] = {}

    # Define which attributes to track for each entity type (excluding immobile entities)
    entity_attributes = {
        "player": [
            "health",
            "position",
            "facing",
            "action",
            "sleeping",
            "achievements",
            "inventory",
            "thirst",
            "hunger",
            "fatigue",
            "recover",
            "last_health",
        ],
        "cow": ["health", "position"],
        "zombie": ["health", "position", "cooldown"],
        "skeleton": ["health", "position", "reload"],
        "arrow": ["health", "position", "facing"],
    }

    # Entity type to state class mapping
    entity_type_to_class = {
        "player": PlayerState,
        "cow": CowState,
        "zombie": ZombieState,
        "skeleton": SkeletonState,
        "arrow": ArrowState,
    }

    for transition_idx, transition in enumerate(all_transitions):
        # Get entities from both states
        state_entities = {obj.entity_id: obj for obj in transition.state.objects}
        next_state_entities = {
            obj.entity_id: obj for obj in transition.next_state.objects
        }

        # Check for changes in each entity type
        for entity_type, attributes in entity_attributes.items():
            entity_class = entity_type_to_class[entity_type]

            # Initialize dict for this entity type
            if entity_type not in attribute_changes:
                attribute_changes[entity_type] = {}

            # Find entities of this type in both states
            for entity_id, state_entity in state_entities.items():
                if isinstance(state_entity, entity_class):
                    if entity_id in next_state_entities:
                        next_entity = next_state_entities[entity_id]
                        if isinstance(next_entity, entity_class):
                            # Check each attribute for changes
                            for attr in attributes:
                                # Initialize list for this attribute
                                if attr not in attribute_changes[entity_type]:
                                    attribute_changes[entity_type][attr] = []

                                # Check if attribute changed
                                if hasattr(state_entity, attr) and hasattr(
                                    next_entity, attr
                                ):
                                    state_val = getattr(state_entity, attr)
                                    next_val = getattr(next_entity, attr)

                                    if state_val != next_val:
                                        # Only add if not already in the list
                                        if (
                                            transition_idx
                                            not in attribute_changes[entity_type][attr]
                                        ):
                                            attribute_changes[entity_type][attr].append(
                                                transition_idx
                                            )

    return attribute_changes
