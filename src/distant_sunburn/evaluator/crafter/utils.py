"""
Utility functions for Crafter evaluation.
"""

from typing import Optional, Type, TypeVar, Union

import crafter.constants as crafter_constants
import crafter.objects as crafter_objects
from crafter.constants import ActionT
from crafter.engine import World
from crafter.env import Env
from crafter.state_export import (
    ArrowState,
    CowState,
    FenceState,
    PlantState,
    PlayerState,
    SkeletonState,
    WorldState,
    ZombieState,
    export_world_state,
)

CrafterObjectState = Union[
    PlayerState,
    CowState,
    ZombieState,
    SkeletonState,
    ArrowState,
    PlantState,
    FenceState,
]

CrafterObjectStateT = TypeVar("CrafterObjectStateT", bound=CrafterObjectState)


def get_world_state(env: Env) -> WorldState:
    """Exports the WorldState from a crafter.Env instance."""
    assert env._step is not None
    return export_world_state(env._world, view=env._config.view, step_count=env._step)


def find_player(world: World) -> crafter_objects.Player:
    """Finds the player in the world."""
    for obj in world.objects:
        if isinstance(obj, crafter_objects.Player):
            return obj
    raise ValueError("No player found in world")


def find_object_in_state(
    state: WorldState, entity_id: int, entity_type: Type[CrafterObjectStateT]
) -> Optional[CrafterObjectStateT]:
    """Finds an object in the state by entity_id and type."""
    for obj in state.objects:
        if isinstance(obj, entity_type) and obj.entity_id == entity_id:
            return obj
    return None


def find_all_objects_for_type(
    state: WorldState, entity_type: Type[CrafterObjectStateT]
) -> list[CrafterObjectStateT]:
    """Finds all objects of a given type in the state."""
    return [obj for obj in state.objects if isinstance(obj, entity_type)]


MAP_ACTION_TO_INDEX: dict[ActionT, int] = {
    action: index for index, action in enumerate(crafter_constants.actions)
}
