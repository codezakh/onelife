from ....typing_utils import implements
from .v1 import Mutator
from crafter.state_export import WorldState, Position
from crafter.constants import ActionT
import random


NON_MOVEMENT_ACTIONS: set[ActionT] = {
    "noop",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
}

MOVEMENT_ACTIONS: set[ActionT] = {
    "move_left",
    "move_right",
    "move_up",
    "move_down",
}

DIRECTIONS = (
    Position(x=0, y=1),
    Position(x=1, y=0),
    Position(x=0, y=-1),
    Position(x=-1, y=0),
)


class IllegalMovementMutator:
    def __init__(self):
        self.category = "Physics"

    def precondition(self, state: WorldState, action: ActionT) -> bool:
        """
        Produce a movement on an action that ordinarily does not result in a movement.
        """

        return action in NON_MOVEMENT_ACTIONS

    def __call__(self, state: WorldState, action: ActionT) -> WorldState:
        mutated_state = state.model_copy(deep=True)
        mutated_state.player.facing = random.choice(DIRECTIONS)

        mutated_state.player.position += mutated_state.player.facing

        return mutated_state


implements(Mutator)(IllegalMovementMutator)
