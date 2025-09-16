from ....typing_utils import implements
from .interface import Mutator
from crafter.state_export import WorldState
from crafter.constants import ActionT


class PlayerHealthMutator:
    def __init__(self):
        self.category = "Combat"

    def precondition(self, state: WorldState, action: ActionT) -> bool:
        return True

    def __call__(self, state: WorldState, action: ActionT) -> WorldState:
        mutated_state = state.model_copy(deep=True)

        delta = state.random_state.randint(-3, 3)
        mutated_state.player.health = mutated_state.player.health + delta

        # Clamp health between 0 and 10
        mutated_state.player.health = max(0, min(mutated_state.player.health, 10))

        return mutated_state


implements(Mutator)(PlayerHealthMutator)
