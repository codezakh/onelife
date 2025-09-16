from ....typing_utils import implements
from .interface import Mutator
from crafter.state_export import WorldState, Position
from crafter.constants import ActionT
import random
from crafter.state_export import PlayerState


class EntityPositionMutator:
    def __init__(self):
        self.category = "Physics"

    def precondition(self, state: WorldState, action: ActionT) -> bool:
        """
        Produce a movement on an action that ordinarily does not result in a movement.
        """
        return True

    def __call__(self, state: WorldState, action: ActionT) -> WorldState:
        # Move all entities by at least 2 tiles in a random direction
        mutated_state = state.model_copy(deep=True)

        for entity in mutated_state.objects:
            if isinstance(entity, PlayerState):
                continue

            # Generate a random direction and move by at least 2 tiles
            # Choose a random direction (up, down, left, right, or diagonal)
            direction_x = random.choice([-1, 0, 1])
            direction_y = random.choice([-1, 0, 1])

            # Ensure we actually move (not (0, 0))
            if direction_x == 0 and direction_y == 0:
                direction_x = random.choice([-1, 1])

            # Move by at least 2 tiles
            distance = random.randint(2, 10)

            new_x = entity.position.x + direction_x * distance
            new_y = entity.position.y + direction_y * distance

            entity.position = Position(x=new_x, y=new_y)

        return mutated_state


implements(Mutator)(EntityPositionMutator)
