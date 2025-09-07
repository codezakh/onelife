from crafter.state_export import WorldState
import jsonpatch
import random
from typing import Optional

from ..core import (
    DistractorGenerator,
    SymbolicTransition,
    EditDistanceCalculator,
    EditDistance,
)
from .mutators import DEFAULT_MUTATORS, Mutator
from crafter.constants import ActionT as CrafterAction
from ...typing_utils import implements
from loguru import logger
from typing import Sequence
from ...json_utils import flatten_json_to_pathmap


# Note: This is almost a copy of the format_state function used to generate
# training data for the neural world model in e0008, except we _do not_ exclude
# the materials field.
def _gamestate_to_json(state: WorldState) -> dict:
    excluded_fields = {"event_bus", "serialized_random_state"}

    serialized_state = state.model_dump(exclude=excluded_fields, mode="json")

    def format_serialized_state(serialized_state: dict) -> dict:
        # Remove the player field from the .objects list, so it isn't duplicated
        # since it is already in the .player field.
        serialized_state["objects"] = [
            obj for obj in serialized_state["objects"] if obj["name"] != "player"
        ]

        # Sort the objects by entity_id
        serialized_state["objects"] = sorted(
            serialized_state["objects"], key=lambda x: x["entity_id"]
        )

        # Sort the chunks by chunk_key
        serialized_state["chunks"] = sorted(
            serialized_state["chunks"], key=lambda x: x["chunk_key"]
        )

        # For each chunk, sort the objects within the chunk
        for chunk in serialized_state["chunks"]:
            chunk["objects"] = sorted(chunk["objects"])

        return serialized_state

    return format_serialized_state(serialized_state)


class JSONPatchEditDistance:
    def __call__(self, state1: WorldState, state2: WorldState) -> EditDistance:

        json1 = _gamestate_to_json(state1)
        json2 = _gamestate_to_json(state2)
        patch = jsonpatch.make_patch(json1, json2)
        raw_edit_distance = len(list(patch))

        flattened_json1 = flatten_json_to_pathmap(json1)
        total_elements = len(flattened_json1)
        normalized_edit_distance = raw_edit_distance / total_elements

        return EditDistance(
            raw=raw_edit_distance,
            normalized=normalized_edit_distance,
            total_elements=total_elements,
        )


implements(EditDistanceCalculator[WorldState])(JSONPatchEditDistance)


class CrafterDistractorGenerator:
    """
    Generates distractors for Crafter evaluation by applying mutators to ground truth states.
    """

    def __init__(self, seed: int = 42, mutators: Optional[Sequence[Mutator]] = None):
        self.rng = random.Random(seed)
        self.mutators = mutators or DEFAULT_MUTATORS
        self.logger = logger.bind(mutator=self.__class__.__name__)
        self.logger.info(
            f"Initialized {self.__class__.__name__} with {len(self.mutators)} mutators"
        )

    def _log_mutator_active_states(
        self, mutator_active_states: dict[tuple[int, Mutator], bool]
    ):
        # Split them into two lists, one for active mutators and one for inactive mutators
        active_mutators = [
            mutator for mutator, active in mutator_active_states.items() if active
        ]
        inactive_mutators = [
            mutator for mutator, active in mutator_active_states.items() if not active
        ]
        self.logger.trace(f"Active mutators: {active_mutators}")
        self.logger.trace(f"Inactive mutators: {inactive_mutators}")

    def __call__(
        self,
        transition: SymbolicTransition[WorldState, CrafterAction],
        all_transitions: list[SymbolicTransition[WorldState, CrafterAction]],
        num_distractors: int,
    ) -> list[WorldState]:
        """
        Generate distractors by applying mutators to the ground truth next state.

        Args:
            transition: The ground truth transition
            all_transitions: All available transitions (unused in current implementation)
            num_distractors: Number of distractors to generate

        Returns:
            List of mutated states that are plausible but incorrect
        """
        distractors: list[WorldState] = []

        # Collect some logging information about which mutators were active
        # for the transition.

        mutator_active_states: dict[tuple[int, Mutator], bool] = dict()

        for idx, mutator in enumerate(self.mutators):
            if mutator.precondition(transition.prev_metadata, transition.action):
                mutator_active_states[(idx, mutator)] = True
                mutated_state = mutator(transition.prev_metadata, transition.action)
                if mutated_state == transition.next_metadata:
                    self.logger.warning(
                        f"Mutator {mutator.__class__.__name__} generated a state that is the same as the true next state."
                    )
                    continue
                else:
                    distractors.append(mutated_state)
            else:
                mutator_active_states[(idx, mutator)] = False

            if len(distractors) >= num_distractors:
                break

        self._log_mutator_active_states(mutator_active_states)

        return distractors


implements(DistractorGenerator[WorldState, CrafterAction])(CrafterDistractorGenerator)
