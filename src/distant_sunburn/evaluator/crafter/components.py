from crafter.state_export import WorldState
import jsonpatch
import random

from ..core import DistractorGenerator, SymbolicTransition
from .mutators import Mutator, AddIllegalItemMutator, TeleportEntityToIllegalTileMutator
from crafter.constants import ActionT as CrafterAction
from ...typing_utils import implements


# Note: This is almost a copy of the format_state function used to generate
# training data for the neural world model in e0008, except we _do not_ exclude
# the materials field.
def _gamestate_to_json(state: WorldState) -> dict:
    excluded_fields = {"event_bus", "serialized_random_state"}

    serialized_state = state.model_dump(exclude=excluded_fields)

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
    def __call__(self, state1: WorldState, state2: WorldState) -> int:
        json1 = _gamestate_to_json(state1)
        json2 = _gamestate_to_json(state2)
        patch = jsonpatch.make_patch(json1, json2)
        return len(list(patch))


class CrafterDistractorGenerator:
    """
    Generates distractors for Crafter evaluation by applying mutators to ground truth states.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.mutators: list[Mutator] = [
            AddIllegalItemMutator("wood", 1),
            AddIllegalItemMutator("stone", 1),
            AddIllegalItemMutator("iron_pickaxe", 1),
            TeleportEntityToIllegalTileMutator(seed=seed),
        ]

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
        max_attempts = num_distractors * 10  # Prevent infinite loops
        attempts = 0

        while len(distractors) < num_distractors and attempts < max_attempts:
            # Sample a random mutator
            mutator = self.rng.choice(self.mutators)

            # Check if the mutator can be applied
            if mutator.precondition(transition.action, transition.next_metadata):
                try:
                    # Apply the mutation
                    mutated_state = mutator(transition.next_metadata)

                    # Ensure we don't return the original state
                    if mutated_state != transition.next_metadata:
                        distractors.append(mutated_state)

                except Exception:
                    # If mutation fails, continue to next attempt
                    pass

            attempts += 1

        return distractors


implements(DistractorGenerator[WorldState, CrafterAction])(CrafterDistractorGenerator)
