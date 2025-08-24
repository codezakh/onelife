import json

from crafter.state_export import WorldState
import jsonpatch


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
