import numpy as np
import pytest

from crafter.functional_env import initial_state
from crafter.state_export import WorldState
from distant_sunburn.evaluator.crafter.components import _gamestate_to_json
from distant_sunburn.json_utils import sanitize_model_numpy_types

from pydantic_core import PydanticSerializationError


def test_gamestate_dump_fails_with_numpy_dtype():
    state: WorldState = initial_state()

    state.player.inventory.wood = np.int64(
        3
    )  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(PydanticSerializationError):
        _ = _gamestate_to_json(state)


def test_gamestate_dump_succeeds_after_sanitization():
    state: WorldState = initial_state()

    state.player.inventory.wood = np.int64(
        7
    )  # pyright: ignore[reportAttributeAccessIssue]

    sanitized = sanitize_model_numpy_types(state)

    result = _gamestate_to_json(sanitized)
    assert isinstance(result["player"]["inventory"]["wood"], int)
