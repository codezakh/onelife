"""
Unit tests for composed PoEWorldModel built from aggregated experts.

Uses simple 1D handwritten experts as stand-ins for multiple object types.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from distant_sunburn.poe_world.core import SymbolicTransition, WeightedExpert
from distant_sunburn.poe_world.world_model import PoEWorldModel
from distant_sunburn.poe_world.simple_1d_env.observable_extractor import (
    ObservableExtractor,
)
from distant_sunburn.poe_world.simple_1d_env.handwritten_experts import (
    CORRECT_EXPERTS,
)
from distant_sunburn.simple_1d_env.environment import (
    initial_state,
    transition_function,
    Action,
    DEFAULT_LAWS,
    GameState,
    WorldConfig,
)


def _generate_transitions(
    n: int = 20, seed: int = 0
) -> List[SymbolicTransition[GameState]]:
    rng = np.random.default_rng(seed)
    state = initial_state(WorldConfig(seed=seed))
    transitions: List[SymbolicTransition[GameState]] = []

    for _ in range(n):
        action = rng.choice(list(Action))
        next_state = transition_function(state, action, DEFAULT_LAWS)
        transitions.append(
            SymbolicTransition(
                prev_metadata=state, action=action, next_metadata=next_state
            )
        )
        state = next_state

    return transitions


def test_logprob_prefers_true_next_state():
    extractor = ObservableExtractor()

    # Build weighted experts from existing correct experts
    experts = [
        WeightedExpert(expert_function=fn, weight=1.0, is_fitted=True)
        for fn in CORRECT_EXPERTS
    ]

    model = PoEWorldModel(extractor, experts)

    transitions = _generate_transitions(10, seed=123)
    t = transitions[0]

    logp_true = model.evaluate_log_probability(
        t.prev_metadata, t.action, t.next_metadata
    )

    # Create a wrong next state by flipping player position in a bounded way
    wrong_next = initial_state(WorldConfig(seed=999))
    wrong_next.player.position = min(
        wrong_next.config.width - 1, t.next_metadata.player.position + 2
    )

    logp_wrong = model.evaluate_log_probability(t.prev_metadata, t.action, wrong_next)

    assert isinstance(logp_true, float)
    assert isinstance(logp_wrong, float)
    assert logp_true >= logp_wrong


def test_sampling_returns_valid_state():
    extractor = ObservableExtractor()
    experts = [
        WeightedExpert(expert_function=fn, weight=1.0, is_fitted=True)
        for fn in CORRECT_EXPERTS
    ]

    model = PoEWorldModel(extractor, experts)

    state = initial_state(WorldConfig(seed=7))
    action = Action.MOVE_RIGHT

    next_state = model.sample_next_state(state, action)

    assert isinstance(next_state, GameState)
    assert 0 <= next_state.player.position < next_state.config.width
