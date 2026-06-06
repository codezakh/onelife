"""Tests for pickle-free LawMixture serialization (onelife.our_method.crafter.serde).

Three invariants, behavioral equivalence being the one that matters most:
  1. law identity + weights survive the round-trip,
  2. the observable extractor config survives *in full* (every field, arrays
     included) — the test that would fail against a hand-picked subset,
  3. a model rebuilt from JSON scores transitions identically to the original,
     i.e. the laws don't just exist, they re-run the same after source->exec.
"""

import copy
import dataclasses
import inspect

import numpy as np
import pytest
from crafter_oo import objects as crafter_objects
from crafter_oo.functional_env import (
    export_world_state,
    initial_state,
    reconstruct_world_from_state,
    transition,
)
from crafter_oo.state_export import CowState, WorldState
from crafter_oo.testing_helpers import player_utils, world_utils

from onelife.evaluator.crafter.utils import MAP_ACTION_TO_INDEX, find_player
from onelife.our_method.action_remapping import remap_slug_actions_to_balrog_actions
from onelife.our_method.core import LawFunctionWrapper, WeightedLaw
from onelife.our_method.crafter.observable_extractor import (
    ObservableExtractor,
    ObservableExtractorConfig,
)
from onelife.our_method.crafter.serde import load_law_mixture, save_law_mixture
from onelife.our_method.world_modeling import LawMixture
from onelife.poe_world.core import DiscreteDistribution


class _StayPutCow:
    """A tiny real law: every cow holds its x position.

    Defined as a normal class so its source (via inspect.getsource) drives the
    fixture, keeping fixture construction independent of serde's own loader.
    """

    def precondition(self, current_state, action) -> bool:
        return True

    def effect(self, current_state, action) -> None:
        for cow in current_state.get_object_of_type_in_update_range(CowState):
            cow.position.x = DiscreteDistribution(support=[cow.position.x])


def _make_model(extractor: ObservableExtractor | None = None) -> LawMixture:
    law = LawFunctionWrapper(
        law=_StayPutCow(),
        source_code=inspect.getsource(_StayPutCow),
        action_remapper=remap_slug_actions_to_balrog_actions,
    )
    return LawMixture(
        observable_extractor=extractor or ObservableExtractor(),
        weighted_laws=[WeightedLaw(law=law, weight=0.42, is_fitted=True)],
    )


@pytest.fixture
def cow_transition() -> tuple[WorldState, str, WorldState]:
    """A (state, action, next_state) with a cow present, so _StayPutCow fires."""
    view = (9, 9)
    state = initial_state(area=(9, 9), view=view, seed=1)
    world = reconstruct_world_from_state(state)

    player = find_player(world)
    player_utils.set_player_position(player, (5, 5))
    for x in range(view[0]):
        for y in range(view[1]):
            world_utils.set_tile_material(world, (x, y), "grass")
    world_utils.add_object_to_world(world, crafter_objects.Cow, (3, 3))

    start_state = export_world_state(world, view=view, step_count=0)
    action = "noop"
    next_state, _ = transition(copy.deepcopy(start_state), MAP_ACTION_TO_INDEX[action])
    return start_state, action, next_state


def test_roundtrip_preserves_laws_and_weights(tmp_path):
    model = _make_model()
    path = tmp_path / "model.json"

    save_law_mixture(model, path)
    loaded = load_law_mixture(path)

    assert [(w.law.__name__, w.weight, w.is_fitted) for w in loaded.laws] == [
        (w.law.__name__, w.weight, w.is_fitted) for w in model.laws
    ]


def test_roundtrip_preserves_full_extractor_config(tmp_path):
    # Non-default scalars AND a non-default array: a hand-picked-subset saver
    # would silently drop top_k and/or position_domain here.
    extractor = ObservableExtractor(
        ObservableExtractorConfig(
            detect_facing_tile=True,
            top_k=5,
            position_domain=np.array([3, 4, 5]),
        )
    )
    path = tmp_path / "model.json"

    save_law_mixture(_make_model(extractor), path)
    loaded = load_law_mixture(path).observable_extractor

    for f in dataclasses.fields(ObservableExtractorConfig):
        original_value = getattr(extractor, f.name)
        loaded_value = getattr(loaded, f.name)
        if isinstance(original_value, np.ndarray):
            assert np.array_equal(loaded_value, original_value), f.name
        else:
            assert loaded_value == original_value, f.name


def test_reconstructed_model_scores_transitions_identically(tmp_path, cow_transition):
    state, action, next_state = cow_transition
    model = _make_model()
    path = tmp_path / "model.json"

    save_law_mixture(model, path)
    loaded = load_law_mixture(path)

    assert loaded.evaluate_log_probability(
        state, action, next_state
    ) == model.evaluate_log_probability(state, action, next_state)
