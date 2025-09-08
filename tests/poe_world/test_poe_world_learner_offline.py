"""
Integration test for PoEWorldLearner: offline synthesis and fast update using 1D env components.
"""

from __future__ import annotations

from typing import List

import numpy as np

from distant_sunburn.poe_world.core import SymbolicTransition, WeightedExpert
from distant_sunburn.poe_world.poe_world_learner import PoEWorldLearner
from distant_sunburn.poe_world.world_model import PoEWorldModel
from distant_sunburn.poe_world.expert_manager import ExpertManager
from distant_sunburn.poe_world.weight_fitter import MaxLikelihoodWeightFitter
from distant_sunburn.poe_world.simple_1d_env.observable_extractor import (
    ObservableExtractor as OneDExtractor,
)
from distant_sunburn.poe_world.object_model_learner import (
    ObjectModelOrchestrator,
    ObjectModelOrchestratorConfig,
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
        action = rng.choice(list(Action))  # type: ignore
        next_state = transition_function(state, action, DEFAULT_LAWS)
        transitions.append(
            SymbolicTransition(
                prev_metadata=state, action=action, next_metadata=next_state
            )
        )
        state = next_state

    return transitions


def test_offline_synthesis_and_fast_update(tmp_path):
    extractor = OneDExtractor()
    fitter = MaxLikelihoodWeightFitter(observable_extractor=extractor, max_iterations=3)

    # Expert managers for a single pseudo object type "player"
    non_creation_mgr = ExpertManager(
        observable_extractor=extractor, weight_fitter=fitter, weight_threshold=0.01
    )
    creation_mgr = ExpertManager(
        observable_extractor=extractor, weight_fitter=fitter, weight_threshold=0.01
    )

    # Create orchestrator with low thresholds (synthetic experts not required here; managers can start empty)
    config = ObjectModelOrchestratorConfig(
        batch_size=10, save_freq=100, surprise_threshold=-2.0
    )

    # For the test, we won't use LLM synthesizers. Use no-op synthesizers by injecting empty lists via minimal mocks.
    class _NoOpSynth:
        async def synthesize_experts(self, transitions, object_type):
            return []

    orchestrator = ObjectModelOrchestrator(
        object_type="player",
        non_creation_expert_manager=non_creation_mgr,
        creation_expert_manager=creation_mgr,
        non_creation_synthesizer=_NoOpSynth(),
        creation_synthesizer=_NoOpSynth(),
        config=config,
        checkpoint_dir=str(tmp_path),
    )

    # Seed some initial experts manually (using correct experts) so composition isn't empty
    seeded = [
        WeightedExpert(expert_function=fn, weight=1.0, is_fitted=False)
        for fn in CORRECT_EXPERTS
    ]
    non_creation_mgr.add_experts(seeded)

    learner = PoEWorldLearner(
        object_type_to_orchestrator={"player": orchestrator},
        observable_extractor=extractor,
    )

    # Offline synthesis (will run full fit over seeded experts)
    transitions = _generate_transitions(20, seed=1)
    model = learner.synthesize_world_model(transitions)

    # Sanity: model should be a PoEWorldModel and have experts
    assert isinstance(model, PoEWorldModel)
    assert len(model.experts) >= len(seeded)

    # Evaluate a fresh transition
    t = _generate_transitions(1, seed=2)[0]
    lp_before = model.evaluate_log_probability(
        t.prev_metadata, t.action, t.next_metadata
    )

    # Add more data and do a fast update
    extra = _generate_transitions(10, seed=3)
    model2 = learner.update_world_model(extra, fast=True)
    lp_after = model2.evaluate_log_probability(
        t.prev_metadata, t.action, t.next_metadata
    )

    assert isinstance(lp_before, float)
    assert isinstance(lp_after, float)
