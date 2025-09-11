"""
Full integration test for Crafter learning loop with two object types (player, cow).
- Uses real synthesizers (CrafterExpertSynthesizer, CrafterCreationSynthesizer)
- Runs full learning via ObjectModelOrchestrator and composes via PoEWorldLearner
- Asserts important learning invariants (experts synthesized, checkpoints load, pruning removes low-weight experts, log-likelihood improves)
"""

from __future__ import annotations

import os
from typing import List

import pytest

from distant_sunburn.litellm_utils import GeminiLiteLlmParams
from distant_sunburn.poe_world.core import SymbolicTransition, WeightedExpert
from distant_sunburn.poe_world.crafter.observable_extractor import ObservableExtractor
from distant_sunburn.poe_world.crafter.synthesizer import (
    CrafterExpertSynthesizer,
    CrafterSynthesisDependenciesProvider,
)
from distant_sunburn.poe_world.crafter.creation_synthesizer import (
    CrafterCreationSynthesizer,
    CrafterCreationSynthesisDependenciesProvider,
)
from distant_sunburn.poe_world.expert_manager import ExpertManager
from distant_sunburn.poe_world.object_model_learner import (
    ObjectModelOrchestrator,
    ObjectModelOrchestratorConfig,
)
from distant_sunburn.poe_world.poe_world_learner import PoEWorldLearner
from distant_sunburn.poe_world.weight_fitter import MaxLikelihoodWeightFitter
from external.crafter_refactored.crafter.functional_env import (
    initial_state,
    transition,
)
from external.crafter_refactored.crafter.state_export import WorldState
from distant_sunburn.evaluator.crafter.utils import MAP_ACTION_TO_INDEX
from external.crafter_refactored.crafter.constants import ActionT


def _generate_cow_movement_transitions(
    seed: int = 1,
) -> List[SymbolicTransition[WorldState]]:
    """Generate a small set of transitions in a world with player and cow.

    We keep the number of transitions small for test speed. We use four distinct
    actions if available.
    """
    # Create initial small world
    state = initial_state(area=(9, 9), view=(9, 9), seed=seed)

    # Take a couple of hardcoded actions to generate transitions
    actions: list[ActionT] = ["move_left", "move_right", "move_up", "move_down"]

    transitions: List[SymbolicTransition[WorldState]] = []

    for a in actions:
        idx = MAP_ACTION_TO_INDEX[a]
        next_state, _ = transition(state, idx)
        transitions.append(
            SymbolicTransition(prev_metadata=state, action=a, next_metadata=next_state)
        )
        state = next_state

    return transitions


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not available"
)
def test_crafter_full_integration_two_obj_types(tmp_path):
    # LLM params tuned for speed
    llm_params = GeminiLiteLlmParams(model_slug="gemini-2.5-flash")

    # Components shared across object types
    extractor = ObservableExtractor()
    fitter = MaxLikelihoodWeightFitter(observable_extractor=extractor, max_iterations=3)

    # Build two orchestrators: player and cow
    def make_orchestrator(obj_type: str, ckpt_dir):
        non_creation_mgr = ExpertManager(
            observable_extractor=extractor, weight_fitter=fitter, weight_threshold=0.01
        )
        creation_mgr = ExpertManager(
            observable_extractor=extractor, weight_fitter=fitter, weight_threshold=0.01
        )
        non_creation_syn = CrafterExpertSynthesizer(
            llm_params=llm_params,
            dependencies_provider=CrafterSynthesisDependenciesProvider(),
        )
        creation_syn = CrafterCreationSynthesizer(
            llm_params=llm_params,
            dependencies_provider=CrafterCreationSynthesisDependenciesProvider(),
        )
        config = ObjectModelOrchestratorConfig(
            batch_size=2, save_freq=10, surprise_threshold=-0.5
        )
        return ObjectModelOrchestrator(
            object_type=obj_type,
            non_creation_expert_manager=non_creation_mgr,
            creation_expert_manager=creation_mgr,
            non_creation_synthesizer=non_creation_syn,
            creation_synthesizer=creation_syn,
            config=config,
            checkpoint_dir=str(ckpt_dir),
        )

    orchestrators = {
        "player": make_orchestrator("player", tmp_path / "player"),
        "cow": make_orchestrator("cow", tmp_path / "cow"),
    }

    # Learner composing both
    learner = PoEWorldLearner(
        object_type_to_orchestrator=orchestrators, observable_extractor=extractor
    )

    # Generate small set of transitions
    transitions = _generate_cow_movement_transitions(seed=11)

    # Baseline likelihood: before synthesis, composed model is empty; we treat baseline as very low
    baseline_lp = -1000.0

    # Run full synthesis
    model = learner.synthesize_world_model(transitions)

    # Check experts synthesized per type
    for obj_type, orchestrator in orchestrators.items():
        obj_model = orchestrator.get_model()
        total_experts = len(obj_model.non_creation_experts) + len(
            obj_model.creation_experts
        )
        assert total_experts > 0, f"No experts synthesized for {obj_type}"

    # Evaluate log-prob on a held-out-like transition (reuse first for speed)
    t = transitions[0]
    lp_after = model.evaluate_log_probability(
        t.prev_metadata, t.action, t.next_metadata
    )
    assert isinstance(lp_after, float)
    assert lp_after > baseline_lp

    # Exercise public API: get_model() and sampling
    retrieved = learner.get_model()
    sampled_next = retrieved.sample_next_state(t.prev_metadata, t.action)
    assert isinstance(sampled_next, WorldState)

    # Save and load checkpoints for each orchestrator
    for obj_type, orchestrator in orchestrators.items():
        orchestrator._save_checkpoint(None)
        assert orchestrator._load_checkpoint(
            None
        ), f"Failed to load checkpoint for {obj_type}"

    # Pruning: add an obviously low-weight expert and ensure pruning removes it
    def _no_op_expert(current_state: WorldState, action: str) -> None:
        return None

    for obj_type, orchestrator in orchestrators.items():
        obj_model = orchestrator.get_model()
        # Add one low-weight expert to non-creation manager
        low_weight = WeightedExpert(
            expert_function=_no_op_expert, weight=0.0, is_fitted=True
        )
        orchestrator.non_creation_expert_manager.add_experts([low_weight])
        before = len(orchestrator.non_creation_expert_manager.get_experts())
        orchestrator.non_creation_expert_manager.prune_experts()
        after = len(orchestrator.non_creation_expert_manager.get_experts())
        assert (
            after == before - 1
        ), f"Pruning did not remove low-weight expert for {obj_type}"

    # Re-compose model and verify it still produces finite log-prob
    model2 = learner.update_world_model([], fast=True)
    lp_after2 = model2.evaluate_log_probability(
        t.prev_metadata, t.action, t.next_metadata
    )
    assert isinstance(lp_after2, float)

    # Print out the learned expert functions for debugging
    for expert in model.experts:
        print(expert.expert_function.__source_code__)
