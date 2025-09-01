"""
Integration test for ExpertManager with ObjectModelOrchestrator using real Crafter components.

This test validates that the full pipeline works end-to-end with:
1. Real CrafterExpertSynthesizer (LLM-based expert generation)
2. CrafterObservableExtractor
3. MaxLikelihoodWeightFitter
4. ExpertManager
5. ObjectModelOrchestrator

The test uses simple movement transitions to trigger expert synthesis.
"""

import os
import pytest
import numpy as np
from pathlib import Path

from distant_sunburn.poe_world.core import SymbolicTransition, WeightedExpert
from distant_sunburn.poe_world.expert_manager import ExpertManager
from distant_sunburn.poe_world.object_model_learner import (
    ObjectModelOrchestrator,
    ObjectModelOrchestratorConfig,
)
from distant_sunburn.poe_world.weight_fitter import MaxLikelihoodWeightFitter
from distant_sunburn.poe_world.crafter.observable_extractor import ObservableExtractor
from distant_sunburn.poe_world.crafter.synthesizer import CrafterExpertSynthesizer
from crafter.state_export import WorldState
from crafter.functional_env import (
    initial_state,
    reconstruct_world_from_state,
    export_world_state,
    transition,
)
from crafter.constants import ActionT


def create_simple_movement_transitions() -> list[SymbolicTransition[WorldState]]:
    """
    Create two simple movement transitions for testing.

    Creates a scenario where:
    1. Player moves right from (2, 2) to (3, 2)
    2. Player moves down from (3, 2) to (3, 3)

    These are simple, predictable movements that should be easy for the LLM to synthesize.
    """
    transitions = []

    # Create initial state with player at (2, 2)
    view = (5, 5)
    initial_state_obj = initial_state(area=(5, 5), view=view, seed=42)
    world = reconstruct_world_from_state(initial_state_obj)

    # Configure player at specific position
    player = _find_player(world)
    player.pos = np.array([2, 2])

    # Export initial state
    initial_state_obj = export_world_state(world, view=view, step_count=0)

    # Transition 1: Move right from (2, 2) to (3, 2)
    world = reconstruct_world_from_state(initial_state_obj)
    player = _find_player(world)
    player.pos = np.array([3, 2])
    next_state_1 = export_world_state(world, view=view, step_count=1)

    transition_1 = SymbolicTransition(
        prev_metadata=initial_state_obj,
        action="move_right",
        next_metadata=next_state_1,
    )
    transitions.append(transition_1)

    # Transition 2: Move down from (3, 2) to (3, 3)
    world = reconstruct_world_from_state(next_state_1)
    player = _find_player(world)
    player.pos = np.array([3, 3])
    next_state_2 = export_world_state(world, view=view, step_count=2)

    transition_2 = SymbolicTransition(
        prev_metadata=next_state_1,
        action="move_down",
        next_metadata=next_state_2,
    )
    transitions.append(transition_2)

    return transitions


def _find_player(world):
    """Find the player object in the world."""
    for obj in world.objects:
        if hasattr(obj, "pos"):  # Player has pos attribute
            return obj
    raise ValueError("No player found in world")


def test_crafter_integration_with_real_synthesizer(tmp_path: Path):
    """
    Integration test that validates the full crafter pipeline works end-to-end.

    This test:
    1. Creates ExpertManager instances for non-creation and creation experts
    2. Uses real CrafterExpertSynthesizer (LLM-based expert generation)
    3. Creates an ObjectModelOrchestrator with the ExpertManager instances
    4. Adds simple movement transitions and runs inference
    5. Validates that the orchestrator can successfully use the ExpertManager
    6. Checks that experts were synthesized and weights learned

    Note: This test requires GEMINI_API_KEY to be set in the environment.
    """
    # Skip if no API key is available
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not available")

    # Set up real crafter components
    observable_extractor = ObservableExtractor()
    weight_fitter = MaxLikelihoodWeightFitter(
        observable_extractor=observable_extractor,
        max_iterations=5,  # Keep low for faster tests
    )

    # Create expert managers
    non_creation_manager = ExpertManager(
        observable_extractor=observable_extractor,
        weight_fitter=weight_fitter,
        weight_threshold=0.01,
    )
    creation_manager = ExpertManager(
        observable_extractor=observable_extractor,
        weight_fitter=weight_fitter,
        weight_threshold=0.01,
    )

    # Create real synthesizers (LLM-based)
    non_creation_synthesizer = CrafterExpertSynthesizer()
    creation_synthesizer = CrafterExpertSynthesizer()

    # Create learning config with low surprise threshold to trigger synthesis
    config = ObjectModelOrchestratorConfig(
        batch_size=2,  # Small batch size for our 2 transitions
        save_freq=10,
        surprise_threshold=-0.5,  # Low threshold to ensure synthesis is triggered
    )

    # Create orchestrator with temporary checkpoint directory
    orchestrator = ObjectModelOrchestrator(
        object_type="player",
        non_creation_expert_manager=non_creation_manager,
        creation_expert_manager=creation_manager,
        non_creation_synthesizer=non_creation_synthesizer,
        creation_synthesizer=creation_synthesizer,
        config=config,
        checkpoint_dir=str(tmp_path),
    )

    # Generate simple movement transitions
    transitions = create_simple_movement_transitions()
    assert len(transitions) == 2, "Should have exactly 2 transitions"

    # Add datapoints to orchestrator
    for transition in transitions:
        orchestrator.add_datapoint(transition)

    # Run inference (should trigger synthesis due to low surprise threshold)
    result = orchestrator.infer_moe()

    # Validate results
    assert result.object_type == "player"
    assert isinstance(result.non_creation_experts, list)
    assert isinstance(result.creation_experts, list)

    # Check that experts were added to managers
    total_experts = len(result.non_creation_experts) + len(result.creation_experts)
    assert total_experts > 0, "Should have at least some experts after inference"

    # Check that expert weights have been learned (not just initial values)
    all_experts = result.non_creation_experts + result.creation_experts
    weights = [expert.weight for expert in all_experts]
    assert any(
        w != 1.0 for w in weights
    ), "At least some weights should have been learned"

    # Print out the synthesized experts for debugging
    print(f"\n=== Synthesized Experts ===")
    print(f"Non-creation experts: {len(result.non_creation_experts)}")
    print(f"Creation experts: {len(result.creation_experts)}")

    for i, expert in enumerate(all_experts):
        expert_type = (
            "non-creation" if expert in result.non_creation_experts else "creation"
        )
        print(f"\n{expert_type.capitalize()} Expert {i+1}:")
        print(f"  Weight: {expert.weight:.4f}")
        print(f"  Is fitted: {expert.is_fitted}")

        # Print basic expert info
        print(f"  Function: {expert.expert_function}")

    # Test that generated experts are actually callable
    if all_experts:
        test_expert = all_experts[0]
        assert callable(
            test_expert.expert_function
        ), "Expert function should be callable"

        # Test that the function can be called without errors
        # Create a simple test state
        test_state = transitions[
            0
        ].prev_metadata  # Use the first transition's initial state

        # Call the expert function with a test action
        result = test_expert.expert_function(test_state, "move_right")
        print(f"\n=== Expert Function Test ===")
        print(f"Function call result: {result}")
        print(f"Function executed successfully without errors")


def test_fast_inference_with_crafter_expert_manager(tmp_path: Path):
    """
    Test that fast inference works correctly with ExpertManager in crafter environment.

    This test validates that the fast inference mode properly uses the
    fast_mode parameter of ExpertManager.fit_weights().
    """
    # Skip if no API key is available
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not available")

    # Set up real crafter components
    observable_extractor = ObservableExtractor()
    weight_fitter = MaxLikelihoodWeightFitter(
        observable_extractor=observable_extractor,
        max_iterations=5,
    )

    # Create expert managers
    non_creation_manager = ExpertManager(
        observable_extractor=observable_extractor,
        weight_fitter=weight_fitter,
        weight_threshold=0.01,
    )
    creation_manager = ExpertManager(
        observable_extractor=observable_extractor,
        weight_fitter=weight_fitter,
        weight_threshold=0.01,
    )

    # Create real synthesizers
    non_creation_synthesizer = CrafterExpertSynthesizer()
    creation_synthesizer = CrafterExpertSynthesizer()

    # Create learning config
    config = ObjectModelOrchestratorConfig(
        batch_size=2,
        save_freq=10,
        surprise_threshold=-0.5,
    )

    # Create orchestrator with temporary checkpoint directory
    orchestrator = ObjectModelOrchestrator(
        object_type="player",
        non_creation_expert_manager=non_creation_manager,
        creation_expert_manager=creation_manager,
        non_creation_synthesizer=non_creation_synthesizer,
        creation_synthesizer=creation_synthesizer,
        config=config,
        checkpoint_dir=str(tmp_path),
    )

    # Add initial datapoints and run full inference
    initial_transitions = create_simple_movement_transitions()
    for transition in initial_transitions:
        orchestrator.add_datapoint(transition)

    # Run full inference first
    full_result = orchestrator.infer_moe()

    # Add more datapoints (same transitions but different step counts)
    additional_transitions = create_simple_movement_transitions()
    for transition in additional_transitions:
        orchestrator.add_datapoint(transition)

    # Run fast inference
    fast_result = orchestrator.fast_infer_moe()

    # Validate that fast inference completed successfully
    assert fast_result.object_type == "player"
    assert len(fast_result.non_creation_experts) >= len(
        full_result.non_creation_experts
    )
    assert len(fast_result.creation_experts) >= len(full_result.creation_experts)

    print(f"\n=== Fast Inference Test ===")
    print(
        f"Full inference experts: {len(full_result.non_creation_experts)} non-creation, {len(full_result.creation_experts)} creation"
    )
    print(
        f"Fast inference experts: {len(fast_result.non_creation_experts)} non-creation, {len(fast_result.creation_experts)} creation"
    )
