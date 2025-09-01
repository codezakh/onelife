"""
Integration test for ExpertManager with ObjectModelOrchestrator.

This test demonstrates the complete integration of ExpertManager with the
ObjectModelOrchestrator, showing how the wrapper enables the orchestrator
to work with concrete implementations rather than just protocols.
"""

import pytest
from typing import List

from distant_sunburn.poe_world.core import SymbolicTransition, WeightedExpert
from distant_sunburn.poe_world.expert_manager import ExpertManager
from distant_sunburn.poe_world.object_model_learner import (
    ObjectModelOrchestrator,
    LearningConfig,
    ExpertManagerProtocol,
)
from distant_sunburn.poe_world.weight_fitter import MaxLikelihoodWeightFitter
from distant_sunburn.poe_world.simple_1d_env.observable_extractor import (
    ObservableExtractor,
)
from distant_sunburn.simple_1d_env.environment import (
    initial_state,
    transition_function,
    Action,
    DEFAULT_LAWS,
    GameState,
    WorldConfig,
)
from distant_sunburn.poe_world.simple_1d_env.handwritten_experts import (
    CORRECT_EXPERTS,
    INCORRECT_EXPERTS,
)


class MockExpertSynthesizer:
    """Mock synthesizer for testing purposes."""

    def __init__(self, experts_to_return: List[WeightedExpert]):
        self.experts_to_return = experts_to_return
        self.synthesize_calls = 0

    async def synthesize_experts(self, transitions, object_type):
        """Mock synthesis that returns predefined experts."""
        self.synthesize_calls += 1
        return self.experts_to_return.copy()


def generate_test_transitions(
    n_transitions: int = 50, seed: int = 42
) -> List[SymbolicTransition[GameState]]:
    """Generate test transitions for integration testing."""
    import random
    import numpy as np

    rng = random.Random(seed)
    np.random.seed(seed)

    transitions = []
    current_state = initial_state(WorldConfig(seed=seed))

    for _ in range(n_transitions):
        action = rng.choice(list(Action))
        next_state = transition_function(current_state, action, DEFAULT_LAWS)

        transition = SymbolicTransition(
            prev_metadata=current_state, action=action, next_metadata=next_state
        )
        transitions.append(transition)
        current_state = next_state

    return transitions


def test_expert_manager_with_orchestrator(tmp_path):
    """
    Integration test that validates ExpertManager works with ObjectModelOrchestrator.

    This test:
    1. Creates ExpertManager instances for non-creation and creation experts
    2. Creates mock synthesizers that return predefined experts
    3. Creates an ObjectModelOrchestrator with the ExpertManager instances
    4. Adds datapoints and runs inference
    5. Validates that the orchestrator can successfully use the ExpertManager
    """
    # Set up components
    observable_extractor = ObservableExtractor()
    weight_fitter = MaxLikelihoodWeightFitter(
        observable_extractor=observable_extractor,
        max_iterations=5,  # Use fewer iterations for faster tests
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

    # Create mock synthesizers that return some predefined experts
    non_creation_experts = [
        WeightedExpert(expert_function=expert, weight=1.0)
        for expert in CORRECT_EXPERTS[:2]
    ]
    creation_experts = [
        WeightedExpert(expert_function=expert, weight=1.0)
        for expert in INCORRECT_EXPERTS[:1]
    ]

    non_creation_synthesizer = MockExpertSynthesizer(non_creation_experts)
    creation_synthesizer = MockExpertSynthesizer(creation_experts)

    # Create learning config
    config = LearningConfig(
        batch_size=5,
        save_freq=20,
        surprise_threshold=-1.0,  # Lower threshold to trigger synthesis
        fast_update_frequency=3,
        max_experts_per_object_type=10,
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

    # Generate test data
    transitions = generate_test_transitions(30)

    # Add datapoints to orchestrator
    for transition in transitions:
        orchestrator.add_datapoint(transition)

    # Run inference
    result = orchestrator.infer_moe()

    # Validate results
    assert result.object_type == "player"
    assert isinstance(result.non_creation_experts, list)
    assert isinstance(result.creation_experts, list)

    # Check that synthesizers were called (indicating surprising transitions were found)
    assert (
        non_creation_synthesizer.synthesize_calls > 0
        or creation_synthesizer.synthesize_calls > 0
    )

    # Check that experts were added to managers
    total_experts = len(result.non_creation_experts) + len(result.creation_experts)
    assert total_experts > 0, "Should have at least some experts after inference"

    # Check that expert weights have been learned (not just initial values)
    all_experts = result.non_creation_experts + result.creation_experts
    weights = [expert.weight for expert in all_experts]
    assert any(
        w != 1.0 for w in weights
    ), "At least some weights should have been learned"


def test_fast_inference_with_expert_manager(tmp_path):
    """
    Test that fast inference works correctly with ExpertManager.

    This test validates that the fast inference mode properly uses the
    fast_mode parameter of ExpertManager.fit_weights().
    """
    # Set up components
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

    # Create mock synthesizers
    non_creation_synthesizer = MockExpertSynthesizer(
        [WeightedExpert(expert_function=CORRECT_EXPERTS[0], weight=1.0)]
    )
    creation_synthesizer = MockExpertSynthesizer(
        [WeightedExpert(expert_function=INCORRECT_EXPERTS[0], weight=1.0)]
    )

    # Create learning config
    config = LearningConfig(
        batch_size=3,
        save_freq=10,
        surprise_threshold=-1.0,
        fast_update_frequency=2,
        max_experts_per_object_type=5,
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
    initial_transitions = generate_test_transitions(10)
    for transition in initial_transitions:
        orchestrator.add_datapoint(transition)

    # Run full inference first
    full_result = orchestrator.infer_moe()

    # Add more datapoints
    additional_transitions = generate_test_transitions(5, seed=123)
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
