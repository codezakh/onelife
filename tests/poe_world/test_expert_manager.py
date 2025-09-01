"""
Tests for ExpertManager implementation.

This module tests the ExpertManager class that implements ExpertManagerProtocol
by wrapping the existing MaxLikelihoodWeightFitter and PoEWorldModel components.
"""

import pytest
from typing import List

from distant_sunburn.poe_world.core import SymbolicTransition, WeightedExpert
from distant_sunburn.poe_world.expert_manager import ExpertManager
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
    ALL_EXPERTS,
)


def generate_test_transitions(
    n_transitions: int = 100, seed: int = 42
) -> List[SymbolicTransition[GameState]]:
    """Generate test transitions for testing."""
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


class TestExpertManager:
    """Test suite for ExpertManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.observable_extractor = ObservableExtractor()
        self.weight_fitter = MaxLikelihoodWeightFitter(
            observable_extractor=self.observable_extractor,
            max_iterations=5,  # Use fewer iterations for faster tests
        )
        self.manager = ExpertManager(
            observable_extractor=self.observable_extractor,
            weight_fitter=self.weight_fitter,
            weight_threshold=0.01,
        )
        self.transitions = generate_test_transitions(50)

    def test_starts_with_no_experts(self):
        """Test that ExpertManager starts with an empty expert list."""
        assert len(self.manager.get_experts()) == 0

    def test_add_experts(self):
        """Test adding experts to the manager."""
        # Create some test experts
        experts = [
            WeightedExpert(expert_function=expert, weight=1.0)
            for expert in CORRECT_EXPERTS[:2]
        ]

        self.manager.add_experts(experts)

        assert len(self.manager.get_experts()) == 2
        assert all(expert.weight == 1.0 for expert in self.manager.get_experts())

    def test_fit_weights_full_mode(self):
        """Test weight fitting in full mode."""
        # Add some experts
        experts = [
            WeightedExpert(expert_function=expert, weight=1.0)
            for expert in ALL_EXPERTS[:3]
        ]
        self.manager.add_experts(experts)

        # Fit weights
        self.manager.fit_weights(self.transitions, fast_mode=False)

        # Check that weights have been updated (should be different from 1.0)
        fitted_experts = self.manager.get_experts()
        assert len(fitted_experts) == 3

        # Weights should be non-negative
        assert all(expert.weight >= 0 for expert in fitted_experts)

        # At least some weights should be different from initial 1.0
        weights = [expert.weight for expert in fitted_experts]
        assert any(w != 1.0 for w in weights)

    def test_fit_weights_fast_mode(self):
        """Test weight fitting in fast mode."""
        # Add initial experts and fit them
        initial_experts = [
            WeightedExpert(expert_function=expert, weight=1.0)
            for expert in CORRECT_EXPERTS[:2]
        ]
        self.manager.add_experts(initial_experts)
        self.manager.fit_weights(self.transitions, fast_mode=False)

        # Get initial weights
        initial_weights = [expert.weight for expert in self.manager.get_experts()]

        # Add new experts
        new_experts = [
            WeightedExpert(expert_function=expert, weight=1.0)
            for expert in INCORRECT_EXPERTS[:2]
        ]
        self.manager.add_experts(new_experts)

        # Fit weights in fast mode (should only fit new experts)
        self.manager.fit_weights(self.transitions, fast_mode=True)

        # Check that we have all experts
        all_experts = self.manager.get_experts()
        assert len(all_experts) == 4

        # Check that initial expert weights are preserved
        for i, expert in enumerate(all_experts[:2]):
            assert expert.weight == initial_weights[i]

        # Check that new experts have been fitted (weights should be different from 1.0)
        new_expert_weights = [expert.weight for expert in all_experts[2:]]
        assert any(w != 1.0 for w in new_expert_weights)

    def test_prune_experts(self):
        """Test expert pruning based on weight threshold."""
        # Add experts with different weights
        experts = [
            WeightedExpert(
                expert_function=CORRECT_EXPERTS[0], weight=0.005
            ),  # Below threshold
            WeightedExpert(
                expert_function=CORRECT_EXPERTS[1], weight=0.02
            ),  # Above threshold
            WeightedExpert(
                expert_function=INCORRECT_EXPERTS[0], weight=0.001
            ),  # Below threshold
        ]
        self.manager.add_experts(experts)

        # Prune experts
        self.manager.prune_experts()

        # Should only have one expert remaining (weight 0.02)
        remaining_experts = self.manager.get_experts()
        assert len(remaining_experts) == 1
        assert remaining_experts[0].weight == 0.02

    def test_evaluate_log_probability(self):
        """Test log probability evaluation."""
        # Add some experts and fit weights
        experts = [
            WeightedExpert(expert_function=expert, weight=1.0)
            for expert in CORRECT_EXPERTS[:2]
        ]
        self.manager.add_experts(experts)
        self.manager.fit_weights(self.transitions, fast_mode=False)

        # Test evaluation on a transition
        transition = self.transitions[0]
        log_prob = self.manager.evaluate_log_probability(
            transition.prev_metadata, transition.action, transition.next_metadata
        )

        # Should return a finite float
        assert isinstance(log_prob, float)
        assert not (log_prob != log_prob)  # Not NaN

    def test_save_checkpoint(self, tmp_path):
        """Test that checkpoint can be saved successfully."""
        # Add experts and fit weights
        experts = [
            WeightedExpert(expert_function=expert, weight=1.0)
            for expert in ALL_EXPERTS[:3]
        ]
        self.manager.add_experts(experts)
        self.manager.fit_weights(self.transitions, fast_mode=False)

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.safetensors"
        self.manager.save(str(checkpoint_path))

        # Check that file was created
        assert checkpoint_path.exists()

    def test_load_checkpoint_fails_without_experts(self, tmp_path):
        """Test that loading checkpoint fails when no experts are present."""
        # Add experts and fit weights
        experts = [
            WeightedExpert(expert_function=expert, weight=1.0)
            for expert in ALL_EXPERTS[:3]
        ]
        self.manager.add_experts(experts)
        self.manager.fit_weights(self.transitions, fast_mode=False)

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.safetensors"
        self.manager.save(str(checkpoint_path))

        # Create new manager without experts
        new_manager = ExpertManager(
            observable_extractor=self.observable_extractor,
            weight_fitter=self.weight_fitter,
            weight_threshold=0.5,
        )

        # Load should fail initially (no experts to match weights)
        assert not new_manager.load(str(checkpoint_path))

    def test_load_checkpoint_restores_state(self, tmp_path):
        """Test that loading checkpoint restores manager state correctly."""
        # Add experts and fit weights
        experts = [
            WeightedExpert(expert_function=expert, weight=1.0)
            for expert in ALL_EXPERTS[:3]
        ]
        self.manager.add_experts(experts)
        self.manager.fit_weights(self.transitions, fast_mode=False)

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.safetensors"
        self.manager.save(str(checkpoint_path))

        # Create new manager and add same experts
        new_manager = ExpertManager(
            observable_extractor=self.observable_extractor,
            weight_fitter=self.weight_fitter,
            weight_threshold=0.5,
        )
        new_manager.add_experts(experts)

        # Load checkpoint
        assert new_manager.load(str(checkpoint_path))

        # Check that weight threshold was restored
        assert abs(new_manager.weight_threshold - 0.01) < 1e-6

        # Check that expert count was restored
        assert len(new_manager.get_experts()) == 3

    def test_load_checkpoint_restores_weights(self, tmp_path):
        """Test that loading checkpoint restores expert weights correctly."""
        # Add experts and fit weights
        experts = [
            WeightedExpert(expert_function=expert, weight=1.0)
            for expert in ALL_EXPERTS[:3]
        ]
        self.manager.add_experts(experts)
        self.manager.fit_weights(self.transitions, fast_mode=False)

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.safetensors"
        self.manager.save(str(checkpoint_path))

        # Create new manager and add same experts
        new_manager = ExpertManager(
            observable_extractor=self.observable_extractor,
            weight_fitter=self.weight_fitter,
            weight_threshold=0.5,
        )
        new_manager.add_experts(experts)

        # Load checkpoint
        new_manager.load(str(checkpoint_path))

        # Check that weights were restored
        original_weights = [expert.weight for expert in self.manager.get_experts()]
        loaded_weights = [expert.weight for expert in new_manager.get_experts()]
        assert original_weights == loaded_weights

    def test_empty_transitions_handling(self):
        """Test handling of empty transitions list."""
        # Add some experts
        experts = [
            WeightedExpert(expert_function=expert, weight=1.0)
            for expert in CORRECT_EXPERTS[:2]
        ]
        self.manager.add_experts(experts)

        # Should handle empty transitions gracefully
        self.manager.fit_weights([], fast_mode=False)

        # Experts should still be there with original weights
        assert len(self.manager.get_experts()) == 2
        assert all(expert.weight == 1.0 for expert in self.manager.get_experts())

    def test_prune_empty_experts(self):
        """Test pruning when no experts are present."""
        # Should handle empty expert list gracefully
        self.manager.prune_experts()
        assert len(self.manager.get_experts()) == 0

    def test_fast_mode_no_new_experts(self):
        """Test fast mode when no new experts are present."""
        # Add experts and fit them
        experts = [
            WeightedExpert(expert_function=expert, weight=1.0)
            for expert in CORRECT_EXPERTS[:2]
        ]
        self.manager.add_experts(experts)
        self.manager.fit_weights(self.transitions, fast_mode=False)

        # Try fast mode with no new experts
        self.manager.fit_weights(self.transitions, fast_mode=True)

        # Should still have the same experts
        assert len(self.manager.get_experts()) == 2

    def test_pruning_respects_custom_threshold(self):
        """Test that pruning respects the configured weight threshold."""
        manager = ExpertManager(
            observable_extractor=self.observable_extractor,
            weight_fitter=self.weight_fitter,
            weight_threshold=0.5,  # Higher threshold
        )

        # Add experts with different weights
        experts = [
            WeightedExpert(
                expert_function=CORRECT_EXPERTS[0], weight=0.3
            ),  # Below threshold
            WeightedExpert(
                expert_function=CORRECT_EXPERTS[1], weight=0.7
            ),  # Above threshold
        ]
        manager.add_experts(experts)

        # Prune experts
        manager.prune_experts()

        # Should only have one expert remaining (weight 0.7)
        remaining_experts = manager.get_experts()
        assert len(remaining_experts) == 1
        assert remaining_experts[0].weight == 0.7
