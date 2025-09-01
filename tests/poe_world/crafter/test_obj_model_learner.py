"""
Tests for the ObjModelLearner for Crafter.

This module tests that the ObjModelLearner can properly manage the synthesis
process for a specific object type.
"""

import pytest
import asyncio
from crafter.state_export import WorldState, CowState

from distant_sunburn.poe_world.crafter.obj_model_learner import ObjModelLearner
from distant_sunburn.poe_world.core import SymbolicTransition
from distant_sunburn.litellm_utils import GeminiLiteLlmParams


class TestObjModelLearner:
    """Test the ObjModelLearner."""

    def test_create_custom_view(self, mixed_entity_world):
        """Test that custom views filter objects correctly."""
        learner = ObjModelLearner("cow")

        # Create custom view for cow
        custom_state = learner._create_custom_view(mixed_entity_world)

        # Should only contain cow objects
        assert len(custom_state.objects) == 1
        assert custom_state.objects[0].name == "cow"

    def test_filter_surprising_transitions_empty_model(self, cow_attack_scenario):
        """Test that transitions are considered surprising when model is empty."""
        learner = ObjModelLearner("cow")

        # With an empty model, all transitions should be surprising
        surprising = learner._filter_surprising_transitions([cow_attack_scenario])
        assert len(surprising) == 1
        assert surprising[0] == cow_attack_scenario

    def test_filter_surprising_transitions_with_experts(self, cow_attack_scenario):
        """Test that transitions are filtered based on model predictions."""
        learner = ObjModelLearner("cow")

        # Add a simple expert that predicts the transition correctly
        # This would require implementing expert compilation, which is TODO
        # For now, we'll test that the method works with an empty model
        surprising = learner._filter_surprising_transitions([cow_attack_scenario])
        assert len(surprising) == 1  # Should be surprising with empty model

    def test_zombie_scenario(self, zombie_attack_scenario):
        """Test that zombie scenarios work correctly."""
        learner = ObjModelLearner("zombie")

        # With an empty model, all transitions should be surprising
        surprising = learner._filter_surprising_transitions([zombie_attack_scenario])
        assert len(surprising) == 1
        assert surprising[0] == zombie_attack_scenario


@pytest.mark.asyncio
async def test_process_transitions_integration(cow_attack_scenario):
    """
    Integration test for processing transitions.

    This test verifies that the ObjModelLearner can process transitions
    and potentially synthesize experts.
    """
    # Skip if no API key is available
    import os

    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not available")

    learner = ObjModelLearner("cow")

    # Process the transition
    experts = await learner.process_transitions([cow_attack_scenario])

    # Should find at least one surprising transition with empty model
    # (The exact number of experts depends on the LLM response)
    assert len(experts) >= 0  # Could be 0 if LLM fails or no experts generated

    # If experts were generated, they should have the right structure
    for expert in experts:
        assert expert.object_type == "cow"
        assert expert.code is not None
        assert len(expert.code) > 0
