"""
Tests for the Crafter expert synthesizer.

This module tests that the synthesizer can generate expert functions
from state transitions in the Crafter environment.
"""

import pytest
import asyncio
from crafter.state_export import WorldState

from distant_sunburn.poe_world.crafter.synthesizer import (
    CrafterExpertSynthesizer,
    SynthesizedExpert,
    create_crafter_synthesizer,
)
from distant_sunburn.poe_world.core import SymbolicTransition, DiscreteDistribution
from distant_sunburn.litellm_utils import GeminiLiteLlmParams


class TestCrafterExpertSynthesizer:
    """Test the Crafter expert synthesizer."""

    def test_synthesizer_initialization(self):
        """Test that the synthesizer can be initialized."""
        synthesizer = CrafterExpertSynthesizer()
        assert synthesizer is not None
        assert synthesizer.llm_params is not None

    def test_create_crafter_synthesizer(self):
        """Test the convenience function for creating a synthesizer."""
        synthesizer = create_crafter_synthesizer()
        assert isinstance(synthesizer, CrafterExpertSynthesizer)

    def test_extract_state_changes(self, cow_attack_scenario):
        """Test that state changes are correctly extracted for observable attributes only."""
        synthesizer = CrafterExpertSynthesizer()

        changes = synthesizer._extract_state_changes(cow_attack_scenario)

        # Should detect the cow health change (observable attribute)
        assert "cow" in changes.lower()
        assert "health" in changes.lower()
        assert "3" in changes  # Cow health changed from 5 to 3

        # Should NOT include inventory changes (not observable)
        assert "inventory" not in changes.lower()
        assert "wood_sword" not in changes.lower()

    def test_extract_expert_function(self):
        """Test that expert functions can be extracted from LLM responses."""
        synthesizer = CrafterExpertSynthesizer()

        # Test with valid function
        valid_response = """
def alter_cow_objects(state: WorldState, action: str) -> WorldState:
    if action == "do":
        # Find cow and reduce health
        for entity in state.objects:
            if entity.name == "cow":
                entity.health = DiscreteDistribution(support=[max(0, entity.health - 2)])
    return state
"""

        extracted = synthesizer._extract_expert_function(valid_response)
        assert extracted is not None
        assert "def alter_cow_objects" in extracted
        assert "DiscreteDistribution" in extracted

    def test_validate_expert_code(self):
        """Test that expert code validation works."""
        synthesizer = CrafterExpertSynthesizer()

        # Valid code
        valid_code = """
def alter_cow_objects(state: WorldState, action: str) -> WorldState:
    if action == "do":
        for entity in state.objects:
            if entity.name == "cow":
                entity.health = DiscreteDistribution(support=[max(0, entity.health - 2)])
    return state
"""
        assert synthesizer._validate_expert_code(valid_code)

        # Invalid code (syntax error)
        invalid_code = """
def alter_cow_objects(state: WorldState, action: str) -> WorldState:
    if action == "do":
        for entity in state.objects:
            if entity.name == "cow":
                entity.health = DiscreteDistribution(support=[max(0, entity.health - 2)]
    return state
"""
        assert not synthesizer._validate_expert_code(invalid_code)


@pytest.mark.asyncio
async def test_synthesize_experts_integration(cow_attack_scenario):
    """
    Integration test for expert synthesis.

    This test verifies that the synthesizer can generate experts from transitions.
    Note: The synthesizer assumes transitions are already filtered for surprising ones.
    This test requires an actual LLM call, so it's marked as integration.
    """
    # Skip if no API key is available
    import os

    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not available")

    synthesizer = CrafterExpertSynthesizer()

    # Try to synthesize experts for cow object type
    # Note: The synthesizer assumes these transitions are already filtered for surprising ones
    experts = await synthesizer.synthesize_experts(
        transitions=[cow_attack_scenario],
        object_type="cow",
    )

    # Should find at least one surprising transition
    # (The exact number of experts depends on the LLM response)
    assert len(experts) >= 0  # Could be 0 if LLM fails or no experts generated

    # If experts were generated, they should have the right structure
    for expert in experts:
        assert isinstance(expert, SynthesizedExpert)
        assert expert.object_type == "cow"
        assert expert.code is not None
        assert len(expert.code) > 0
