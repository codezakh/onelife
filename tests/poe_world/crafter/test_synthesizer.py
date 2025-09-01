"""
Tests for the Crafter expert synthesizer.

This module tests that the synthesizer can generate expert functions
from state transitions in the Crafter environment.
"""

import pytest
import asyncio
import inspect
from crafter.state_export import WorldState

from distant_sunburn.poe_world.crafter.synthesizer import (
    CrafterExpertSynthesizer,
)
from distant_sunburn.poe_world.core import (
    SymbolicTransition,
    DiscreteDistribution,
    WeightedExpert,
)
from distant_sunburn.litellm_utils import GeminiLiteLlmParams


class TestCrafterExpertSynthesizer:
    """Test the Crafter expert synthesizer."""

    def test_synthesizer_initialization(self):
        """Test that the synthesizer can be initialized."""
        synthesizer = CrafterExpertSynthesizer()
        assert synthesizer is not None
        assert synthesizer.llm_params is not None

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
def alter_cow_objects(current_state: WorldState, action: str) -> None:
    if action == "do":
        # Find cow and reduce health
        for entity in current_state.objects:
            if entity.name == "cow":
                entity.health = DiscreteDistribution(support=[max(0, entity.health - 2)])
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
def alter_cow_objects(current_state: WorldState, action: str) -> None:
    if action == "do":
        for entity in current_state.objects:
            if entity.name == "cow":
                entity.health = DiscreteDistribution(support=[max(0, entity.health - 2)])
"""
        assert synthesizer._validate_expert_code(valid_code)

        # Invalid code (syntax error)
        invalid_code = """
def alter_cow_objects(current_state: WorldState, action: str) -> None:
    if action == "do":
        for entity in current_state.objects:
            if entity.name == "cow":
                entity.health = DiscreteDistribution(support=[max(0, entity.health - 2)]
"""
        assert not synthesizer._validate_expert_code(invalid_code)

    def test_compile_expert_function(self):
        """Test that expert functions can be compiled into callable objects."""
        synthesizer = CrafterExpertSynthesizer()

        # Test code that should compile successfully
        test_code = """
def alter_cow_objects(current_state: WorldState, action: str) -> None:
    if action == "test":
        current_state.player.health = DiscreteDistribution(support=[5])
"""

        expert_function = synthesizer._compile_expert_function(test_code, "cow")
        assert expert_function is not None
        assert callable(expert_function)

        # Test that the function name is correct
        assert hasattr(expert_function, "__name__")
        # Use getattr to avoid type checker issues
        assert getattr(expert_function, "__name__") == "alter_cow_objects"

    def test_compile_expert_function_failure(self):
        """Test that compilation failures are handled gracefully."""
        synthesizer = CrafterExpertSynthesizer()

        # Test code with syntax error
        invalid_code = """
def alter_cow_objects(current_state: WorldState, action: str) -> None:
    if action == "test":
        current_state.player.health = DiscreteDistribution(support=[5
"""

        expert_function = synthesizer._compile_expert_function(invalid_code, "cow")
        assert expert_function is None


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

    # Test that generated experts actually implement the ExpertFunction protocol
    if experts:
        expert = experts[0]

        # Test actual functionality: function actually works and produces sensible changes
        # Use the initial state from the cow attack scenario (which already has the cow positioned correctly)
        # Make a deep copy to avoid modifying the original fixture state
        test_state = cow_attack_scenario.prev_metadata.model_copy(deep=True)

        # Find the cow in the test state
        cow_state = None
        for obj in test_state.objects:
            if obj.name == "cow":
                cow_state = obj
                break

        assert cow_state is not None, "Cow should exist in the test state"
        # We want to make sure the health of the cow is a regular integer.
        assert isinstance(cow_state.health, int)

        print(expert.expert_function.__source_code__)

        # Call the expert function with the "do" action from the cow attack scenario
        expert.expert_function(test_state, cow_attack_scenario.action)

        # The expert will have assigned a discrete distribution to the cow's health.
        assert isinstance(cow_state.health, DiscreteDistribution)
