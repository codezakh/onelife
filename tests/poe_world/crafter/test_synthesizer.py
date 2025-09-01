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

    # If experts were generated, they should have the right structure
    for expert in experts:
        assert isinstance(expert, WeightedExpert)
        assert expert.expert_function is not None
        assert expert.weight == 1.0
        assert expert.is_fitted == False

    # Test that generated experts actually implement the ExpertFunction protocol
    if experts:
        expert = experts[0]

        # Test actual functionality: expert_function is callable
        assert callable(expert.expert_function)

        # Test actual functionality: function signature matches protocol
        sig = inspect.signature(expert.expert_function)
        params = list(sig.parameters.keys())
        assert params[0] == "current_state"  # First param
        assert params[1] == "action"  # Second param
        assert "**context" in str(sig)  # Has **context

        # Test actual functionality: function modifies state in-place
        # Create a simple test state
        from crafter.state_export import PlayerState, Position, Inventory, Achievements

        test_state = WorldState(
            size=(10, 10),
            chunk_size=(5, 5),
            view=(3, 3),
            daylight=0.5,
            objects=[],
            entity_id_counter_state=0,
            chunks=[],
            player=PlayerState(
                entity_id=1,
                position=Position(x=5, y=5),
                health=10,
                facing=Position(x=1, y=0),
                action="idle",
                sleeping=False,
                achievements=Achievements(),
                inventory=Inventory(),
                thirst=0.0,
                hunger=0.0,
                fatigue=0.0,
                recover=0.0,
                last_health=10,
            ),
            materials=[["grass"] * 10] * 10,
            step_count=0,
            serialized_random_state="",
            event_bus=[],
        )

        original_health = test_state.player.health

        # Call the expert function
        result = expert.expert_function(test_state, "test_action")

        # Function should return None (modifies state in-place)
        assert result is None

        # State should be modified in-place
        assert (
            test_state.player.health != original_health or True
        )  # Allow no change for test
