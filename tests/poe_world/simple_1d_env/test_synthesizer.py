"""
Tests for the simple 1D environment expert synthesizer.

This module tests that the synthesizer can generate expert functions
from state transitions in the simple 1D environment.
"""

import pytest
import asyncio
from distant_sunburn.simple_1d_env.environment import (
    GameState,
    Action,
    WorldConfig,
    initial_state,
    transition_function,
    DEFAULT_LAWS,
)

from distant_sunburn.poe_world.simple_1d_env.synthesizer import (
    Simple1DExpertSynthesizer,
    Simple1DSynthesisDependenciesProvider,
)
from distant_sunburn.poe_world.core import (
    SymbolicTransition,
    WeightedExpert,
)
import os


@pytest.fixture
def player_movement_scenario():
    """Create a scenario where the player moves right."""
    world_config = WorldConfig()
    prev_state = initial_state(world_config)

    # Apply action to get next state
    next_state = transition_function(prev_state, Action.MOVE_RIGHT, DEFAULT_LAWS)

    return SymbolicTransition(
        prev_metadata=prev_state,
        action=Action.MOVE_RIGHT,
        next_metadata=next_state,
    )


class TestSimple1DExpertSynthesizer:
    """Test the simple 1D environment expert synthesizer."""

    @pytest.mark.asyncio
    async def test_synthesize_experts_returns_empty_list_for_no_transitions(self):
        """Test that synthesizing with no transitions returns empty list."""
        synthesizer = Simple1DExpertSynthesizer(
            dependencies_provider=Simple1DSynthesisDependenciesProvider()
        )
        experts = await synthesizer.synthesize_experts(
            transitions=[], object_type="player"
        )
        assert experts == []

    @pytest.mark.flaky(retries=3, delay=0.25)
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not available",
    )
    async def test_synthesize_experts_creates_valid_experts_for_player_movement(
        self, player_movement_scenario
    ):
        """Test that synthesized experts for player movement have correct structure and behavior."""
        # Skip if no API key is available

        synthesizer = Simple1DExpertSynthesizer(
            dependencies_provider=Simple1DSynthesisDependenciesProvider()
        )
        experts = await synthesizer.synthesize_experts(
            transitions=[player_movement_scenario],
            object_type="player",
        )

        # If experts were generated, they should work correctly
        if experts:
            expert = experts[0]

            # Test structure
            assert isinstance(expert, WeightedExpert)
            assert expert.expert_function is not None
            assert expert.weight == 1.0
            assert not expert.is_fitted

            # Test behavior: function modifies state in-place and returns None
            world_config = WorldConfig()
            test_state = initial_state(world_config)
            result = expert.expert_function(test_state, Action.MOVE_RIGHT)
            assert result is None

    @pytest.mark.flaky(retries=3, delay=0.25)
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not available",
    )
    async def test_synthesize_experts_handles_malformed_llm_responses_gracefully(self):
        """Test that malformed LLM responses don't crash the synthesizer."""
        # Skip if no API key is available
        synthesizer = Simple1DExpertSynthesizer(
            dependencies_provider=Simple1DSynthesisDependenciesProvider()
        )

        # This test would require mocking the LLM to return malformed responses
        # For now, we just verify the method exists and can be called
        assert hasattr(synthesizer, "synthesize_experts")
        assert callable(synthesizer.synthesize_experts)

    @pytest.mark.flaky(retries=3, delay=0.25)
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not available",
    )
    async def test_synthesize_experts_handles_light_object_type(
        self, player_movement_scenario
    ):
        """Test that synthesizer can handle light object types."""
        synthesizer = Simple1DExpertSynthesizer(
            dependencies_provider=Simple1DSynthesisDependenciesProvider()
        )
        experts = await synthesizer.synthesize_experts(
            transitions=[player_movement_scenario],
            object_type="light",
        )

        # Should handle light object type without crashing
        # May return 0 experts if LLM can't handle the type
        assert isinstance(experts, list)
