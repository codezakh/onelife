"""
Test for the UnsupervisedTextRenderer to visually inspect output.
"""

import pytest
from crafter.functional_env import initial_state
from crafter.state_export import Position
from distant_sunburn.unsupervised_crafter_env_factory import UnsupervisedTextRenderer


def test_unsupervised_text_renderer():
    """Test the unsupervised text renderer with a generated world state."""

    world_state = initial_state(
        seed=42,
    )

    renderer = UnsupervisedTextRenderer()

    output = renderer(world_state)

    print("=" * 80)
    print("UNSUPERVISED TEXT RENDERER OUTPUT")
    print("=" * 80)
    print("\nLONG TERM CONTEXT:")
    print("-" * 40)
    print(output.long_term_context)
    print("\nSHORT TERM CONTEXT:")
    print("-" * 40)
    print(output.short_term_context)
    print("=" * 80)

    assert output.long_term_context is not None
    assert output.short_term_context is not None
    assert len(output.long_term_context) > 0
    assert len(output.short_term_context) > 0

    assert "Local view" in output.long_term_context
    assert "Distant view" in output.long_term_context
    assert "You are targeting" in output.long_term_context
    assert "Your status:" in output.short_term_context
    assert "Your inventory:" in output.short_term_context
