"""
Baseline world models for testing and validation.

This module provides baseline implementations of world models that can be used
to validate the evaluation framework and establish performance bounds.
"""

import copy
import math
from typing import Any, Generic, TypeVar, Callable
import random

from .core import SymbolicTransitionFunction

SymbolicStateT = TypeVar("SymbolicStateT")
ActionT = TypeVar("ActionT")


class TrueTransitionWorldModel(Generic[SymbolicStateT, ActionT]):
    """Perfect world model using actual transition function."""

    def __init__(
        self,
        environment: SymbolicTransitionFunction[SymbolicStateT, ActionT],
        equal_fn: Callable[[SymbolicStateT, SymbolicStateT], bool],
    ):
        self.environment = environment
        self.equal_fn = equal_fn

    def sample_next_state(
        self, current_state: SymbolicStateT, action: ActionT
    ) -> SymbolicStateT:
        """Use the true transition function."""
        return self.environment(current_state, action)

    def evaluate_log_probability(
        self, next_state: SymbolicStateT, current_state: SymbolicStateT, action: ActionT
    ) -> float:
        """Perfect model: probability 1 for correct transition, 0 otherwise."""
        true_next = self.environment(current_state, action)
        return 0.0 if self._states_equal(next_state, true_next) else -math.inf

    def _states_equal(self, state1: SymbolicStateT, state2: SymbolicStateT) -> bool:
        """Check if two states are equal."""
        return self.equal_fn(state1, state2)


class NullWorldModel(Generic[SymbolicStateT, ActionT]):
    """Baseline model that predicts no state changes."""

    def __init__(self, equal_fn: Callable[[SymbolicStateT, SymbolicStateT], bool]):
        self.equal_fn = equal_fn

    def sample_next_state(
        self, current_state: SymbolicStateT, action: ActionT
    ) -> SymbolicStateT:
        """Always predict no change."""
        return copy.deepcopy(current_state)

    def evaluate_log_probability(
        self, next_state: SymbolicStateT, current_state: SymbolicStateT, action: ActionT
    ) -> float:
        """Give high probability to no change, low to changes."""
        if self._states_equal(next_state, current_state):
            return 0.0  # High probability for no change
        else:
            return -5.0  # Low but not impossible probability for changes

    def _states_equal(self, state1: SymbolicStateT, state2: SymbolicStateT) -> bool:
        """Check if two states are equal."""
        return self.equal_fn(state1, state2)


class RandomWorldModel(Generic[SymbolicStateT, ActionT]):
    """Random baseline model for comparison."""

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def sample_next_state(
        self, current_state: SymbolicStateT, action: ActionT
    ) -> SymbolicStateT:
        """Generate random state."""
        # This is a simplified version - in practice, you'd need to generate
        # valid states for the specific environment
        return copy.deepcopy(current_state)  # Placeholder

    def evaluate_log_probability(
        self, next_state: SymbolicStateT, current_state: SymbolicStateT, action: ActionT
    ) -> float:
        """Return random log probability."""
        return self.rng.uniform(-10.0, 0.0)
