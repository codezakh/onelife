"""
Environment adapters for the hybrid evaluation framework.

This module provides environment-specific implementations of the injected
component protocols, enabling the core evaluator to work with different
environments.
"""

import random

from ..core import (
    SymbolicEnvironment,
    TrajectoryCollector,
    EditDistanceCalculator,
    DistractorGenerator,
)
from .components import (
    RandomPolicy1DTrajectoryCollector,
    JSONPatchEditDistance,
    Semantic1DDistractorGenerator,
)
from ...poe_world.benchmark_1d.environment import (
    GameState,
    Action,
    WorldConfig,
    transition_function,
    DEFAULT_LAWS,
)


class Environment1DWrapper:
    """Minimal environment wrapper - only transition function."""

    def __init__(self, config: WorldConfig, seed: int):
        self.config = config
        self.base_seed = seed

    def transition(self, state: GameState, action: Action) -> GameState:
        """Apply transition function with deterministic randomness."""
        return transition_function(state, action, DEFAULT_LAWS)


class Environment1DAdapter:
    """Complete adapter for 1D benchmark environment."""

    def __init__(self, config: WorldConfig, seed: int):
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)

    def create_environment(self) -> SymbolicEnvironment[GameState]:
        """Create a 1D environment wrapper."""
        return Environment1DWrapper(self.config, self.seed)

    def create_trajectory_collector(self) -> TrajectoryCollector[GameState]:
        """Create a random policy trajectory collector."""
        return RandomPolicy1DTrajectoryCollector(self.rng)

    def create_edit_distance_calculator(self) -> EditDistanceCalculator[GameState]:
        """Create a JSON patch edit distance calculator."""
        return JSONPatchEditDistance()

    def create_distractor_generator(self) -> DistractorGenerator[GameState]:
        """Create a semantic distractor generator."""
        return Semantic1DDistractorGenerator(self.config)
