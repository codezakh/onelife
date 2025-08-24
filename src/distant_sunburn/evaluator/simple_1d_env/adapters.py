"""
Environment adapters for the hybrid evaluation framework.

This module provides environment-specific implementations of the injected
component protocols, enabling the core evaluator to work with different
environments.
"""

import random

from ..core import (
    SymbolicTransitionFunction,
    TrajectoryCollector,
    EditDistanceCalculator,
    DistractorGenerator,
)
from .components import (
    RandomPolicy1DTrajectoryCollector,
    Semantic1DDistractorGenerator,
    JSONPatchEditDistance,
)
from ...poe_world.benchmark_1d.environment import (
    GameState,
    WorldConfig,
    default_transition_function,
)


class Environment1DAdapter:
    """Complete adapter for 1D benchmark environment."""

    def __init__(self, config: WorldConfig, seed: int):
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)

    def create_environment(self) -> SymbolicTransitionFunction[GameState]:
        """Create a 1D environment wrapper."""
        return default_transition_function

    def create_trajectory_collector(self) -> TrajectoryCollector[GameState]:
        """Create a random policy trajectory collector."""
        return RandomPolicy1DTrajectoryCollector(self.rng)

    def create_edit_distance_calculator(self) -> EditDistanceCalculator[GameState]:
        """Create a JSON patch edit distance calculator."""
        return JSONPatchEditDistance()

    def create_distractor_generator(self) -> DistractorGenerator[GameState]:
        """Create a semantic distractor generator."""
        return Semantic1DDistractorGenerator(self.config)
