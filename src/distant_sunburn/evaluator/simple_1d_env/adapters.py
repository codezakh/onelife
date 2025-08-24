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
from ...simple_1d_env.environment import (
    GameState,
    WorldConfig,
    default_transition_function,
    initial_state,
)


class Environment1DAdapter:
    """Complete adapter for 1D benchmark environment."""

    def __init__(self, world_config: WorldConfig, policy_seed: int = 42):
        self.world_config = world_config
        self.policy_seed = policy_seed
        self.policy_rng = random.Random(policy_seed)
        self.initial_state = initial_state(self.world_config)

    def create_environment(self) -> SymbolicTransitionFunction[GameState]:
        """Create a 1D environment wrapper."""
        return default_transition_function

    def create_trajectory_collector(self) -> TrajectoryCollector[GameState]:
        """Create a random policy trajectory collector."""
        return RandomPolicy1DTrajectoryCollector(self.policy_rng, self.initial_state)

    def create_edit_distance_calculator(self) -> EditDistanceCalculator[GameState]:
        """Create a JSON patch edit distance calculator."""
        return JSONPatchEditDistance()

    def create_distractor_generator(self) -> DistractorGenerator[GameState]:
        """Create a semantic distractor generator."""
        return Semantic1DDistractorGenerator(self.world_config)
