"""
Factory for creating evaluation contexts for the 1D benchmark environment.
"""

import random

from ..core import EvaluationContext, EvaluationConfig
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
    Action,
)


class OneDEvaluationFactory:
    """Builds a complete evaluation context for the 1D environment."""

    def __init__(self, world_config: WorldConfig, policy_seed: int = 42):
        self.world_config = world_config
        self.policy_seed = policy_seed
        self.policy_rng = random.Random(policy_seed)
        self.initial_state = initial_state(self.world_config)
        self.environment = default_transition_function

    def create_context(
        self, config: EvaluationConfig, num_transitions: int
    ) -> EvaluationContext[GameState, Action]:
        """Creates a fully configured evaluation context."""

        collector = RandomPolicy1DTrajectoryCollector(
            self.policy_rng, self.initial_state
        )

        test_transitions = collector.collect_transitions(
            self.environment, num_transitions
        )

        return EvaluationContext(
            config=config,
            test_transitions={"random": test_transitions},
            distractor_generator=Semantic1DDistractorGenerator(self.world_config),
            edit_distance_calculator=JSONPatchEditDistance(),
        )
