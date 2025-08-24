"""
Factory for creating evaluation contexts for the Crafter environment.
"""

from crafter.state_export import WorldState
from crafter.functional_env import transition, initial_state, EnvConfig
import random

from ..core import EvaluationContext, EvaluationConfig, SymbolicTransition
from .components import JSONPatchEditDistance


class CrafterEvaluationFactory:
    def __init__(self, env_config: EnvConfig, policy_seed: int = 42):
        self.env_config = env_config
        self.policy_seed = policy_seed
        self.policy_rng = random.Random(policy_seed)
        self.initial_state = initial_state(
            area=env_config.size,
            view=env_config.view,
            episode=1,
            seed=policy_seed,
        )
        self.transition_fn = transition

    def create_context(
        self, config: EvaluationConfig, num_transitions: int
    ) -> EvaluationContext[WorldState]:

        test_transitions: list[SymbolicTransition[WorldState]] = []

        return EvaluationContext(
            config=config,
            test_transitions=test_transitions,
            distractor_generator=Semantic1DDistractorGenerator(self.world_config),
            edit_distance_calculator=JSONPatchEditDistance(),
        )
