from .balrog_interfaces import (
    EnvironmentProtocol,
    Experience,
    OnResetExperience,
    Text,
    Observation,
)
from .balrog_components import CrafterEnvironmentConfig
from .balrog_utilities import get_base_env
from .typing_utils import implements
from crafter.env import Env as CrafterEnv
from crafter.state_export import WorldState, export_world_state


class LanguageSymbolicCrafter:
    """Environment that provides both language and symbolic observations from Crafter."""

    def __init__(self, config: CrafterEnvironmentConfig):
        self.language_env = config.create_balrog_env()
        # Get the base CrafterEnv to use for symbolic observations
        self.crafter_base_env = get_base_env(
            self.language_env, expected_type=CrafterEnv
        )

    def reset(self, **kwargs) -> OnResetExperience[WorldState]:
        """Reset the environment and return both language and symbolic observations."""
        obs, info = self.language_env.reset(**kwargs)

        assert self.crafter_base_env._step is not None
        assert self.crafter_base_env._view is not None

        # Get symbolic observation
        world_state = export_world_state(
            self.crafter_base_env._world,
            tuple(self.crafter_base_env._view),
            self.crafter_base_env._step,
        )

        # Create language observation
        language_obs = Observation(
            text=Text(
                short_term_context=obs["text"]["short_term_context"],
                long_term_context=obs["text"]["long_term_context"],
            ),
            image=obs.get("image", None),
            obs=obs,
        )

        return OnResetExperience(
            obs=language_obs,
            info=world_state,
        )

    def step(self, action: str) -> Experience[WorldState]:
        """Execute action and return both language and symbolic observations."""
        obs, reward, terminated, truncated, info = self.language_env.step(action)

        assert self.crafter_base_env._step is not None
        assert self.crafter_base_env._view is not None

        # Get symbolic observation
        world_state = export_world_state(
            self.crafter_base_env._world,
            tuple(self.crafter_base_env._view),
            self.crafter_base_env._step,
        )

        # Create language observation
        language_obs = Observation(
            text=Text(
                short_term_context=obs["text"]["short_term_context"],
                long_term_context=obs["text"]["long_term_context"],
            ),
            image=obs.get("image", None),
            obs=obs,
        )

        return Experience(
            obs=language_obs,
            action=action,
            reward=float(reward),
            done=terminated,
            truncated=truncated,
            info=world_state,
        )

    def get_instruction_prompt(self, instructions: str | None = None) -> str:
        return self.language_env.get_instruction_prompt(instructions)

    def check_action_validity(self, candidate_action: str) -> str:
        return self.language_env.check_action_validity(candidate_action)

    @property
    def failed_candidates(self) -> list[str]:
        return self.language_env.failed_candidates

    def get_stats(self) -> dict:
        return self.language_env.get_stats()


implements(EnvironmentProtocol[WorldState])(LanguageSymbolicCrafter)
