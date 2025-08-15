from distant_sunburn_gameenv.engine.transition_function import (
    transition_function,
    initial_state,
)
from distant_sunburn_gameenv.engine.state import State
from distant_sunburn_gameenv.engine.actions import NoOp
from distant_sunburn_gameenv.text_interface import TextRenderer, ActionParser
from .typing_utils import implements
from .balrog_interfaces import (
    EnvironmentProtocol,
    Experience,
    Observation,
    OnResetExperience,
    Text,
)
from .balrog_components import EnvironmentConfig
from pydantic import BaseModel
from typing import Optional
import textwrap
from typing import Literal
from loguru import logger


INSTRUCTION_PROMPT = textwrap.dedent(
    """
    You are an agent playing a 2D tile-based game.

    Available commands:
    - Movement: move_north, move_south, move_east, move_west
    - Combat: attack_north, attack_south, attack_east, attack_west
    - Other: wait

    To take an action, output the command you want to take and nothing else.

    Movement commands move the player in the specified direction.
    Attack commands target entities in the specified direction from the player.
    Wait command skips the turn and is a no-op.

    In a moment I will present a history of actions and observations from the game.
    Your goals are to experience as much of the mechanics of the game world as possible.
    To do this, you will need to balance exploration and survival.

    PLAY!
    """
).strip()


class DistantSunburnConfig(EnvironmentConfig):
    max_episode_steps: int
    view_size: int
    name: str = "distant_sunburn"
    task: str = "open_ended"
    seed: Optional[int] = None


class LanguageSymbolicWrapper:
    def __init__(self, config: DistantSunburnConfig):
        self.config = config
        self.state = initial_state()
        self.step_count = 0
        self.text_renderer = TextRenderer(config.view_size)
        self.action_parser = ActionParser()
        self.failed_candidates: list[str] = []

    def reset(self, seed: Optional[int] = None) -> OnResetExperience[State]:
        self.state = initial_state()
        self.step_count = 0

        renderered_observation = self.text_renderer.render_observation(self.state)
        observation = Observation(
            text=Text(
                long_term_context=renderered_observation.long_term,
                short_term_context=renderered_observation.short_term,
            ),
        )

        return OnResetExperience(obs=observation, info=self.state)

    def step(self, action: str) -> Experience[State]:
        realized_action, _ = self.action_parser.parse_action(action, self.state)
        assert realized_action is not None

        self.state = transition_function(self.state, realized_action)

        rendered_observation = self.text_renderer.render_observation(self.state)
        observation = Observation(
            text=Text(
                long_term_context=rendered_observation.long_term,
                short_term_context=rendered_observation.short_term,
            ),
        )

        self.step_count += 1

        truncated = self.step_count >= self.config.max_episode_steps
        done = self.state.player.health <= 0

        return Experience(
            obs=observation,
            info=self.state,
            action=action,
            reward=0.0,
            done=done,
            truncated=truncated,
        )

    def get_stats(self) -> dict:
        return {}

    def check_action_validity(self, candidate_action: str) -> str:
        maybe_engine_action, _ = self.action_parser.parse_action(
            candidate_action, self.state
        )
        logger.info(f"Maybe engine action: {maybe_engine_action}")
        if maybe_engine_action is None:
            logger.warning(f"Invalid action: {candidate_action}")
            self.failed_candidates.append(candidate_action)
            return "wait"  # Swallow the invalid action as a no-op.
        else:
            return candidate_action

    def get_instruction_prompt(self, instructions: str | None = None) -> str:
        return INSTRUCTION_PROMPT


implements(EnvironmentProtocol[State])(LanguageSymbolicWrapper)
