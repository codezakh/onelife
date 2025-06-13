import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import numpy as np
from balrog.utils import get_unique_seed
from distant_sunburn.io_utils import PydanticJSONLinesWriter
from distant_sunburn.typing_utils import implements
from pydantic import BaseModel
from tqdm import tqdm
from typing_extensions import Self

from .balrog_components import (
    CrafterEnvironmentConfig,
    environment_factory,
    EnvironmentConfig,
)
from .balrog_interfaces import AgentProtocol, EnvironmentProtocol, MetadataT
from typing import Generic, TypeAlias
from .typing_utils import BaseModelT

logger = logging.getLogger(__name__)


class TrajectoryStep(BaseModel, Generic[MetadataT]):
    step: int
    action: str
    reasoning: Optional[str]
    observation: str
    reward: float
    done: bool
    # NOTE: It is possible that MetadataT contains
    # non-serializable objects, so attempting to blindly
    # serialize it _may_ fail.
    info: MetadataT


TrajectoryStepDictMetadata: TypeAlias = TrajectoryStep[dict]


class TrajectoryStepWriter(Protocol[MetadataT]):
    def __call__(self, step: TrajectoryStep[MetadataT]) -> None:
        pass

    def __enter__(self) -> Any:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


class PydanticTrajectoryStepWriter(Generic[MetadataT]):
    def __init__(self, file_path: str | Path = Path("trajectory_steps.jsonl")):
        self.file_path = file_path
        self.writer = PydanticJSONLinesWriter(file_path)

    def __call__(self, step: TrajectoryStep[MetadataT]) -> None:
        self.writer(step)

    def __enter__(self) -> Self:
        self.writer.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.writer.__exit__(exc_type, exc_value, traceback)


implements(TrajectoryStepWriter)(PydanticTrajectoryStepWriter)


class EvaluatorConfig(BaseModel):
    num_episodes: int
    max_steps_per_episode: Optional[int] = None
    environment_config: CrafterEnvironmentConfig
    output_dir: Path
    feedback_on_invalid_action: bool = True
    save_images: bool = False
    num_workers: int = 1


class Evaluator(Generic[MetadataT]):
    def __init__(
        self,
        config: EvaluatorConfig,
        environment_factory: Callable[
            [EnvironmentConfig], EnvironmentProtocol[MetadataT]
        ],
    ):
        self.config = config
        self.environment_factory = environment_factory

    def run_episode(
        self,
        agent: AgentProtocol,
        process_num=None,
        position=0,
        episode_idx=0,
    ):
        """Run a single evaluation episode.

        Args:
            task (str): Task name.
            agent (Agent): Agent to evaluate.
            process_num (str, optional): Identifier of the process running the episode. Defaults to None.
            position (int, optional): Position index for the progress bar. Defaults to 0.
            episode_idx (int, optional): Index of the episode. Defaults to 0.

        Returns:
            dict: Log of the episode containing statistics and results.
        """
        env = self.environment_factory(self.config.environment_config)
        agent.reset()

        seed = self.config.environment_config.seed
        if seed is None:
            seed = get_unique_seed(process_num=process_num, episode_idx=episode_idx)
        random.seed(seed)
        np.random.seed(seed)
        on_reset_experience = env.reset(seed=seed)
        obs = on_reset_experience.obs
        info = on_reset_experience.info
        episode_log = {
            "task": self.config.environment_config.task,
            "action_frequency": defaultdict(int),
            "input_tokens": 0,
            "output_tokens": 0,
        }

        instructions = None
        agent.prompt_builder.update_instruction_prompt(
            env.get_instruction_prompt(instructions=instructions)
        )

        episode_return = 0.0

        max_steps_per_episode = (
            self.config.environment_config.max_episode_steps
            if self.config.max_steps_per_episode is None
            else self.config.max_steps_per_episode
        )

        trajectory_log_filename = os.path.join(
            self.config.output_dir,
            self.config.environment_config.name,
            self.config.environment_config.task,
            f"{self.config.environment_config.task}_run_{episode_idx:02d}.csv",
        )
        Path(trajectory_log_filename).parent.mkdir(exist_ok=True, parents=True)

        with PydanticTrajectoryStepWriter(
            trajectory_log_filename
        ) as trajectory_step_writer:

            pbar_desc = (
                f"Task: {self.config.environment_config.task}, Proc: {process_num}"
            )
            pbar = tqdm(
                total=max_steps_per_episode,
                desc=pbar_desc,
                position=position,
                leave=False,  # Keep the progress bar after completion
                dynamic_ncols=True,
            )

            action = None
            for step in range(max_steps_per_episode):
                # Agent has an act method that returns an LLMResponse
                response = agent.act(obs, prev_action=action)
                action = env.check_action_validity(response.completion)
                reasoning = response.reasoning if response.reasoning else ""

                episode_log["action_frequency"][action] += 1
                episode_log["input_tokens"] += response.input_tokens
                episode_log["output_tokens"] += response.output_tokens

                experience = env.step(action)
                obs = experience.obs
                reward = experience.reward
                done = experience.done
                info = experience.info

                episode_return += reward  # type: ignore

                # Give feedback on the action (if not valid)
                obs.text.long_term_context = (
                    f"\n\nYour previous output did not contain a valid action. Defaulted to action: {action}\n\nObservation:\n"
                    + obs.text.long_term_context
                    if (action != response.completion)
                    and (self.config.feedback_on_invalid_action)
                    else obs.text.long_term_context
                )
                action = response.completion

                trajectory_step = TrajectoryStep(
                    step=step,
                    action=action,
                    reasoning=reasoning,
                    observation=obs.text.long_term_context
                    + obs.text.short_term_context,
                    reward=float(reward),
                    done=done,
                    info=experience.info,
                )
                trajectory_step_writer(trajectory_step)

                pbar.update(1)

                if self.config.save_images and obs.image:
                    images_dir = os.path.join(
                        self.config.output_dir,
                        self.config.environment_config.name,
                        self.config.environment_config.task,
                        f"episode_{episode_idx:02d}",
                    )
                    Path(images_dir).mkdir(exist_ok=True, parents=True)
                    image_filename = os.path.join(images_dir, f"step_{step:04d}.png")
                    image = obs.image
                    image.save(image_filename)

                if done:
                    logging.info(f"Episode done with reward: {episode_return}")
                    episode_log["done"] = True
                    if pbar.n < pbar.total:
                        pbar.update(pbar.total - pbar.n)
                    pbar.set_postfix_str("DONE")
                    break

            if pbar.n < pbar.total:
                pbar.update(pbar.total - pbar.n)
            if "done" not in episode_log:
                pbar.set_postfix_str("DONE")
            pbar.close()

            episode_log["episode_return"] = episode_return
            episode_log["num_steps"] = step + 1
            episode_log["failed_candidates"] = env.failed_candidates
            episode_log.update(env.get_stats())
            episode_log["process_num"] = process_num
            episode_log["seed"] = seed
            # episode_log["agent"] = OmegaConf.to_container(
            #     self.config.agent, resolve=True
            # )
            # episode_log["client"] = OmegaConf.to_container(
            #     self.config.client, resolve=True
            # )

            # # Save the episode_log to a JSON file
            json_filename = os.path.join(
                self.config.output_dir,
                self.config.environment_config.name,
                self.config.environment_config.task,
                f"{self.config.environment_config.task}_run_{episode_idx:02d}.json",
            )
            Path(json_filename).parent.mkdir(exist_ok=True, parents=True)
            with open(json_filename, "w") as f:
                json.dump(episode_log, f, indent=4)

        return episode_log, trajectory_log_filename
