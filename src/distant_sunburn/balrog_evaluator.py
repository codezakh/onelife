import json
import logging
import multiprocessing
import os
import random
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
from balrog.utils import get_unique_seed
from distant_sunburn.io_utils import PydanticJSONLinesWriter
from distant_sunburn.typing_utils import implements
from pydantic import BaseModel
from tqdm import tqdm
from typing_extensions import Self

from .balrog_components import CrafterEnvironmentConfig, environment_factory
from .balrog_interfaces import AgentProtocol

logger = logging.getLogger(__name__)


class TrajectoryStep(BaseModel):
    step: int
    action: str
    reasoning: Optional[str]
    observation: str
    reward: float
    done: bool


class TrajectoryStepWriter(Protocol):
    def __call__(self, step: TrajectoryStep) -> None:
        pass

    def __enter__(self) -> Any:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


class PydanticTrajectoryStepWriter:
    def __init__(self, file_path: str | Path = Path("trajectory_steps.jsonl")):
        self.file_path = file_path
        self.writer = PydanticJSONLinesWriter(file_path)

    def __call__(self, step: TrajectoryStep) -> None:
        self.writer(step)

    def __enter__(self) -> Self:
        self.writer.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.writer.__exit__(exc_type, exc_value, traceback)


implements(TrajectoryStepWriter)(PydanticTrajectoryStepWriter)


class EvaluatorManager:
    """Manages evaluation of agents across multiple environments and tasks.

    The EvaluatorManager initializes evaluators for each specified environment and handles the execution
    of evaluation tasks either sequentially or in parallel using multiple workers.
    """

    def __init__(self, config, output_dir=".", balrog_root: Path = Path(".")):
        """Initialize the EvaluatorManager.

        Args:
            config (omegaconf.DictConfig): Configuration object containing evaluation settings.
            original_cwd (str, optional): Original current working directory. Defaults to "".
            output_dir (str, optional): Directory to save evaluation outputs. Defaults to ".".
        """
        self.config = config
        self.output_dir = output_dir

        self.env_names = config.envs.names.split("-")
        self.env_evaluators = {}
        self.tasks = []
        for env_name in self.env_names:
            evaluator = Evaluator(
                env_name,
                config,
                balrog_root=balrog_root,
                output_dir=self.output_dir,
            )
            self.env_evaluators[env_name] = evaluator
            for task in evaluator.tasks:
                for episode_idx in range(evaluator.num_episodes):
                    # Check if task has been completed
                    json_filename = os.path.join(
                        self.output_dir,
                        env_name,
                        task,
                        f"{task}_run_{episode_idx:02d}.json",
                    )
                    if os.path.exists(json_filename):
                        logging.info(
                            f"Skipping completed task: {env_name}, {task}, episode {episode_idx}"
                        )
                    else:
                        self.tasks.append((env_name, task, episode_idx))
        self.num_workers = config.eval.num_workers

    def run(self, agent_factory):
        """Run the evaluation using the specified agent factory.

        Args:
            agent_factory (AgentFactory): Factory object to create agents for evaluation.

        Returns:
            dict: Results of the evaluation aggregated by environment name.
        """
        if self.num_workers > 1:
            results = self._run_parallel(agent_factory)
        else:
            results = self._run_sequential(agent_factory)
        return results

    def _run_sequential(self, agent_factory):
        """Run the evaluation sequentially.

        Args:
            agent_factory (AgentFactory): Factory object to create agents for evaluation.

        Returns:
            dict: Results of the evaluation aggregated by environment name.
        """
        results = defaultdict(list)
        total_episodes = len(self.tasks)
        with tqdm(total=total_episodes, desc="Evaluating Episodes", position=0) as pbar:
            for env_name, task, episode_idx in self.tasks:
                evaluator = self.env_evaluators[env_name]
                agent = agent_factory.create_agent()
                episode_log = evaluator.run_episode(
                    task, agent, position=1, episode_idx=episode_idx
                )
                results[env_name].append(episode_log)
                pbar.update(1)
        return results

    def _run_parallel(self, agent_factory):
        """Run the evaluation in parallel using multiple workers.

        Args:
            agent_factory (AgentFactory): Factory object to create agents for evaluation.

        Returns:
            dict: Results of the evaluation aggregated by environment name.
        """
        task_queue = multiprocessing.Queue()
        results_queue = multiprocessing.Queue()

        ctx = multiprocessing.get_context("fork")

        # Initially fill the task queue with tasks up to the number of workers
        for item in self.tasks[: self.num_workers]:
            task_queue.put(item)

        # Create a master progress bar
        pbar = tqdm(total=len(self.tasks), position=0, leave=True)

        # Assign unique positions for progress bars
        positions = list(range(self.num_workers))

        processes = []
        for idx in range(self.num_workers):
            position = positions[idx]
            p = ctx.Process(
                target=self._worker,
                args=(task_queue, results_queue, agent_factory, position),
            )
            processes.append(p)
            p.start()

        results = defaultdict(list)
        tasks_completed = 0
        tasks_queued = self.num_workers

        total_tasks = len(self.tasks)

        while tasks_completed < total_tasks:
            result = results_queue.get()
            if "error" in result:
                logging.error(
                    f"Error in task {result['task']} processed by {result['process_num']}: {result['error']}"
                )
                logging.error(f"Traceback:\n{result['traceback']}")
            else:
                results[result["env_name"]].append(result)
            tasks_completed += 1

            # Update progress bar
            pbar.update(1)
            pbar.set_description(
                f"Last task: {result['task']}, Process: {result.get('process_num', 'N/A')}"
            )

            # Queue another task if there are any left
            if tasks_queued < total_tasks:
                task_queue.put(self.tasks[tasks_queued])
                tasks_queued += 1

        # Signal workers to stop
        for _ in range(self.num_workers):
            task_queue.put(None)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Close the master bar when done
        pbar.close()

        return results

    def _worker(self, task_queue, results_queue, agent_factory, position):
        """Worker process for parallel evaluation.

        Args:
            task_queue (multiprocessing.Queue): Queue containing tasks to process.
            results_queue (multiprocessing.Queue): Queue to put the results.
            agent_factory (AgentFactory): Factory object to create agents.
            position (int): Position index for the progress bar.
        """
        seed = get_unique_seed(process_num=position)
        random.seed(seed)
        np.random.seed(seed)

        agent = agent_factory.create_agent()
        process_num = multiprocessing.current_process().name
        while True:
            item = task_queue.get()
            if item is None:
                break
            try:
                env_name, task, episode_idx = item
                evaluator = self.env_evaluators[env_name]
                result = evaluator.run_episode(
                    task,
                    agent,
                    process_num=process_num,
                    position=position + 1,
                    episode_idx=episode_idx,
                )
                result["process_num"] = process_num  # Include process number in result
                result["env_name"] = env_name
                results_queue.put(result)
            except Exception as e:
                tb = traceback.format_exc()
                logging.error(f"Error in worker processing task {task}: {e}\n{tb}")
                results_queue.put(
                    {
                        "env_name": env_name,
                        "task": task,
                        "error": str(e),
                        "traceback": tb,
                        "process_num": process_num,
                    }
                )


class EvaluatorConfig(BaseModel):
    num_episodes: int
    max_steps_per_episode: Optional[int] = None
    environment_config: CrafterEnvironmentConfig
    output_dir: Path
    feedback_on_invalid_action: bool = True
    save_images: bool = False
    num_workers: int = 1


class Evaluator:

    def __init__(
        self,
        config: EvaluatorConfig,
    ):
        self.config = config

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
        env = environment_factory(self.config.environment_config)
        agent.reset()

        seed = self.config.environment_config.seed
        if seed is None:
            seed = get_unique_seed(process_num=process_num, episode_idx=episode_idx)
        random.seed(seed)
        np.random.seed(seed)
        obs, info = env.reset(seed=seed)
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

        return episode_log
