"""Runbook step 1: generate an unguided exploration trajectory.

This is the front of the OneLife pipeline. An LLM agent is dropped into Crafter
with no task or goal and left to explore. Every step it takes is recorded, and
the resulting trajectory is what every downstream step consumes.

Inputs:  none (configuration only; requires GEMINI_API_KEY in the environment).
Output:  a single trajectory file (JSON-lines, one TrajectoryStep[WorldState]
         per line). The path is returned by ``generate_exploration_trajectory``
         and printed when this script is run directly. Hand that path to the
         next step (law synthesis).

Run from the repository root:

    uv run --env-file .env python runbook/01_generate_exploration_trajectory.py
"""

import argparse
from pathlib import Path

from loguru import logger

from onelife.balrog_client import (
    GenerateKwargs,
    LlmClientConfig,
    make_llm_client_factory,
)
from onelife.balrog_components import (
    HistoryPromptBuilder,
    HistoryPromptBuilderConfig,
    NaiveAgent,
)
from onelife.balrog_evaluator import Evaluator, EvaluatorConfig
from onelife.unsupervised_crafter_env_factory import (
    LanguageSymbolicWrapper,
    UnsupervisedCrafterEnvironmentConfig,
)


def generate_exploration_trajectory(
    output_dir: Path,
    llm_config: LlmClientConfig,
    prompt_builder_config: HistoryPromptBuilderConfig,
    env_config: UnsupervisedCrafterEnvironmentConfig,
    num_episodes: int = 1,
    feedback_on_invalid_action: bool = True,
) -> Path:
    """Run one unguided exploration episode and write its trajectory to disk.

    Args:
        output_dir: Directory to write the trajectory under. The file lands at
            ``output_dir / <name> / <task> / <task>_run_00.jsonl``.
        llm_config: Which model drives the agent, and how it is sampled.
        prompt_builder_config: How much history the agent is shown each step.
        env_config: Crafter world dimensions, episode length, and seed.
        num_episodes: Carried into EvaluatorConfig; one episode is run.
        feedback_on_invalid_action: Whether the agent is told when it emits an
            invalid action.

    Returns:
        The path of the trajectory file that was written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator_config = EvaluatorConfig(
        num_episodes=num_episodes,
        environment_config=env_config,
        output_dir=output_dir,
        feedback_on_invalid_action=feedback_on_invalid_action,
    )

    client_factory = make_llm_client_factory(llm_config)
    prompt_builder_factory = HistoryPromptBuilder.as_factory(prompt_builder_config)
    agent = NaiveAgent.as_factory(client_factory, prompt_builder_factory)()

    evaluator = Evaluator(
        config=evaluator_config,
        environment_factory=lambda _: LanguageSymbolicWrapper(env_config),
    )

    # Name the output .jsonl rather than letting the evaluator default to .csv:
    # the file is JSON-lines, so the extension should say so.
    trajectory_path = (
        output_dir / env_config.name / env_config.task / f"{env_config.task}_run_00.jsonl"
    )

    _, trajectory_path = evaluator.run_episode(
        agent, trajectory_log_filename=trajectory_path
    )
    return Path(trajectory_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a short episode (10 steps) so the script can be smoke-tested quickly.",
    )
    args = parser.parse_args()

    output_dir = Path("runbook_output/01_exploration")

    llm_config = LlmClientConfig(
        client_name="gemini",
        model_id="gemini-2.5-flash",
        # base_url is unused by the Gemini client (it calls the hosted API,
        # authenticated by GEMINI_API_KEY); LlmClientConfig requires the field.
        base_url="",
        generate_kwargs=GenerateKwargs(temperature=1.0, max_tokens=4096),
        timeout=60,
        max_retries=5,
        delay=2,
        alternate_roles=False,
    )

    prompt_builder_config = HistoryPromptBuilderConfig(
        max_text_history=16,
        max_image_history=0,
        max_cot_history=1,
    )

    env_config = UnsupervisedCrafterEnvironmentConfig(
        area=(64, 64),
        view=(9, 9),
        size=(256, 256),
        reward=True,
        seed=None,
        max_episode_steps=10 if args.debug else 2000,
        name="crafter",
    )

    trajectory_path = generate_exploration_trajectory(
        output_dir=output_dir,
        llm_config=llm_config,
        prompt_builder_config=prompt_builder_config,
        env_config=env_config,
    )

    logger.info(f"Wrote exploration trajectory to {trajectory_path}")
