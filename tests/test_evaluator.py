from distant_sunburn.balrog_evaluator import (
    EvaluatorManager,
    EvaluatorConfig,
    Evaluator,
)
from omegaconf import DictConfig
from pathlib import Path
from balrog.agents import AgentFactory
import pytest
from distant_sunburn import REPO_ROOT
from balrog.utils import collect_and_summarize_results, print_summary_table
from distant_sunburn.balrog_client import (
    make_llm_client_factory,
    LlmClientConfig,
    GenerateKwargs,
)
from distant_sunburn.balrog_components import (
    HistoryPromptBuilderConfig,
    HistoryPromptBuilder,
    NaiveAgent,
    CrafterEnvironmentConfig,
)


BALROG_ROOT = REPO_ROOT / "external" / "BALROG"


@pytest.mark.skip(reason="skip")
def test_evaluator(tmp_path: Path):

    output_dir = str(tmp_path)

    config = DictConfig(
        {
            "agent": {
                "type": "naive",
                "remember_cot": True,
                "max_text_history": 16,
                "max_image_history": 0,
                "max_cot_history": 1,
                "max_icl_history": 1000,
                "cache_icl": False,
            },
            "eval": {
                "output_dir": output_dir,
                "resume_from": None,
                "num_workers": 1,
                "num_episodes": {
                    "crafter": 1,
                },
                "max_steps_per_episode": None,
                "save_trajectories": True,
                "save_images": False,
                "icl_episodes": 1,
                "icl_dataset": "records",
                "feedback_on_invalid_action": True,
            },
            "client": {
                "client_name": "gemini",
                "model_id": "gemini-2.0-flash",
                "base_url": "http://localhost:8080/v1",
                "generate_kwargs": {"temperature": 1.0, "max_tokens": 4096},
                "timeout": 60,
                "max_retries": 5,
                "delay": 2,
                "alternate_roles": False,
            },
            "envs": {
                "names": "crafter",
                "env_kwargs": {"seed": None},
                "crafter_kwargs": {
                    "area": [64, 64],
                    "view": [9, 9],
                    "size": [256, 256],
                    "reward": True,
                    "seed": None,
                    "max_episode_steps": 16,
                },
            },
            "tasks": {
                "crafter_tasks": ["default"],
            },
        }
    )

    agent_factory = AgentFactory(config)
    evaluator_manager = EvaluatorManager(
        config, output_dir=output_dir, balrog_root=BALROG_ROOT
    )
    results = evaluator_manager.run(agent_factory)
    summary = collect_and_summarize_results(output_dir)
    print_summary_table(summary)


def test_evaluator_custom(tmp_path: Path):
    client_config = LlmClientConfig(
        client_name="gemini",
        model_id="gemini-2.0-flash",
        base_url="http://localhost:8080/v1",
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

    crafter_config = CrafterEnvironmentConfig(
        area=(64, 64),
        view=(9, 9),
        size=(256, 256),
        reward=True,
        seed=None,
        max_episode_steps=16,
        name="crafter",
    )

    evaluator_config = EvaluatorConfig(
        num_episodes=1,
        environment_config=crafter_config,
        output_dir=tmp_path,
        feedback_on_invalid_action=True,
    )

    prompt_builder_factory = HistoryPromptBuilder.as_factory(prompt_builder_config)

    client_factory = make_llm_client_factory(client_config)

    naive_agent_factory = NaiveAgent.as_factory(client_factory, prompt_builder_factory)

    prompt_builder = prompt_builder_factory()
    naive_agent = naive_agent_factory()
    evaluator = Evaluator(config=evaluator_config)
    episode_log = evaluator.run_episode(naive_agent)
