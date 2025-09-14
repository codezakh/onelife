import itertools
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from crafter.constants import ActionT as CrafterAction
from crafter.state_export import WorldState
from loguru import logger
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from distant_sunburn.evaluator.core import SymbolicTransition
from distant_sunburn.evaluator.crafter.scenarios import (
    CowMovementScenario,
    EatCowScenario,
    PlayerDeathScenario,
    RandomMovementScenario,
    ZombieDefeatScenario,
    run_scenarios,
)
from distant_sunburn.poe_world.core import ObservableId, WeightedExpert
from distant_sunburn.poe_world.core import (
    SymbolicTransition as PoEWorldSymbolicTransition,
)
from distant_sunburn.poe_world.crafter.handwritten_experts import (
    CORRECT_EXPERTS,
)
from distant_sunburn.poe_world.crafter.observable_extractor import (
    ObservableExtractor,
    ObservableExtractorConfig,
)
from distant_sunburn.poe_world.expert_manager import ExpertManager
from distant_sunburn.poe_world.object_model_learner import (
    ObjectModelOrchestrator,
    ObjectModelOrchestratorConfig,
)
from distant_sunburn.poe_world.poe_world_learner import PoEWorldLearner
from distant_sunburn.poe_world.synthesizer import NoOpSynthesizer
from distant_sunburn.poe_world.weight_fitter import MaxLikelihoodWeightFitter


def generate_training_transitions() -> (
    dict[str, list[SymbolicTransition[WorldState, CrafterAction]]]
):
    transitions: dict[str, list[SymbolicTransition[WorldState, CrafterAction]]] = {}
    scenarios = [
        PlayerDeathScenario(),
        CowMovementScenario(),
        EatCowScenario(),
        ZombieDefeatScenario(),
        RandomMovementScenario(max_steps=12),
    ]
    results = run_scenarios(scenarios)
    for result in results:
        if result.goal_test:
            transitions[result.scenario.name] = result.transitions
        else:
            logger.warning(
                f"Scenario {result.scenario.name} failed: {result.goal_test.message}"
            )
    return transitions


extractor_config = ObservableExtractorConfig(
    position_domain=np.arange(0, 10),
    health_domain=np.arange(0, 10),
    entity_types=["cow", "zombie", "skeleton", "plant", "arrow", "fence"],
    entity_count_domain=np.arange(0, 5),
    entity_existence_domain=np.array([0, 1]),
)


def build_poe_world_learner(
    tmp_path: Path,
) -> PoEWorldLearner[WorldState, CrafterAction]:
    extractor = ObservableExtractor(config=extractor_config)
    fitter = MaxLikelihoodWeightFitter(observable_extractor=extractor, max_iterations=3)
    non_creation_mgr = ExpertManager(
        observable_extractor=extractor, weight_fitter=fitter, weight_threshold=0.01
    )
    creation_mgr = ExpertManager(
        observable_extractor=extractor, weight_fitter=fitter, weight_threshold=0.01
    )

    experts = [
        WeightedExpert(expert_function=fn, weight=1.0, is_fitted=False)
        for fn in CORRECT_EXPERTS
    ]
    non_creation_mgr.add_experts(experts)

    orchestrator = ObjectModelOrchestrator(
        object_type="all",
        non_creation_expert_manager=non_creation_mgr,
        creation_expert_manager=creation_mgr,
        non_creation_synthesizer=NoOpSynthesizer(),
        creation_synthesizer=NoOpSynthesizer(),
        config=ObjectModelOrchestratorConfig(
            batch_size=10, save_freq=10, surprise_threshold=-0.5
        ),
        checkpoint_dir=str(tmp_path),
    )
    return PoEWorldLearner(
        object_type_to_orchestrator={"all": orchestrator},
        observable_extractor=extractor,
    )


def test_jumpy_posterior(tmp_path):
    learner = build_poe_world_learner(tmp_path)
    transitions = generate_training_transitions()
    logger.info(f"Generated {len(transitions)} transitions")
    model = learner.synthesize_world_model(
        [
            PoEWorldSymbolicTransition(
                prev_metadata=t.prev_metadata,
                action=t.action,
                next_metadata=t.next_metadata,
            )
            for t in itertools.chain.from_iterable(transitions.values())
        ]
    )

    # Print out the weights for each expert
    for expert in model.experts:
        logger.info(f"Expert {expert.expert_function.__name__}: {expert.weight}")

    player_death_transitions = transitions[PlayerDeathScenario().name]
    samples: list[WorldState] = []
    n_samples = 100
    for _ in range(n_samples):
        samples.append(
            model.sample_next_state(
                player_death_transitions[0].prev_metadata,
                player_death_transitions[0].action,
            )
        )
    logger.info(f"Sampled {len(samples)} states")

    # Now we extract observables from the samples and compute the empirical posterior
    extractor = ObservableExtractor(config=extractor_config)
    observed_outcomes: dict[ObservableId, list[int]] = defaultdict(list)
    for sample in samples:
        outcomes_for_sample = extractor.get_observed_outcomes(sample)
        for k, v in outcomes_for_sample.items():
            observed_outcomes[k].append(v)

    # For each of the observables, we compute the empirical posterior (the observed values)
    console = Console()
    empirical_counts: dict[ObservableId, dict[int, int]] = {}

    # Create panels for each observable's empirical posterior
    panels = []

    for observable_id, values in observed_outcomes.items():
        # Count occurrences of each value
        value_counts = Counter(values)
        empirical_counts[observable_id] = dict(value_counts)

        # Create a table for this observable
        table = Table(
            title=f"Empirical Posterior: {observable_id}",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Value", style="cyan", no_wrap=True)
        table.add_column("Count", justify="right", style="green")
        table.add_column("Percentage", justify="right", style="yellow")
        table.add_column("Bar", style="blue")

        # Sort values for consistent display
        sorted_values = sorted(value_counts.items())
        max_count = max(value_counts.values()) if value_counts else 1

        for value, count in sorted_values:
            percentage = (count / len(values)) * 100
            # Create a simple bar chart using characters
            bar_length = int((count / max_count) * 20)  # Scale to 20 characters max
            bar = "█" * bar_length + "░" * (20 - bar_length)

            table.add_row(str(value), str(count), f"{percentage:.1f}%", bar)

        # Add summary row
        table.add_row("", "", "", "", style="dim")
        table.add_row("Total", str(len(values)), "100.0%", "█" * 20, style="bold")

        panels.append(
            Panel(table, title=f"[bold]{observable_id}[/bold]", border_style="blue")
        )

    # Display all panels in columns
    console.print(
        "\n[bold green]Empirical Posteriors from World Model Samples[/bold green]"
    )
    console.print(
        f"[dim]Based on {n_samples} samples from the trained world model[/dim]\n"
    )

    if panels:
        # Display in columns (2 per row)
        for i in range(0, len(panels), 2):
            row_panels = panels[i : i + 2]
            console.print(Columns(row_panels, equal=True, expand=True))
            if i + 2 < len(panels):
                console.print()  # Add spacing between rows

    # Now, we assert that the model puts a nontrivial weight on the
    # possibility that the player is miraculously healed in the player death
    # scenario.
    player_health_counts = empirical_counts[ObservableId("player_health")]
    player_health_posterior = np.zeros((max(player_health_counts.keys()) + 1))
    for value, count in player_health_counts.items():
        player_health_posterior[value] = count
    player_health_posterior /= player_health_posterior.sum()
    # Now we check that the P(player_health >0) is greater than 0.5
    assert np.sum(player_health_posterior[1:]) > 0.5
