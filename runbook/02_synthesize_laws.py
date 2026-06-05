"""Runbook step 2: synthesize symbolic laws from an exploration trajectory.

Reads the trajectory produced by step 1, regroups it into transitions, runs the
change detectors to find transitions worth examining, renders a prompt for each
changed aspect, and asks an LLM to propose laws (as icon code) for it. The
induced laws are written as JSON-lines, one file per (transition, aspect).

Input:   a trajectory file (JSON-lines of TrajectoryStep[WorldState]) from step 1.
Output:  induced laws under <output_dir>/laws/*.jsonl, plus rendered prompts and
         token-usage stats. The laws directory is the handoff to the next step.

Requires GEMINI_API_KEY in the environment.

Run from the repository root:

    uv run --env-file .env python runbook/02_synthesize_laws.py --debug
"""

import argparse
import asyncio
import random
from pathlib import Path

from crafter_oo.state_reconstruction import (
    ArrowState,
    CowState,
    FenceState,
    PlantState,
    SkeletonState,
    WorldState,
    ZombieState,
)
from loguru import logger
from onelife.balrog_evaluator import TrajectoryStep
from onelife.law_synthesis.crafter.change_detection import (
    ChangeDetector,
    EntityTypeChangeDetector,
    MapTilesChangeDetector,
    PlayerHealthChangeDetector,
    PlayerInventoryChangeDetector,
    PlayerPositionChangeDetector,
    ZombieHealthChangeDetector,
)
from onelife.law_synthesis.crafter.rendering import PromptRenderer
from onelife.law_synthesis.crafter.selection import (
    filter_cow_damage_transitions,
    filter_player_health_damage_transitions,
    filter_player_move_transitions,
    filter_zombie_damage_transitions,
    select_additional_interesting_transitions,
)
from onelife.law_synthesis.crafter.transition import Transition
from onelife.law_synthesis.synthesis import (
    SynthesisTask,
    execute_synthesis_tasks,
    execute_synthesis_tasks_sync,
)
from onelife.law_synthesis.token_counting import TokenTracker
from onelife.pipeline_utils import PydanticJsonLinesFileTarget
from tqdm.asyncio import tqdm


# Create change detectors for each entity type
CHANGE_DETECTORS: list[ChangeDetector] = [
    PlayerInventoryChangeDetector(),
    PlayerPositionChangeDetector(),
    PlayerHealthChangeDetector(),  # Specific detector for player health changes
    MapTilesChangeDetector(),
    EntityTypeChangeDetector("zombies", ZombieState),
    ZombieHealthChangeDetector(),  # Specific detector for zombie health changes
    EntityTypeChangeDetector("cows", CowState),
    EntityTypeChangeDetector("skeletons", SkeletonState),
    EntityTypeChangeDetector("arrows", ArrowState),
    EntityTypeChangeDetector("plants", PlantState),
    EntityTypeChangeDetector("fences", FenceState),
]

# Global token tracker for the experiment
token_tracker = TokenTracker()


def regroup_trajectory_steps(
    trajectory_steps: list[TrajectoryStep[WorldState]],
) -> list[Transition]:
    transitions: list[Transition] = []
    for i in range(1, len(trajectory_steps)):
        transitions.append(
            Transition(
                state=trajectory_steps[i - 1].info,
                action=trajectory_steps[i].action,
                next_state=trajectory_steps[i].info,
                reward=trajectory_steps[i].reward,
            )
        )
    return transitions


async def main_async(
    trajectory_path: Path,
    output_dir: Path,
    debug_mode: bool = False,
    debug_zombie_damage: bool = False,
    debug_cow_damage: bool = False,
    debug_player_move: bool = False,
    debug_player_health: bool = False,
    only_interesting: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_steps_target = PydanticJsonLinesFileTarget(
        file_path=trajectory_path,
        model=TrajectoryStep[WorldState],
    )
    trajectory_steps = trajectory_steps_target.load()
    print(f"Loaded {len(trajectory_steps)} trajectory steps")
    transitions = regroup_trajectory_steps(trajectory_steps)
    if debug_zombie_damage:
        # In debug zombie damage mode, filter to only transitions where zombie health decreases
        zombie_damage_transitions = filter_zombie_damage_transitions(transitions)

        if not zombie_damage_transitions:
            logger.warning(
                "No zombie damage transitions found for debug zombie damage mode"
            )
            return

        interesting_transitions = zombie_damage_transitions
        uninteresting_transitions = []
        logger.info(
            f"Debug zombie damage mode: using {len(zombie_damage_transitions)} zombie damage transitions"
        )
    elif debug_cow_damage:
        # In debug cow damage mode, filter to only transitions where cow health decreases
        cow_damage_transitions = filter_cow_damage_transitions(transitions)

        if not cow_damage_transitions:
            logger.warning("No cow damage transitions found for debug cow damage mode")
            return

        interesting_transitions = cow_damage_transitions
        uninteresting_transitions = []
        logger.info(
            f"Debug cow damage mode: using {len(cow_damage_transitions)} cow damage transitions"
        )
    elif debug_player_move:
        # In debug player move mode, filter to only transitions where player took move action and position changed
        player_move_transitions = filter_player_move_transitions(transitions)

        if not player_move_transitions:
            logger.warning(
                "No player move transitions found for debug player move mode"
            )
            return

        interesting_transitions = player_move_transitions
        uninteresting_transitions = []
        logger.info(
            f"Debug player move mode: using {len(player_move_transitions)} player move transitions"
        )
    elif debug_player_health:
        # In debug player health mode, filter to only transitions where player health decreases
        player_health_damage_transitions = filter_player_health_damage_transitions(
            transitions
        )

        if not player_health_damage_transitions:
            logger.warning(
                "No player health damage transitions found for debug player health mode"
            )
            return

        interesting_transitions = player_health_damage_transitions
        uninteresting_transitions = []
        logger.info(
            f"Debug player health mode: using {len(player_health_damage_transitions)} player health damage transitions"
        )
    elif only_interesting:
        # Filter to only interesting transitions (reward > 0)
        interesting_transitions = [
            t for t in transitions if t.reward is not None and t.reward > 0
        ]

        if not interesting_transitions:
            logger.warning("No interesting transitions found")
            return

        uninteresting_transitions = []
        logger.info(
            f"Only interesting mode: using {len(interesting_transitions)} interesting transitions"
        )

    elif debug_mode:
        # In debug mode, use only the first interesting transition
        interesting_transitions = []
        uninteresting_transitions = []
        for transition in transitions:
            if transition.reward is not None and transition.reward > 0:
                interesting_transitions = [transition]
                break

        if not interesting_transitions:
            logger.warning("No interesting transitions found for debug mode")
            return

        logger.info("Debug mode: using only 1 interesting transition")
    else:
        interesting_transition_indices = [
            i
            for i, t in enumerate(transitions)
            if t.reward is not None and t.reward > 0
        ]
        logger.info(
            f"Number of interesting transitions: {len(interesting_transition_indices)}"
        )

        # Also sample ~60 or so "uninteresting" transitions
        # to help us find more laws
        uninteresting_transition_indices = [
            i
            for i, t in enumerate(transitions)
            if t.reward is not None and t.reward == 0
        ]
        logger.info(
            f"Number of uninteresting transitions: {len(uninteresting_transition_indices)}"
        )

        uninteresting_transitions = [
            transitions[i] for i in uninteresting_transition_indices
        ]
        interesting_transitions = [
            transitions[i] for i in interesting_transition_indices
        ]

        # Select additional interesting transitions to ensure we have examples of all attribute changes
        additional_interesting_indices = select_additional_interesting_transitions(
            transitions, set(interesting_transition_indices)
        )

        if additional_interesting_indices:
            logger.info(
                f"Adding {len(additional_interesting_indices)} additional transitions to ensure coverage of all entity attribute changes"
            )
            for idx in additional_interesting_indices:
                interesting_transitions.append(transitions[idx])
                interesting_transition_indices.append(idx)

        # Sample 60 uninteresting transitions (with fixed seed for reproducibility)
        random.seed(42)  # Fixed seed for reproducible sampling
        # uninteresting_transitions = random.sample(
        #     uninteresting_transitions, min(60, len(uninteresting_transitions))
        # )
        uninteresting_transitions = []

    prompts_dir = output_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    print(f"Loaded {len(transitions)} transitions")

    # Create synthesis tasks
    synthesis_tasks: list[SynthesisTask] = []

    prompt_renderer = PromptRenderer()

    # Process each transition with change detection
    for transition_idx, transition in enumerate(
        tqdm(
            interesting_transitions + uninteresting_transitions,
            desc="Creating synthesis tasks",
        )
    ):
        logger.info(f"Processing transition {transition_idx}")

        # Check which aspects of the state changed
        changed_aspects: list[str] = []
        for detector in CHANGE_DETECTORS:
            if detector.has_changes(transition):
                changed_aspects.append(detector.aspect)
                logger.info(f"Detected changes in aspect: {detector.aspect}")

        # Create a synthesis task for each changed aspect
        for aspect in changed_aspects:
            prompt = prompt_renderer.render_prompt(transition, aspect)

            # Save prompt for debugging
            prompt_filename = f"transition_{transition_idx}_aspect_{aspect}.md"
            with open(prompts_dir / prompt_filename, "w") as f:
                f.write(prompt)

            # Create synthesis task
            task = SynthesisTask(
                transition_idx=transition_idx,
                aspect=aspect,
                prompt=prompt,
                output_dir=output_dir,
            )
            synthesis_tasks.append(task)

    logger.info(f"Created {len(synthesis_tasks)} synthesis tasks")

    # Execute synthesis tasks based on debug mode
    if debug_mode or debug_zombie_damage or debug_player_health:
        execute_synthesis_tasks_sync(synthesis_tasks, token_tracker)
    else:
        # Execute all synthesis tasks asynchronously
        await execute_synthesis_tasks(synthesis_tasks, token_tracker, max_concurrent=5)

    # Gather all laws from all tasks (both newly completed and previously completed)
    induced_laws = SynthesisTask.gather_all_laws(synthesis_tasks)

    logger.info(f"Completed synthesis of {len(induced_laws)} total laws")

    # Log and save final token usage statistics
    token_tracker.log_final_stats()
    token_tracker.save_final_stats(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Law induction experiment")
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path(
            "runbook_output/01_exploration/crafter/open_ended/open_ended_run_00.jsonl"
        ),
        help="Path to the exploration trajectory produced by step 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runbook_output/02_synthesis"),
        help="Directory to write induced laws, prompts, and token stats under.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode: use only one transition and execute synchronously",
    )
    parser.add_argument(
        "--debug-zombie-damage",
        action="store_true",
        help="Run in debug zombie damage mode: filter to only transitions where zombie health decreases",
    )
    parser.add_argument(
        "--debug-cow-damage",
        action="store_true",
        help="Run in debug cow damage mode: filter to only transitions where cow health decreases",
    )
    parser.add_argument(
        "--debug-player-move",
        action="store_true",
        help="Run in debug player move mode: filter to only transitions where player took a move action and position changed",
    )
    parser.add_argument(
        "--debug-player-health",
        action="store_true",
        help="Run in debug player health mode: filter to only transitions where player health decreases",
    )
    parser.add_argument(
        "--only-interesting",
        action="store_true",
        help="Filter to only interesting transitions (reward > 0)",
    )

    args = parser.parse_args()

    # Run the async main function with debug flags
    asyncio.run(
        main_async(
            trajectory_path=args.trajectory,
            output_dir=args.output_dir,
            debug_mode=args.debug,
            debug_zombie_damage=args.debug_zombie_damage,
            debug_cow_damage=args.debug_cow_damage,
            debug_player_move=args.debug_player_move,
            debug_player_health=args.debug_player_health,
            only_interesting=args.only_interesting,
        )
    )


if __name__ == "__main__":
    main()
