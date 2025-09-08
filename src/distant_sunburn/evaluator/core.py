"""
Core interfaces and classes for the hybrid evaluation framework.

This module defines the core protocols and data structures that enable
environment-agnostic evaluation of symbolic world models.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar
import numpy as np
from icecream import ic
import random
import copy
from loguru import logger
from typing_extensions import Self

SymbolicStateT = TypeVar("SymbolicStateT")
SymbolicStateT_contra = TypeVar("SymbolicStateT_contra", contravariant=True)
ActionT_contra = TypeVar("ActionT_contra", contravariant=True)
ActionT = TypeVar("ActionT")
ActionT_co = TypeVar("ActionT_co", covariant=True)


class EvaluatableWorldModel(Protocol[SymbolicStateT, ActionT_contra]):
    """Protocol for world models that can be evaluated."""

    def sample_next_state(
        self, current_state: SymbolicStateT, action: ActionT_contra
    ) -> SymbolicStateT:
        """Generate single prediction by sampling from posterior P(s_next | s, a)"""
        ...

    def evaluate_log_probability(
        self,
        state: SymbolicStateT,
        action: ActionT_contra,
        next_state: SymbolicStateT,
    ) -> float:
        """Compute log P(next_state | current_state, action)"""
        ...


class SymbolicTransitionFunction(Protocol[SymbolicStateT, ActionT_contra]):
    """Minimal protocol for symbolic environments."""

    def __call__(self, state: SymbolicStateT, action: ActionT_contra) -> SymbolicStateT:
        """True transition function: (s, a) -> s'"""
        ...


class TrajectoryCollector(Protocol[SymbolicStateT, ActionT]):
    """Protocol for collecting symbolic transitions."""

    def collect_transitions(
        self,
        transition_function: SymbolicTransitionFunction[SymbolicStateT, ActionT],
        num_transitions: int,
    ) -> list["SymbolicTransition[SymbolicStateT, ActionT]"]:
        """Collect symbolic transitions using environment-specific policy"""
        ...


@dataclass(frozen=True)
class EditDistance:
    raw: float
    normalized: float
    total_elements: float
    intersection_over_union: float

    @classmethod
    def reduce(cls: type[Self], edit_distances: list[Self]) -> Self:
        return cls(
            raw=float(np.mean([ed.raw for ed in edit_distances])),
            normalized=float(np.mean([ed.normalized for ed in edit_distances])),
            total_elements=float(np.mean([ed.total_elements for ed in edit_distances])),
            intersection_over_union=float(
                np.mean([ed.intersection_over_union for ed in edit_distances])
            ),
        )


class EditDistanceCalculator(Protocol[SymbolicStateT_contra]):
    """Protocol for computing edit distances between states."""

    def __call__(
        self,
        state: SymbolicStateT_contra,
        true_next_state: SymbolicStateT_contra,
        pred_next_state: SymbolicStateT_contra,
    ) -> EditDistance:
        """Compute structured edit distance between two states"""
        ...


class DistractorGenerator(Protocol[SymbolicStateT, ActionT]):
    """Protocol for generating plausible but incorrect next states."""

    def __call__(
        self,
        transition: "SymbolicTransition[SymbolicStateT, ActionT]",
        all_transitions: list["SymbolicTransition[SymbolicStateT, ActionT]"],
        num_distractors: int,
    ) -> list[SymbolicStateT]:
        """Generate plausible but incorrect next states"""
        ...


@dataclass(frozen=True)
class SymbolicTransition(Generic[SymbolicStateT, ActionT_co]):
    """A single symbolic transition."""

    prev_metadata: SymbolicStateT
    action: ActionT_co
    next_metadata: SymbolicStateT


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for evaluation runs."""

    num_distractors: int = 5


@dataclass
class EvaluationContext(Generic[SymbolicStateT, ActionT_contra]):
    """Dependencies for evaluation."""

    test_transitions: list[SymbolicTransition[SymbolicStateT, ActionT_contra]]
    distractor_generator: DistractorGenerator[SymbolicStateT, ActionT_contra]
    edit_distance_calculator: EditDistanceCalculator[SymbolicStateT]
    config: EvaluationConfig


@dataclass(frozen=True)
class EvaluationResults:
    """Results from a evaluation run."""

    edit_distance: EditDistance
    discriminative_accuracy: float
    discriminative_accuracy_by_distractor_type: dict[str, float]
    total_transitions_evaluated: int


class Evaluator(Generic[SymbolicStateT, ActionT]):
    """
    Measures the performance of a symbolic world model against a sequence of transitions collected offline.
    """

    def __init__(self, context: EvaluationContext[SymbolicStateT, ActionT]):
        self.ctx = context

    def evaluate(
        self,
        world_model: EvaluatableWorldModel[SymbolicStateT, ActionT],
    ) -> EvaluationResults:

        edit_distances: list[EditDistance] = []
        discriminative_successes: list[bool] = []
        distractor_type_results: dict[str, list[bool]] = defaultdict(list)

        # TODO: This function does a _bunch_ of deepcopies. This is because
        # we can't trust that the world model and the methods the world model
        # uses are not mutating the input states. This is technically also true for the distractor
        # generator, except that is human-written code so we can confirm that it
        # is not mutating the input state.

        n_distractors: list[int] = []

        for idx, transition in enumerate(self.ctx.test_transitions):
            # 2. Generate prediction
            pred_state = world_model.sample_next_state(
                copy.deepcopy(transition.prev_metadata), transition.action
            )

            # 3. Measure generative error using injected calculator
            edit_distance = self.ctx.edit_distance_calculator(
                state=transition.prev_metadata,
                true_next_state=transition.next_metadata,
                pred_next_state=pred_state,
            )
            edit_distances.append(edit_distance)

            # 4. Generate distractors using injected generator
            distractors = self.ctx.distractor_generator(
                transition, self.ctx.test_transitions, self.ctx.config.num_distractors
            )

            # 5. Construct candidate set and whether they are the true next state
            candidates = [(transition.next_metadata, True), (pred_state, False)] + [
                (distractor, False) for distractor in distractors
            ]

            # Shuffle the candidates
            random.shuffle(candidates)

            n_distractors.append(len(distractors) + 1)

            # 6. Evaluate log probabilities using indices
            log_probs: list[float] = []
            for candidate, _ in candidates:
                log_prob = world_model.evaluate_log_probability(
                    state=copy.deepcopy(transition.prev_metadata),
                    action=transition.action,
                    next_state=copy.deepcopy(candidate),
                )
                log_probs.append(log_prob)

            # 7. Check discriminative success
            max_prob_idx = max(range(len(log_probs)), key=lambda i: log_probs[i])
            # The true next state should have the highest probability
            # In the case where the true and predicted states
            # are the same, we could wrongly penalize the model for picking
            # the predicted state instead of the true state.
            max_prob_candidate, chose_true_next_state = candidates[max_prob_idx]
            pred_next_state_eq_true_next_state = (
                max_prob_candidate == transition.next_metadata
            )
            match (chose_true_next_state, pred_next_state_eq_true_next_state):
                case (True, _):
                    # The model correctly chose the true next step,
                    # so nothing else matters and we can mark as successful
                    discriminative_successes.append(True)
                case (False, True):
                    # The max probability candidate chosen by the model is equivalent
                    # to the true next state! This can happen in the case where the
                    # model perfectly predicts the true next state, or where one of the
                    # distractors is equivalent to the true next state (though this is rare)
                    discriminative_successes.append(True)
                case (False, False):
                    # The model predicted a next state that was not the true next state
                    # and the chosen candidate is not equivalent to the true next state
                    # Therefore, this is a prediction error
                    discriminative_successes.append(False)

        # Convert distractor type results to means
        distractor_type_means = {
            distractor_type: float(np.mean(results))
            for distractor_type, results in distractor_type_results.items()
        }

        logger.info(
            "Distractor Stats",
            distractor_stats={
                "mean": float(np.mean(n_distractors)),
                "max": float(np.max(n_distractors)),
                "min": float(np.min(n_distractors)),
                "std": float(np.std(n_distractors)),
            },
        )

        logger.info(
            "Edit Distance Stats/Raw",
            edit_distance_stats={
                "mean": float(np.mean([ed.raw for ed in edit_distances])),
                "max": float(np.max([ed.raw for ed in edit_distances])),
                "min": float(np.min([ed.raw for ed in edit_distances])),
                "std": float(np.std([ed.raw for ed in edit_distances])),
            },
        )

        logger.info(
            "Edit Distance Stats/Intersection Over Union",
            edit_distance_stats={
                "mean": float(
                    np.mean([ed.intersection_over_union for ed in edit_distances])
                ),
                "max": float(
                    np.max([ed.intersection_over_union for ed in edit_distances])
                ),
                "min": float(
                    np.min([ed.intersection_over_union for ed in edit_distances])
                ),
                "std": float(
                    np.std([ed.intersection_over_union for ed in edit_distances])
                ),
            },
        )

        return EvaluationResults(
            edit_distance=EditDistance.reduce(edit_distances),
            discriminative_accuracy=float(np.mean(discriminative_successes)),
            discriminative_accuracy_by_distractor_type=distractor_type_means,
            total_transitions_evaluated=len(self.ctx.test_transitions),
        )
