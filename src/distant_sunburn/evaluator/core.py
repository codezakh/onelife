"""
Core interfaces and classes for the hybrid evaluation framework.

This module defines the core protocols and data structures that enable
environment-agnostic evaluation of symbolic world models.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar
import numpy as np
import random
import copy
from loguru import logger
from typing_extensions import Self
from typing import TypeAlias, Mapping, Sequence
from typing_extensions import assert_never
import itertools
from typing import Iterable, Literal, Any

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
    def mean(cls: type[Self], edit_distances: Iterable[Self]) -> Self:
        return cls(
            raw=float(np.mean([ed.raw for ed in edit_distances])),
            normalized=float(np.mean([ed.normalized for ed in edit_distances])),
            total_elements=float(np.mean([ed.total_elements for ed in edit_distances])),
            intersection_over_union=float(
                np.mean([ed.intersection_over_union for ed in edit_distances])
            ),
        )

    @classmethod
    def std(cls: type[Self], edit_distances: Iterable[Self]) -> Self:
        return cls(
            raw=float(np.std([ed.raw for ed in edit_distances])),
            normalized=float(np.std([ed.normalized for ed in edit_distances])),
            total_elements=float(np.std([ed.total_elements for ed in edit_distances])),
            intersection_over_union=float(
                np.std([ed.intersection_over_union for ed in edit_distances])
            ),
        )

    @classmethod
    def min(cls: type[Self], edit_distances: Iterable[Self]) -> Self:
        return cls(
            raw=float(np.min([ed.raw for ed in edit_distances])),
            normalized=float(np.min([ed.normalized for ed in edit_distances])),
            total_elements=float(np.min([ed.total_elements for ed in edit_distances])),
            intersection_over_union=float(
                np.min([ed.intersection_over_union for ed in edit_distances])
            ),
        )

    @classmethod
    def max(cls: type[Self], edit_distances: Iterable[Self]) -> Self:
        return cls(
            raw=float(np.max([ed.raw for ed in edit_distances])),
            normalized=float(np.max([ed.normalized for ed in edit_distances])),
            total_elements=float(np.max([ed.total_elements for ed in edit_distances])),
            intersection_over_union=float(
                np.max([ed.intersection_over_union for ed in edit_distances])
            ),
        )


@dataclass(frozen=True)
class EvaluationMetrics:
    edit_distance: EditDistance
    discriminative_accuracy: float
    normalized_recall: float
    n_distractors: float


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
    num_trials: int = 1


TransitionSource: TypeAlias = str


@dataclass
class EvaluationContext(Generic[SymbolicStateT, ActionT_contra]):
    """Dependencies for evaluation."""

    test_transitions: Mapping[
        TransitionSource, Sequence[SymbolicTransition[SymbolicStateT, ActionT_contra]]
    ]
    distractor_generator: DistractorGenerator[SymbolicStateT, ActionT_contra]
    edit_distance_calculator: EditDistanceCalculator[SymbolicStateT]
    config: EvaluationConfig


@dataclass(frozen=True)
class EvaluationResults:
    """Results from a evaluation run."""

    edit_distance: EditDistance
    discriminative_accuracy: float
    normalized_recall: float
    total_transitions_evaluated: int
    metrics_by_source: Mapping[
        TransitionSource, Mapping[Literal["mean", "std"], EvaluationMetrics]
    ]


class Evaluator(Generic[SymbolicStateT, ActionT]):
    """
    Measures the performance of a symbolic world model against a sequence of transitions collected offline.
    """

    def __init__(self, context: EvaluationContext[SymbolicStateT, ActionT]):
        self.ctx = context

    def _evaluate_transition(
        self,
        world_model: EvaluatableWorldModel[SymbolicStateT, ActionT],
        transition: SymbolicTransition[SymbolicStateT, ActionT],
        all_transitions: list[SymbolicTransition[SymbolicStateT, ActionT]],
    ) -> EvaluationMetrics:

        # TODO: This function does a _bunch_ of deepcopies. This is because
        # we can't trust that the world model and the methods the world model
        # uses are not mutating the input states. This is technically also true for the distractor
        # generator, except that is human-written code so we can confirm that it
        # is not mutating the input state.
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

        # 4. Generate distractors using injected generator
        distractors = self.ctx.distractor_generator(
            transition=transition,
            all_transitions=all_transitions,
            num_distractors=self.ctx.config.num_distractors,
        )

        # 5. Construct candidate set and whether they are the true next state
        candidates = [(transition.next_metadata, True), (pred_state, False)] + [
            (distractor, False) for distractor in distractors
        ]

        # Shuffle the candidates
        random.shuffle(candidates)

        n_distractors = len(distractors) + 1

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
                discriminative_accuracy = 1.0
            case (False, True):
                # The max probability candidate chosen by the model is equivalent
                # to the true next state! This can happen in the case where the
                # model perfectly predicts the true next state, or where one of the
                # distractors is equivalent to the true next state (though this is rare)
                discriminative_accuracy = 1.0
            case (False, False):
                # The model predicted a next state that was not the true next state
                # and the chosen candidate is not equivalent to the true next state
                # Therefore, this is a prediction error
                discriminative_accuracy = 0.0
            case _:
                assert_never(_)

        # Calculate the recall of the true next state. This is the
        # rank at which the true next state appears in the candidate
        # set if we order the candidates by log probability (highest to lowest)
        # We must not penalize the model if it assigns higher probability
        # to any candidate that is equivalent to the true next state.
        # To handle this, we form the set of all candidates equal to the
        # true next state and use the best (smallest) rank among them.
        ordered_indices = sorted(
            range(len(log_probs)), key=lambda i: log_probs[i], reverse=True
        )
        # Map candidate index -> 1-based rank
        rank_by_index = {idx: rank for rank, idx in enumerate(ordered_indices, start=1)}
        # Find all candidates equivalent to the true next state
        equivalent_indices = [
            i
            for i, (candidate, _) in enumerate(candidates)
            if candidate == transition.next_metadata
        ]
        # There should always be at least one (the true next state),
        # but guard defensively and raise if none found.
        if equivalent_indices:
            best_rank = min(rank_by_index[i] for i in equivalent_indices)
            # Normalize to [0, 1]: 1.0 if top-ranked, 0.0 if last
            max_rank = len(candidates)
            if max_rank > 1:
                normalized_recall = 1.0 - (best_rank - 1) / (max_rank - 1)
            else:
                normalized_recall = 1.0
        else:
            # This should be impossible because the true next state is explicitly
            # included in the candidates list. If it happens, it indicates a serious
            # bug or state corruption. Log details and raise.
            logger.error(
                "True next state was not found among candidates; this likely indicates"
                " that the __eq__ method is not implemented correctly.",
                num_candidates=len(candidates),
            )
            raise RuntimeError("True next state missing from candidates")

        return EvaluationMetrics(
            edit_distance=edit_distance,
            discriminative_accuracy=discriminative_accuracy,
            normalized_recall=normalized_recall,
            n_distractors=n_distractors,
        )

    def _aggregate_metrics(
        self,
        iterable: Iterable[EvaluationMetrics],
        reduction: Literal["mean", "std", "min", "max"],
    ) -> EvaluationMetrics:
        edit_distances: list[EditDistance] = []
        discriminative_accuracies: list[float] = []
        normalized_recalls: list[float] = []
        n_distractors: list[float] = []

        for metrics in iterable:
            edit_distances.append(metrics.edit_distance)
            discriminative_accuracies.append(metrics.discriminative_accuracy)
            normalized_recalls.append(metrics.normalized_recall)
            n_distractors.append(metrics.n_distractors)

        match reduction:
            case "mean":
                return EvaluationMetrics(
                    EditDistance.mean(edit_distances),
                    float(np.mean(discriminative_accuracies)),
                    float(np.mean(normalized_recalls)),
                    float(np.mean(n_distractors)),
                )
            case "std":
                return EvaluationMetrics(
                    EditDistance.std(edit_distances),
                    float(np.std(discriminative_accuracies)),
                    float(np.std(normalized_recalls)),
                    float(np.std(n_distractors)),
                )
            case "min":
                return EvaluationMetrics(
                    EditDistance.min(edit_distances),
                    float(np.min(discriminative_accuracies)),
                    float(np.min(normalized_recalls)),
                    int(np.min(n_distractors)),
                )
            case "max":
                return EvaluationMetrics(
                    EditDistance.max(edit_distances),
                    float(np.max(discriminative_accuracies)),
                    float(np.max(normalized_recalls)),
                    int(np.max(n_distractors)),
                )
            case _:
                assert_never(reduction)

    def _log_distractor_instrumentation(
        self,
        min_n_distractors: Any,
        max_n_distractors: Any,
        mean_n_distractors: Any,
        std_n_distractors: Any,
    ):
        logger.info(
            "Distractor Stats",
            distractor_stats={
                "min": min_n_distractors,
                "max": max_n_distractors,
                "mean": mean_n_distractors,
                "std": std_n_distractors,
            },
        )

    def evaluate(
        self,
        world_model: EvaluatableWorldModel[SymbolicStateT, ActionT],
    ) -> EvaluationResults:

        edit_distances: list[EditDistance] = []

        unaggregated_metrics_by_source: dict[
            TransitionSource, list[EvaluationMetrics]
        ] = defaultdict(list)

        all_transitions = list(
            itertools.chain.from_iterable(self.ctx.test_transitions.values())
        )

        for transition_source, transitions in self.ctx.test_transitions.items():
            # We handle multiple trials by concatenating the transitions
            # num_trial times, so we get num_trials * len(transitions) transitions
            # effectively running each transition source num_trials times

            transitions_iterator = itertools.chain.from_iterable(
                itertools.repeat(transitions, self.ctx.config.num_trials)
            )

            for transition in transitions_iterator:
                evaluation_metrics = self._evaluate_transition(
                    world_model, transition, all_transitions
                )
                edit_distances.append(evaluation_metrics.edit_distance)
                unaggregated_metrics_by_source[transition_source].append(
                    evaluation_metrics
                )

        mean_metrics = self._aggregate_metrics(
            itertools.chain.from_iterable(unaggregated_metrics_by_source.values()),
            reduction="mean",
        )

        std_metrics = self._aggregate_metrics(
            itertools.chain.from_iterable(unaggregated_metrics_by_source.values()),
            reduction="std",
        )

        min_metrics = self._aggregate_metrics(
            itertools.chain.from_iterable(unaggregated_metrics_by_source.values()),
            reduction="min",
        )

        max_metrics = self._aggregate_metrics(
            itertools.chain.from_iterable(unaggregated_metrics_by_source.values()),
            reduction="max",
        )

        self._log_distractor_instrumentation(
            min_metrics.n_distractors,
            max_metrics.n_distractors,
            mean_metrics.n_distractors,
            std_metrics.n_distractors,
        )

        aggregated_metrics_by_source: dict[
            TransitionSource, dict[Literal["mean", "std"], EvaluationMetrics]
        ] = {
            transition_source: {
                reduction: self._aggregate_metrics(
                    metrics_for_transition_source, reduction=reduction
                )
                for reduction in ("mean", "std")
            }
            for transition_source, metrics_for_transition_source in unaggregated_metrics_by_source.items()
        }

        return EvaluationResults(
            edit_distance=mean_metrics.edit_distance,
            discriminative_accuracy=mean_metrics.discriminative_accuracy,
            normalized_recall=mean_metrics.normalized_recall,
            total_transitions_evaluated=len(all_transitions),
            metrics_by_source=aggregated_metrics_by_source,
        )
