"""
Core interfaces and classes for the hybrid evaluation framework.

This module defines the core protocols and data structures that enable
environment-agnostic evaluation of symbolic world models.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar
import numpy as np

SymbolicStateT = TypeVar("SymbolicStateT")
SymbolicStateT_contra = TypeVar("SymbolicStateT_contra", contravariant=True)


class EvaluatableWorldModel(Protocol[SymbolicStateT]):
    """Protocol for world models that can be evaluated."""

    def sample_next_state(
        self, current_state: SymbolicStateT, action: Any
    ) -> SymbolicStateT:
        """Generate single prediction by sampling from posterior P(s_next | s, a)"""
        ...

    def evaluate_log_probability(
        self, next_state: SymbolicStateT, current_state: SymbolicStateT, action: Any
    ) -> float:
        """Compute log P(next_state | current_state, action)"""
        ...


class SymbolicTransitionFunction(Protocol[SymbolicStateT]):
    """Minimal protocol for symbolic environments."""

    def __call__(self, state: SymbolicStateT, action: Any) -> SymbolicStateT:
        """True transition function: (s, a) -> s'"""
        ...


class TrajectoryCollector(Protocol[SymbolicStateT]):
    """Protocol for collecting symbolic transitions."""

    def collect_transitions(
        self,
        transition_function: SymbolicTransitionFunction[SymbolicStateT],
        num_transitions: int,
    ) -> list["SymbolicTransition[SymbolicStateT]"]:
        """Collect symbolic transitions using environment-specific policy"""
        ...


class EditDistanceCalculator(Protocol[SymbolicStateT_contra]):
    """Protocol for computing edit distances between states."""

    def __call__(
        self, state1: SymbolicStateT_contra, state2: SymbolicStateT_contra
    ) -> int:
        """Compute structured edit distance between two states"""
        ...


class DistractorGenerator(Protocol[SymbolicStateT]):
    """Protocol for generating plausible but incorrect next states."""

    def __call__(
        self,
        transition: "SymbolicTransition[SymbolicStateT]",
        all_transitions: list["SymbolicTransition[SymbolicStateT]"],
        num_distractors: int,
    ) -> list[SymbolicStateT]:
        """Generate plausible but incorrect next states"""
        ...


@dataclass(frozen=True)
class SymbolicTransition(Generic[SymbolicStateT]):
    """A single symbolic transition."""

    prev_metadata: SymbolicStateT
    action: Any
    next_metadata: SymbolicStateT


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for evaluation runs."""

    num_distractors: int = 5


@dataclass
class EvaluationContext(Generic[SymbolicStateT]):
    """Dependencies for evaluation."""

    test_transitions: list[SymbolicTransition[SymbolicStateT]]
    distractor_generator: DistractorGenerator[SymbolicStateT]
    edit_distance_calculator: EditDistanceCalculator[SymbolicStateT]
    config: EvaluationConfig


@dataclass(frozen=True)
class EvaluationResults:
    """Results from a evaluation run."""

    mean_generative_error: float
    discriminative_accuracy: float
    discriminative_accuracy_by_distractor_type: dict[str, float]
    total_transitions_evaluated: int


class Evaluator(Generic[SymbolicStateT]):
    """
    Measures the performance of a symbolic world model against a sequence of transitions collected offline.
    """

    def __init__(self, context: EvaluationContext[SymbolicStateT]):
        self.ctx = context

    def evaluate(
        self,
        world_model: EvaluatableWorldModel[SymbolicStateT],
    ) -> EvaluationResults:

        generative_errors: list[int | float] = []
        discriminative_successes: list[bool] = []
        distractor_type_results: dict[str, list[bool]] = defaultdict(list)

        for transition in self.ctx.test_transitions:
            # 2. Generate prediction
            pred_state = world_model.sample_next_state(
                transition.prev_metadata, transition.action
            )

            # 3. Measure generative error using injected calculator
            gen_error = self.ctx.edit_distance_calculator(
                pred_state, transition.next_metadata
            )
            generative_errors.append(gen_error)

            # 4. Generate distractors using injected generator
            distractors = self.ctx.distractor_generator(
                transition, self.ctx.test_transitions, self.ctx.config.num_distractors
            )

            # 5. Construct candidate set
            candidates = [transition.next_metadata, pred_state] + distractors

            # 6. Evaluate log probabilities using indices
            log_probs: list[float] = []
            for candidate in candidates:
                log_prob = world_model.evaluate_log_probability(
                    candidate, transition.prev_metadata, transition.action
                )
                log_probs.append(log_prob)

            # 7. Check discriminative success
            # Index 0 is the true next state, index 1 is the predicted state
            max_prob_idx = max(range(len(log_probs)), key=lambda i: log_probs[i])
            discriminative_successes.append(
                max_prob_idx == 0
            )  # True state should have highest probability

        # Convert distractor type results to means
        distractor_type_means = {
            distractor_type: float(np.mean(results))
            for distractor_type, results in distractor_type_results.items()
        }

        return EvaluationResults(
            mean_generative_error=float(np.mean(generative_errors)),
            discriminative_accuracy=float(np.mean(discriminative_successes)),
            discriminative_accuracy_by_distractor_type=distractor_type_means,
            total_transitions_evaluated=len(self.ctx.test_transitions),
        )
