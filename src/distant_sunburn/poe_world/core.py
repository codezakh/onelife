"""
Core interfaces and data structures for PoE-World.

This module contains the essential protocols and data structures that are shared
across the PoE-World system, including the RandomValues class for probabilistic
predictions and the ExpertFunction protocol.
"""

from typing import (
    Any,
    Dict,
    Generic,
    NewType,
    Optional,
    Protocol,
    TypeVar,
)

import attrs
import numpy as np
import numpy.typing as npt
import torch
from scipy.special import logsumexp

from typing import Sequence

# Type variable for the metadata type used by different environments
MetadataT = TypeVar("MetadataT")


class DiscreteDistribution:
    """
    Represents a discrete probability distribution over a set of integer or boolean values.

    This is the core mechanism for interpreting deterministic expert outputs
    as probabilistic predictions. Expert functions create "sharp" distributions
    by specifying only the values they believe are possible. These are then
    expanded via noise addition to cover all possible values in the domain,
    with the expert's preferred values having much higher log-probabilities
    than the rest.
    """

    def __init__(
        self,
        support: npt.NDArray[np.int32] | Sequence[int],
        logscores: Optional[npt.NDArray[np.float32] | Sequence[float]] = None,
    ):
        self.support = np.array(support)
        # Assign uniform logscores if not provided
        self.logscores = (
            np.array(logscores)
            if logscores is not None
            else np.zeros_like(support, dtype=np.float32)
        )

    @classmethod
    def from_uniform(cls, support: npt.NDArray[np.int32]) -> "DiscreteDistribution":
        return cls(support=support, logscores=np.zeros_like(support, dtype=np.float32))

    def expand_support(
        self, new_support: npt.NDArray[np.int32], noise_logscore: float = -10.0
    ) -> "DiscreteDistribution":
        """
        Expands the support of the distribution to include all values in the new support.

        The logscores for the new values are set to the noise_logscore.
        Expert functions often only predict a subset of possible values for an attribute
        (e.g., putting all probability on a single value by specifying a single value with
        a logscore of 0.0).
        This function expands such partial distributions to cover the full domain by
        assigning a low probability (noise_logscore) to values the expert didn't predict.

        This is necessary for proper combination of expert predictions, as all experts
        must have distributions over the same set of possible values to be combined
        via weighted averaging.
        """
        new_logscores = np.full_like(new_support, noise_logscore, dtype=np.float32)
        for i, val in enumerate(self.support):
            if val in new_support:
                idx = np.where(new_support == val)[0][0]
                new_logscores[idx] = self.logscores[i]
        return DiscreteDistribution(support=new_support, logscores=new_logscores)

    def sample(self) -> int:
        """Samples a value from the distribution."""
        probabilities = np.exp(self.logscores - logsumexp(self.logscores))
        return np.random.choice(self.support, p=probabilities)

    def evaluate_log_probability(self, value: int) -> float:
        """Calculates the log-probability of a given value."""
        # Cache normalized log probabilities to avoid repeated logsumexp
        if not hasattr(self, "_cached_log_probs"):
            self._cached_log_probs = self.logscores - logsumexp(self.logscores)

        try:
            # Find the index of the value and return its log probability
            return float(self._cached_log_probs[np.where(self.support == value)[0][0]])
        except IndexError:
            # The value was not a possible outcome under this distribution
            return -np.inf

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(values={self.support}, logscores={self.logscores})"


class ExpertFunction(Protocol[MetadataT]):
    """
    Protocol defining the interface that all expert functions must implement.

    Expert functions are callable objects that take a current state and action,
    then modify the state in-place by assigning RandomValues objects to attributes
    they have opinions about.
    """

    def __call__(self, current_state: MetadataT, action: Any, **context: Any) -> None:
        """
        Execute this expert's logic on the current state.

        Args:
            current_state: The symbolic state to modify (mutated in-place)
            action: The action being taken
            **context: Additional context (e.g., touch_side, touch_percent)

        Note:
            This function should modify current_state in-place by assigning
            RandomValues objects to attributes that the expert has an opinion about.
            Attributes not modified are assumed to have uniform distributions.
        """
        ...


@attrs.define(frozen=True)
class SymbolicTransition(Generic[MetadataT]):
    """
    Represents a single transition at the symbolic level: (s_t, a_t, s_{t+1}).
    This is the fundamental unit of data for learning and evaluation.
    """

    prev_metadata: MetadataT
    action: Any
    next_metadata: MetadataT


@attrs.define(frozen=True)
class WeightedExpert:
    """An expert function associated with its learned weight."""

    expert_function: Any  # ExpertFunction - avoiding generic issue
    weight: float


class WorldModelProtocol(Protocol[MetadataT]):
    """
    Represents the complete, learned symbolic world model. Operates purely on
    symbolic states (MetadataT), not raw observations.
    """

    def sample_next_state(self, current_state: MetadataT, action: Any) -> MetadataT: ...
    def evaluate_log_probability(
        self, state: MetadataT, action: Any, next_state: MetadataT
    ) -> float: ...
    def with_new_experts(
        self, new_experts: list[WeightedExpert]
    ) -> "WorldModelProtocol[MetadataT]": ...
    @property
    def experts(self) -> list[WeightedExpert]: ...


class WeightFitterProtocol(Protocol[MetadataT]):
    """
    Fits weights to a set of experts based on a dataset of transitions.
    """

    def fit(
        self,
        experts: list[ExpertFunction[MetadataT]],
        transitions: list[SymbolicTransition[MetadataT]],
    ) -> list[WeightedExpert]: ...


SymbolicStateT = TypeVar("SymbolicStateT")
ActionT = TypeVar("ActionT")


ObservableId = NewType("ObservableId", str)


class ObservableExtractorProtocol(Protocol[SymbolicStateT]):
    def extract_attribute_predictions(
        self, state: SymbolicStateT
    ) -> Dict[ObservableId, DiscreteDistribution]: ...

    def get_observed_outcomes(
        self, state: SymbolicStateT
    ) -> Dict[ObservableId, int]: ...

    def apply_expert_predictions(
        self,
        new_state: SymbolicStateT,
        expert_predictions: Dict[ObservableId, list[DiscreteDistribution]],
        weights: torch.Tensor,
    ) -> SymbolicStateT: ...
