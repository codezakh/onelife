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
    List,
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

from typing import Sequence, Callable
import inspect

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
        """
        Args:
            support: The set of possible values for the distribution. Always 1D
            logscores: The log-probabilities of the values in the support. Should be the same
                length as support. If None, the logscores are set to 0.0 for all values in the
                support, corresponding to a uniform distribution.
        """
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
        probabilities = np.exp(self.log_probs)
        # Explicitly cast to int to avoid numpy.int64
        return int(np.random.choice(self.support, p=probabilities))

    @property
    def log_probs(self) -> npt.NDArray[np.float32]:
        return self.logscores - logsumexp(self.logscores)

    def evaluate_log_probability(self, value: int) -> float:
        """Calculates the log-probability of a given value."""
        # Cache normalized log probabilities to avoid repeated logsumexp
        try:
            # Find the index of the value and return its log probability
            return float(self.log_probs[np.where(self.support == value)[0][0]])
        except IndexError:
            # The value was not a possible outcome under this distribution
            return -np.inf

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(support={self.support}, logscores={self.logscores})"


MetadataT_contra = TypeVar("MetadataT_contra", contravariant=True)


class ExpertFunction(Protocol[MetadataT_contra]):
    """
    Protocol defining the interface that all expert functions must implement.

    Expert functions are callable objects that take a current state and action,
    then modify the state in-place by assigning RandomValues objects to attributes
    they have opinions about.
    """

    def __call__(self, current_state: MetadataT_contra, action: Any) -> None:
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

    @property
    def __source_code__(self) -> str:
        """The source code of the expert function."""
        ...


class ExpertFunctionWrapper(Generic[MetadataT_contra]):
    def __init__(
        self,
        expert_func: Callable[[MetadataT_contra, Any], None],
        source_code: str,
    ):
        self.expert_func = expert_func
        self.source_code = source_code

    def __call__(self, current_state: MetadataT_contra, action: Any) -> None:
        self.expert_func(current_state, action)

    @classmethod
    def from_non_runtime_created(
        cls, expert_func: Callable[[MetadataT_contra, Any], None]
    ) -> "ExpertFunctionWrapper[MetadataT_contra]":
        return cls(expert_func, inspect.getsource(expert_func))

    @property
    def __source_code__(self) -> str:
        return self.source_code

    @property
    def __name__(self) -> str:
        return self.expert_func.__name__


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
class WeightedExpert(Generic[MetadataT]):
    """An expert function associated with its learned weight."""

    expert_function: ExpertFunction[MetadataT]
    weight: float
    is_fitted: bool = False


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
    ) -> list[WeightedExpert]:
        """
        Fit weights to a set of experts based on a dataset of transitions.

        Args:
            experts: List of expert functions to fit weights for
            transitions: Training data as symbolic transitions

        Returns:
            List of weighted experts with learned weights. The returned list maintains
            the same order as the input experts list - experts[i] corresponds to
            returned_weighted_experts[i]. All returned WeightedExpert instances have
            is_fitted=True to indicate they have been fitted with learned weights.
        """
        ...


SymbolicStateT = TypeVar("SymbolicStateT")
ActionT = TypeVar("ActionT")


ObservableId = NewType("ObservableId", str)


class ObservableExtractorProtocol(Protocol[SymbolicStateT]):
    """
    Protocol for extracting observable attributes from symbolic states.

    This protocol defines the interface for components that can:
    1. Extract probabilistic predictions from states after expert execution
    2. Extract ground truth observed values from states
    3. Apply combined expert predictions to create new states

    The ObservableExtractor is a core component of the PoE-World system that bridges
    between environment-specific state representations and the generic expert prediction
    framework. It handles the conversion between symbolic states and the observable
    attributes that experts can predict.

    Key Requirements:
    - Must provide consistent ObservableId mappings across all methods
    - Must handle both DiscreteDistribution predictions and primitive values
    - Must ensure all observable attributes are covered in all methods
    - Must preserve the structure and integrity of the symbolic state
    """

    def extract_attribute_predictions(
        self, state: SymbolicStateT
    ) -> Dict[ObservableId, DiscreteDistribution]:
        """
        Extract probabilistic predictions from a state after expert execution.

        This method is called after experts have modified a state by assigning
        DiscreteDistribution objects to attributes they have opinions about.
        The method should:
        1. Identify all observable attributes in the state
        2. Extract DiscreteDistribution predictions where experts made them
        3. Create uniform distributions for attributes that experts didn't modify
        4. Ensure all observable attributes are represented in the output

        Args:
            state: The symbolic state after expert execution. May contain both
                   primitive values and DiscreteDistribution objects.

        Returns:
            Dictionary mapping ObservableId to DiscreteDistribution for each
            observable attribute. All DiscreteDistribution objects should have
            the same support (domain) for a given attribute across calls.

        Requirements:
            - Must return the same set of ObservableIds for the same state type
            - Must handle both DiscreteDistribution and primitive value attributes
            - Must create uniform distributions for unmodified attributes
            - Must expand DiscreteDistribution support to full domain if needed
        """
        ...

    def get_observed_outcomes(self, state: SymbolicStateT) -> Dict[ObservableId, int]:
        """
        Extract ground truth observed values from a symbolic state.

        This method extracts the actual observed values from a state for use
        in training and evaluation. It should return the same ObservableIds
        as extract_attribute_predictions but with primitive integer values
        instead of distributions.

        Args:
            state: The symbolic state containing ground truth values.
                   Should contain only primitive values (no DiscreteDistribution).

        Returns:
            Dictionary mapping ObservableId to integer values for each
            observable attribute.

        Requirements:
            - Must return the same ObservableIds as extract_attribute_predictions
            - Must handle boolean values by converting to int (0/1)
            - Must handle all observable attributes present in the state
            - Should be deterministic for the same input state
        """
        ...

    def apply_expert_predictions(
        self,
        new_state: SymbolicStateT,
        expert_predictions: Dict[ObservableId, list[DiscreteDistribution]],
        weights: torch.Tensor,
    ) -> SymbolicStateT:
        """
        Apply combined expert predictions to create a new state.

        This method takes the predictions from multiple experts for each attribute,
        combines them using the provided weights, and applies the results to
        create a new state. It implements the core Product of Experts (PoE)
        combination logic.

        Args:
            new_state: A copy of the current state to be modified. This state
                       should contain primitive values and will be mutated in-place.
            expert_predictions: Dictionary mapping ObservableId to list of
                               DiscreteDistribution predictions from each expert.
                               Each list should have the same length as the weights tensor.
            weights: Tensor of expert weights [n_experts] with dtype=torch.float32.
                     weights[i] determines how much expert i's prediction contributes.

        Returns:
            The modified state with sampled values from combined expert predictions.

        Requirements:
            - Must mutate new_state in-place and return it
            - Must handle all ObservableIds present in expert_predictions
            - Must combine predictions using the provided weights
            - Must sample from combined distributions to get concrete values
            - Must convert sampled values to appropriate types (e.g., bool for boolean attributes)
            - Must preserve state structure and handle missing predictions gracefully
        """
        ...


class ExpertSynthesizerProtocol(Protocol[SymbolicStateT]):
    """
    Protocol for expert synthesizers that can generate new experts from transitions.

    This corresponds to the synthesizer modules in external poe-world that generate
    Python code from observed transitions.
    """

    async def synthesize_experts(
        self, transitions: List[SymbolicTransition[SymbolicStateT]], object_type: str
    ) -> List[WeightedExpert]:
        """Synthesize expert programs from state transitions."""
        ...
