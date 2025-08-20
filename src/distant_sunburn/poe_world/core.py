"""
Core interfaces and data structures for PoE-World.

This module contains the essential protocols and data structures that are shared
across the PoE-World system, including the RandomValues class for probabilistic
predictions and the ExpertFunction protocol.
"""

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp
from typing import Protocol, Any, TypeVar, Optional
from typing_extensions import Self

# Type variable for the metadata type used by different environments
MetadataT = TypeVar("MetadataT")


class RandomValues:
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
        values: npt.NDArray[np.int32],
        logscores: Optional[npt.NDArray[np.float32]] = None,
    ):
        self.values = values
        self.logscores = logscores or np.zeros_like(values, dtype=float)

    def sample(self) -> int:
        """Samples a value from the distribution."""
        probabilities = np.exp(self.logscores - logsumexp(self.logscores))
        return np.random.choice(self.values, p=probabilities)

    def evaluate_log_probability(self, value: int) -> float:
        """Calculates the log-probability of a given value."""
        log_probs = self.logscores - logsumexp(self.logscores)
        try:
            # Find the index of the value and return its log probability
            return float(log_probs[np.where(self.values == value)[0][0]])
        except IndexError:
            # The value was not a possible outcome under this distribution
            return -np.inf

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(values={self.values}, logscores={self.logscores})"

    # def add_noise_to_full_domain(
    #     self, all_possible_values: np.ndarray, noise_logScore: float = -10.0
    # ) -> Self:
    #     """
    #     Expands this distribution to cover all possible values in the domain.
    #     Values not in the current distribution get the noise_logScore.
    #     This converts expert "opinions" into full probability distributions.
    #     """
    #     new_logscores = np.full_like(all_possible_values, noise_logScore, dtype=float)
    #     for i, val in enumerate(self.values):
    #         if val in all_possible_values:
    #             idx = np.where(all_possible_values == val)[0][0]
    #             new_logscores[idx] = self.logscores[i]
    #     return RandomValues(values=all_possible_values, logscores=new_logscores)


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
