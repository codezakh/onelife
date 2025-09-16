from torch.utils import data
from distant_sunburn.poe_world.core import ExpertFunction
from typing import Protocol, TypeVar, Any
from pathlib import Path
import cloudpickle
from typing_extensions import Self, Generic
from ..typing_utils import implements
import inspect
from typing import Type
from dataclasses import dataclass
from ..poe_world.core import ObservableId, DiscreteDistribution
import torch
from typing import Mapping, TypeAlias


SymbolicStateT = TypeVar("SymbolicStateT")
SymbolicStateT_contra = TypeVar("SymbolicStateT_contra", contravariant=True)


class LawProtocol(Protocol[SymbolicStateT_contra]):
    def precondition(
        self, current_state: SymbolicStateT_contra, action: Any
    ) -> bool: ...

    def effect(self, current_state: SymbolicStateT_contra, action: Any) -> None: ...

    @property
    def __source_code__(self) -> str: ...

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> "LawProtocol[SymbolicStateT_contra]": ...

    @property
    def __name__(self) -> str: ...


class BaseLawProtocol(Protocol[SymbolicStateT_contra]):
    def precondition(
        self, current_state: SymbolicStateT_contra, action: Any
    ) -> bool: ...

    def effect(self, current_state: SymbolicStateT_contra, action: Any) -> None: ...


class LawFunctionWrapper(Generic[SymbolicStateT_contra]):
    def __init__(self, law: BaseLawProtocol[SymbolicStateT_contra], source_code: str):
        self.law = law
        self.source_code = source_code

    def precondition(self, current_state: SymbolicStateT_contra, action: Any) -> bool:
        return self.law.precondition(current_state, action)

    def effect(self, current_state: SymbolicStateT_contra, action: Any) -> None:
        self.law.effect(current_state, action)

    def save(self, path: str | Path) -> None:
        if not isinstance(path, Path):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "LawFunctionWrapper[SymbolicStateT_contra]":
        """
        Deserializes an ExpertFunctionWrapper instance from a file using cloudpickle.

        Args:
            file_path: The path of the file to load.

        Returns:
            A new instance of ExpertFunctionWrapper.

        Raises:
            TypeError: If the unpickled object is not an instance of this class.
        """
        if not isinstance(path, Path):
            path = Path(path)
        with path.open("rb") as f:
            instance = cloudpickle.load(f)

        if not isinstance(instance, cls):
            raise TypeError(
                f"File '{path}' did not contain an instance of "
                f"{cls.__name__}, but of {type(instance).__name__}."
            )

        return instance

    @classmethod
    def from_non_runtime_created(
        cls, law: BaseLawProtocol[SymbolicStateT_contra]
    ) -> "LawFunctionWrapper[SymbolicStateT_contra]":
        return cls(law, inspect.getsource(law.__class__))

    @property
    def __name__(self) -> str:
        return self.law.__class__.__name__

    @property
    def __source_code__(self) -> str:
        return self.source_code


implements(LawProtocol)(LawFunctionWrapper)


@dataclass
class WeightedLaw(Generic[SymbolicStateT_contra]):
    law: LawProtocol[SymbolicStateT_contra]
    weight: float
    is_fitted: bool = False


SymbolicStateT = TypeVar("SymbolicStateT")


class WorldModelProtocol(Protocol[SymbolicStateT]):
    def sample_next_state(
        self, current_state: SymbolicStateT, action: Any
    ) -> SymbolicStateT: ...
    def evaluate_log_probability(
        self, state: SymbolicStateT, action: Any, next_state: SymbolicStateT
    ) -> float: ...
    def with_new_laws(
        self, new_laws: list[WeightedLaw[SymbolicStateT]]
    ) -> "WorldModelProtocol[SymbolicStateT]": ...
    @property
    def laws(self) -> list[WeightedLaw[SymbolicStateT]]: ...


@dataclass
class SymbolicTransition(Generic[SymbolicStateT]):
    prev_state: SymbolicStateT
    action: Any
    next_state: SymbolicStateT


class LawOptimizerProtocol(Protocol[SymbolicStateT]):
    def fit(
        self,
        laws: list[LawProtocol[SymbolicStateT]],
        transitions: list[SymbolicTransition[SymbolicStateT]],
    ) -> list[WeightedLaw[SymbolicStateT]]: ...


ExpertIndex: TypeAlias = int


class ObservableExtractorProtocol(Protocol[SymbolicStateT]):
    def extract_attribute_predictions(
        self, state: SymbolicStateT
    ) -> Mapping[ObservableId, DiscreteDistribution]:
        """
        Extract probabilistic predictions from a state after expert execution.

        This method is called after experts have modified a state by assigning
        DiscreteDistribution objects to attributes they have opinions about.
        The method should:
        1. Identify all observable attributes in the state
        2. Extract DiscreteDistribution predictions where experts made them
        3. Ensure all observable attributes are represented in the output

        Args:
            state: The symbolic state after expert execution. May contain both
                   primitive values and DiscreteDistribution objects.

        Returns:
            Dictionary mapping ObservableId to DiscreteDistribution for each
            observable attribute. All DiscreteDistribution objects should have
            the same support (domain) for a given attribute across calls.
        """
        ...

    def get_observed_outcomes(
        self, state: SymbolicStateT
    ) -> Mapping[ObservableId, int]:
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
        """
        ...

    def apply_expert_predictions(
        self,
        new_state: SymbolicStateT,
        expert_predictions: Mapping[
            ObservableId, Mapping[ExpertIndex, DiscreteDistribution]
        ],
        weights: torch.Tensor,
    ) -> SymbolicStateT:
        """
        Apply combined expert predictions to create a new state.

        This method takes the predictions from multiple experts for each attribute,
        combines them using the provided weights, and applies the results to
        create a new state.

        Args:
            new_state: A copy of the current state to be modified. This state
                       should contain primitive values and will be mutated in-place.
            expert_predictions: Dictionary mapping ObservableId to dict of
                               DiscreteDistribution predictions from each expert.
                               Each dict should have the same length as the weights tensor.
                               The keys of the dict are the indices of the experts.
            weights: Tensor of expert weights [n_experts] with dtype=torch.float32.
                     weights[i] determines how much expert i's prediction contributes.

        Returns:
            The modified state with sampled values from combined expert predictions.

        Requirements:
            - Must mutate new_state in-place and return it
            - Must combine predictions using the provided weights
            - Must sample from combined distributions to get concrete values
            - Must convert sampled values to appropriate types (e.g., bool for boolean attributes)
            - Must preserve state structure and handle missing predictions gracefully
        """
        ...
