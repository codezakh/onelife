from .core import LawProtocol
from crafter.state_export import WorldState
from typing import Any
from pathlib import Path
import cloudpickle


class LawToExpertWrapper:
    """
    Compatibility wrapper that converts a LawProtocol to an ExpertFunction.

    This wrapper runs the law's precondition and, if true, runs the effect.
    The effect should modify the state by assigning DiscreteDistribution objects
    to attributes the law has opinions about.
    """

    def __init__(self, law: LawProtocol[WorldState], source_code: str):
        self.law = law
        self.source_code = source_code
        # Surface a simple, instance-level name for display/logging
        self.__name__ = law.__name__

    def __call__(self, current_state: WorldState, action: Any) -> None:
        """
        Execute the law as an expert function.

        Args:
            current_state: The symbolic state to modify (mutated in-place)
            action: The action being taken
        """
        # Check precondition
        if self.law.precondition(current_state, action):
            # If precondition is true, run the effect
            self.law.effect(current_state, action)

    @property
    def __source_code__(self) -> str:
        """The source code of the law."""
        return self.source_code

    def save(self, path: str | Path) -> None:
        """Save the law to a file using cloudpickle."""
        if not isinstance(path, Path):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "LawToExpertWrapper":
        """Load a law from a file using cloudpickle."""
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
