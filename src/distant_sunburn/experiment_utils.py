from pathlib import Path
from loguru import logger
from dataclasses import dataclass
import re
from typing import Self
import shutil


@dataclass(frozen=True)
class ExperimentIdentifier:
    number: int
    description: str

    @classmethod
    def parse_from_script_name(cls, script_name: str) -> Self:
        name_as_path = Path(script_name)
        stem = name_as_path.stem
        # We expect the script name to be of the form e<number>_<description>.py
        # where <number> is a unique identifier for the experiment and <name> is the name of the experiment.
        # Ex: e201_train_model.py
        pattern = r"^e(\d+)_(.+)$"
        match = re.match(pattern, stem)
        if not match:
            raise ValueError(f"Invalid experiment name format: {stem}")
        number = int(match.group(1))
        description = match.group(2)
        return cls(number=number, description=description)


class ExperimentWorkspaceManager:
    def __init__(
        self,
        identifier: ExperimentIdentifier,
        workspace_path: Path = Path("./workspace"),
    ):
        self.identifier = identifier
        self.workspace_path = workspace_path

    @property
    def output_dir_name(self) -> str:
        return (
            f"experiments__{self.identifier.number:04d}_{self.identifier.description}"
        )

    @property
    def output_dir(self) -> Path:
        return self.workspace_path / self.output_dir_name

    def setup(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    @classmethod
    def from_dunder_file(cls, __file__: str) -> Self:
        identifier = ExperimentIdentifier.parse_from_script_name(__file__)
        return cls(identifier)
