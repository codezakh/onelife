from typing import Protocol
import attrs
from PIL.Image import Image as PILImage
from typing import Any
from typing import Literal, Optional
from dataclasses import dataclass
from typing import Generic, TypeVar
from pydantic import BaseModel
from typing import Union
from collections import namedtuple


LLMResponse = namedtuple(
    "LLMResponse",
    [
        "model_id",
        "completion",
        "stop_reason",
        "input_tokens",
        "output_tokens",
        "reasoning",
    ],
)


@attrs.define
class Message:
    role: Literal["system", "user", "assistant"]
    content: str
    attachment: Any | None = None


@attrs.define
class Text:
    long_term_context: str
    short_term_context: str


@attrs.define
class Observation:
    text: Text
    image: Optional[PILImage] = None
    obs: Any = None


MetadataT = TypeVar("MetadataT", bound=Union[BaseModel, dict])
MetadataT_co = TypeVar("MetadataT_co", bound=Union[BaseModel, dict], covariant=True)


@attrs.define
class Experience(Generic[MetadataT]):
    obs: Observation
    action: str
    reward: float
    done: bool
    truncated: bool
    info: MetadataT


@attrs.define
class OnResetExperience(Generic[MetadataT]):
    obs: Observation
    info: MetadataT


class PromptBuilderProtocol(Protocol):
    def update_instruction_prompt(self, instruction: str) -> None: ...
    def update_observation(self, obs: Observation) -> None: ...
    def update_action(self, action: str) -> None: ...
    def reset(self) -> None: ...
    def get_prompt(self, icl_episodes: bool = False) -> list[Message]: ...


class LlmClientProtocol(Protocol):
    def generate(self, messages: list[Message]) -> LLMResponse: ...


class AgentProtocol(Protocol):
    def act(self, obs: Observation, prev_action: str | None = None) -> LLMResponse: ...

    def reset(self) -> None: ...

    @property
    def prompt_builder(self) -> PromptBuilderProtocol: ...


class EnvironmentProtocol(Protocol[MetadataT]):
    def reset(self, **kwargs) -> OnResetExperience[MetadataT]: ...

    def step(self, action: str) -> Experience[MetadataT]: ...

    def get_instruction_prompt(self, instructions: str | None = None) -> str: ...

    def check_action_validity(self, candidate_action: str) -> str: ...

    def get_stats(self) -> dict: ...

    @property
    def failed_candidates(self) -> list[str]:
        """
        Return a list of invalid actions that were tried.
        """
        ...
