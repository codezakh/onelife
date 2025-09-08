from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional, Any, cast
from litellm.types.utils import ModelResponse, Choices
import litellm
from ulid import ULID
import os


class LiteLlmMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


# This class exists to make the type checker happy.
# The .choices field in the ModelResponse type is a union
# of Choice and StreamingChoice. The options we give the model
# will in practice always result in a Choice, so we narrow the
# type here and avoid tedious assertions or isinstance checks
# elsewhere in the code.
class NonStreamingModelResponse(ModelResponse):
    choices: list[Choices]  # pyright: ignore[reportIncompatibleVariableOverride]


class LiteLlmParamsBase(BaseModel):
    """
    A class encapsulating most of the parameters we're interested in
    specifying for a LiteLLM model call.

    Usage:
    ```python
    params = LiteLlmParamsBase(
        provider="openai",
        model_slug="gpt-4o",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    print(litellm.completion(
        model=params.model,
        messages=["Hello, world!"],
        **params.kwargs,
    ))
    ```
    """

    provider: str
    model_slug: str
    api_key: str
    n: int = 1
    stream: bool = False
    max_tokens: Optional[int] = None
    num_retries: int = 3
    api_base: Optional[str] = None

    @property
    def kwargs(self) -> dict[str, Any]:
        """
        Returns a dictionary of parameters that can be passed to
        litellm.completion.
        """
        # Exclude the model_slug and provider fields from the dump
        # because they have to be passed to litellm.completion after merging.
        _kwargs = self.model_dump(exclude={"model_slug", "provider"})
        return _kwargs

    @property
    def model(self) -> str:
        """
        Returns the model name as a string in the format LiteLLM expects.
        """
        return f"{self.provider}/{self.model_slug}"


class OpenAILiteLlmParams(LiteLlmParamsBase):
    provider: str = "openai"
    model_slug: str = "gpt-4.1-mini"
    api_key: str = Field(default_factory=lambda: os.environ["OPENAI_API_KEY"])


class GeminiLiteLlmParams(LiteLlmParamsBase):
    provider: str = "gemini"
    model_slug: str = "gemini-2.5-flash"
    api_key: str = Field(default_factory=lambda: os.environ["GEMINI_API_KEY"])


class HuggingFaceLiteLlmParams(LiteLlmParamsBase):
    provider: str = "huggingface"
    model_slug: str = "meta-llama/Llama-3.1-8B-Instruct"
    api_key: str = Field(default_factory=lambda: os.environ["HF_TOKEN"])


class VllmLiteLlmParams(LiteLlmParamsBase):
    provider: str = "hosted_vllm"
    model_slug: str = ""  # Replace with the name of the LoRA or model you want to use
    api_base: Optional[str] = "http://localhost:8000/v1"
    api_key: str = ""


class LiteLlmRequest(BaseModel):
    messages: list[LiteLlmMessage]
    params: LiteLlmParamsBase
    ulid: ULID = Field(default_factory=ULID)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self) -> NonStreamingModelResponse:
        response = litellm.completion(
            model=self.params.model,
            messages=[m.model_dump() for m in self.messages],
            **self.params.kwargs,
        )
        response = cast(ModelResponse, response)
        return NonStreamingModelResponse.model_validate(response, from_attributes=True)


if __name__ == "__main__":
    litellm._turn_on_debug()
    params = GeminiLiteLlmParams()
    request = LiteLlmRequest(
        messages=[
            LiteLlmMessage(role="user", content="What's 1+1?"),
        ],
        params=params,
    )
    print(request())
