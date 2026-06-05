from dataclasses import dataclass
from pathlib import Path
from typing import cast
import litellm
import asyncio
from litellm.types.utils import Choices, ModelResponse
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm

from .core import LawInduction
from .token_counting import TokenTracker


class EmptyResponseError(Exception):
    """Exception raised when an LLM returns an empty response that should trigger a retry."""

    pass


llm_retry = retry(
    stop=stop_after_attempt(3),  # Retry up to 3 times total (including initial attempt)
    wait=wait_exponential(multiplier=1, min=1, max=10),  # Exponential backoff
    retry=retry_if_exception_type(
        EmptyResponseError
    ),  # Only retry on EmptyResponseError
    reraise=True,  # Re-raise the exception after all retries are exhausted
)


# Retry decorator for LLM calls that retries on empty responses
@dataclass
class SynthesisTask:
    """A synthesis task that encapsulates all context needed to generate laws for a specific transition and aspect."""

    transition_idx: int
    aspect: str
    prompt: str
    output_dir: Path

    @property
    def unique_name(self) -> str:
        """Generate a unique name for this task based on transition index and aspect."""
        return f"transition_{self.transition_idx}_aspect_{self.aspect}"

    @property
    def output_file_path(self) -> Path:
        """Get the output file path for this task's laws."""
        return self.output_dir / "laws" / f"{self.unique_name}.jsonl"

    def output_exists(self) -> bool:
        """Check if the output file for this task already exists."""
        return self.output_file_path.exists()

    def save_laws(self, laws: list["LawInduction"]) -> None:
        """Save the generated laws to the task's output file."""
        # Ensure the laws directory exists
        self.output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save laws as JSON lines
        with open(self.output_file_path, "w") as f:
            for law in laws:
                f.write(law.model_dump_json() + "\n")

    def load_existing_laws(self) -> list["LawInduction"]:
        """Load existing laws from the task's output file if it exists."""
        if not self.output_exists():
            return []

        laws = []
        with open(self.output_file_path, "r") as f:
            for line in f:
                if line.strip():
                    laws.append(LawInduction.model_validate_json(line.strip()))
        return laws

    @classmethod
    def gather_all_laws(cls, tasks: list["SynthesisTask"]) -> list["LawInduction"]:
        """Gather all laws from all completed tasks."""
        all_laws = []
        for task in tasks:
            if task.output_exists():
                all_laws.extend(task.load_existing_laws())
        return all_laws


def _process_synthesis_response(
    task: SynthesisTask,
    response: ModelResponse,
    messages: list[dict],
    token_tracker: TokenTracker,
) -> list[LawInduction]:
    """Process the LLM response for a synthesis task.

    This is the common logic shared between async and sync execution methods.

    Args:
        task: The synthesis task being executed
        response: The LLM response
        messages: The messages sent to the LLM

    Returns:
        List of laws generated for this task

    Raises:
        EmptyResponseError: If the response is empty or cannot be parsed
    """
    # Track token usage and cost
    token_tracker.track_call(response, messages)

    # Log token usage for this call
    usage = getattr(response, "usage", None)
    if usage:
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0
    else:
        input_tokens = output_tokens = total_tokens = 0

    cost = getattr(response, "_hidden_params", {}).get("response_cost", 0) or 0

    cost_str = f"${float(cost):.4f}"
    logger.info(
        f"Task {task.unique_name}: {input_tokens} input, {output_tokens} output, {total_tokens} total tokens, cost: {cost_str}"
    )

    choices = cast(Choices, response.choices)
    response_content = choices[0].message.content

    if not response_content:
        logger.warning(f"Empty response for task {task.unique_name}, triggering retry")
        raise EmptyResponseError(f"Empty response received for task {task.unique_name}")

    # Try to parse multiple laws first
    try:
        laws = LawInduction.parse_multiple(response_content)
        logger.info(f"Parsed {len(laws)} laws from response for {task.unique_name}")
    except ValueError:
        # Fallback to single law parsing if multiple parsing fails
        try:
            law = LawInduction.parse(response_content)
            laws = [law]
            logger.info(
                f"Parsed 1 law from response for {task.unique_name} (single law mode)"
            )
        except ValueError as e:
            logger.error(
                f"Failed to parse any laws from response for {task.unique_name}: {e}"
            )
            raise EmptyResponseError(
                f"Failed to parse laws from response for {task.unique_name}: {e}"
            )

    # Save laws to task's output file
    task.save_laws(laws)
    return laws


@llm_retry
async def execute_synthesis_task(
    task: SynthesisTask, token_tracker: TokenTracker
) -> list[LawInduction]:
    """Execute a single synthesis task asynchronously.

    Args:
        task: The synthesis task to execute
        token_tracker: TokenTracker to track token usage

    Returns:
        List of laws generated for this task
    """
    logger.info(f"Executing synthesis task: {task.unique_name}")

    # Make async LLM call using direct litellm.acompletion
    messages = [{"role": "user", "content": task.prompt}]
    response = cast(
        ModelResponse,
        await litellm.acompletion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
            max_tokens=8192,
            num_retries=0,  # Disable built-in retries, we handle retries with tenacity
        ),
    )

    return _process_synthesis_response(task, response, messages, token_tracker)


@llm_retry
def execute_synthesis_task_sync(
    task: SynthesisTask, token_tracker: TokenTracker
) -> list[LawInduction]:
    """Execute a single synthesis task synchronously.

    Args:
        task: The synthesis task to execute
        token_tracker: TokenTracker to track token usage

    Returns:
        List of laws generated for this task
    """
    logger.info(f"Executing synthesis task (sync): {task.unique_name}")

    # Make sync LLM call using litellm.completion
    messages = [{"role": "user", "content": task.prompt}]
    response = cast(
        ModelResponse,
        litellm.completion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
            max_tokens=8192,
            num_retries=0,  # Disable built-in retries, we handle retries with tenacity
        ),
    )

    return _process_synthesis_response(task, response, messages, token_tracker)


def execute_synthesis_tasks_sync(
    tasks: list[SynthesisTask], token_tracker: TokenTracker
) -> None:
    """Execute multiple synthesis tasks synchronously.

    Args:
        tasks: List of synthesis tasks to execute
        token_tracker: TokenTracker to track token usage
    """
    # Filter out tasks that already have output
    pending_tasks = [task for task in tasks if not task.output_exists()]

    if not pending_tasks:
        logger.info("All tasks already completed")
        return

    logger.info(
        f"Executing {len(pending_tasks)} synthesis tasks synchronously (skipping {len(tasks) - len(pending_tasks)} completed)"
    )

    # Execute tasks synchronously
    for task in pending_tasks:
        execute_synthesis_task_sync(task, token_tracker)


async def execute_synthesis_tasks(
    tasks: list[SynthesisTask], token_tracker: TokenTracker, max_concurrent: int = 5
) -> None:
    """Execute multiple synthesis tasks asynchronously with concurrency control.

    Args:
        tasks: List of synthesis tasks to execute
        token_tracker: TokenTracker to track token usage
        max_concurrent: Maximum number of concurrent tasks
    """
    # Filter out tasks that already have output
    pending_tasks = [task for task in tasks if not task.output_exists()]

    if not pending_tasks:
        logger.info("All tasks already completed")
        return

    logger.info(
        f"Executing {len(pending_tasks)} synthesis tasks (skipping {len(tasks) - len(pending_tasks)} completed)"
    )

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_with_semaphore(task: SynthesisTask) -> None:
        async with semaphore:
            await execute_synthesis_task(task, token_tracker)

    # Execute tasks concurrently
    tasks_with_progress = [
        execute_with_semaphore(task)
        for task in tqdm(pending_tasks, desc="Executing synthesis tasks")
    ]

    # Gather results (but we don't need the return values since tasks save themselves)
    results = await asyncio.gather(*tasks_with_progress, return_exceptions=True)

    # Handle any exceptions
    for i, result in enumerate(results):
        task = pending_tasks[i]
        if isinstance(result, Exception):
            logger.error(f"Task {task.unique_name} failed with exception: {result}")
