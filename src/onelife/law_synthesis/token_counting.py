from litellm.types.utils import ModelResponse
from loguru import logger


import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class TokenUsageStats:
    """Statistics for token usage and costs."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    call_count: int = 0
    start_time: Optional[datetime] = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()

    @property
    def avg_input_tokens_per_call(self) -> float:
        return self.total_input_tokens / self.call_count if self.call_count > 0 else 0.0

    @property
    def avg_output_tokens_per_call(self) -> float:
        return (
            self.total_output_tokens / self.call_count if self.call_count > 0 else 0.0
        )

    @property
    def avg_total_tokens_per_call(self) -> float:
        return self.total_tokens / self.call_count if self.call_count > 0 else 0.0

    @property
    def avg_cost_per_call(self) -> float:
        return self.total_cost / self.call_count if self.call_count > 0 else 0.0

    @property
    def runtime_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()

    def log_current_stats(self, logger_instance) -> None:
        """Log current token usage statistics."""
        logger_instance.info(
            f"Token Usage Stats - Calls: {self.call_count}, "
            f"Input: {self.total_input_tokens} ({self.avg_input_tokens_per_call:.1f}/call), "
            f"Output: {self.total_output_tokens} ({self.avg_output_tokens_per_call:.1f}/call), "
            f"Total: {self.total_tokens} ({self.avg_total_tokens_per_call:.1f}/call), "
            f"Cost: ${self.total_cost:.4f} (${self.avg_cost_per_call:.4f}/call), "
            f"Runtime: {self.runtime_seconds:.1f}s"
        )

    def to_dict(self) -> dict:
        """Convert stats to dictionary for JSON serialization."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "call_count": self.call_count,
            "avg_input_tokens_per_call": self.avg_input_tokens_per_call,
            "avg_output_tokens_per_call": self.avg_output_tokens_per_call,
            "avg_total_tokens_per_call": self.avg_total_tokens_per_call,
            "avg_cost_per_call": self.avg_cost_per_call,
            "runtime_seconds": self.runtime_seconds,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": datetime.now().isoformat(),
        }

    def save_to_file(self, filepath: Path) -> None:
        """Save statistics to a JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved token usage statistics to {filepath}")


class TokenTracker:
    """Tracks token usage and costs across multiple LLM calls."""

    def __init__(self):
        self.stats = TokenUsageStats()

    def track_call(self, response: ModelResponse, messages: list[dict]) -> None:
        """Track a single LLM call's token usage and cost."""
        # Get token counts from response
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or 0

            self.stats.total_input_tokens += input_tokens
            self.stats.total_output_tokens += output_tokens
            self.stats.total_tokens += total_tokens

        # Get cost from response if available
        if hasattr(response, "_hidden_params") and response._hidden_params:
            cost = response._hidden_params.get("response_cost")
            if cost is not None:
                self.stats.total_cost += float(cost)

        self.stats.call_count += 1

        # Log stats every 10 calls
        if self.stats.call_count % 10 == 0:
            self.stats.log_current_stats(logger)

    def get_stats(self) -> TokenUsageStats:
        """Get current statistics."""
        return self.stats

    def log_final_stats(self) -> None:
        """Log final statistics."""
        logger.info("=== FINAL TOKEN USAGE STATISTICS ===")
        self.stats.log_current_stats(logger)
        logger.info("===================================")

    def save_final_stats(self, output_dir: Path) -> None:
        """Save final statistics to file."""
        stats_file = output_dir / "token_usage_stats.json"
        self.stats.save_to_file(stats_file)
