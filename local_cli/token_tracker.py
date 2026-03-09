"""Token usage tracking for local-cli.

Provides per-message and session-cumulative token counting for both
Ollama and Claude API providers.  Tracks input (prompt) and output
(completion) tokens separately.  Includes estimated cost calculation
for Claude API calls with configurable per-token rates (cost is N/A
for Ollama since it runs locally).
"""

from typing import Any

# ---------------------------------------------------------------------------
# Default cost rates (USD per token)
# ---------------------------------------------------------------------------

# Claude API pricing — configurable via TokenTracker constructor.
# These are default placeholder rates; callers should pass current
# pricing when creating a tracker.
_DEFAULT_INPUT_COST_PER_TOKEN: float = 0.000003  # $3.00 per 1M input tokens
_DEFAULT_OUTPUT_COST_PER_TOKEN: float = 0.000015  # $15.00 per 1M output tokens


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class TokenUsage:
    """Token usage for a single message exchange.

    Attributes:
        input_tokens: Number of input (prompt) tokens.
        output_tokens: Number of output (completion) tokens.
        provider: Provider name (e.g. ``"ollama"`` or ``"claude"``).
    """

    __slots__ = ("input_tokens", "output_tokens", "provider")

    def __init__(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        provider: str = "ollama",
    ) -> None:
        self.input_tokens: int = input_tokens
        self.output_tokens: int = output_tokens
        self.provider: str = provider

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for JSON storage.

        Returns:
            A dictionary with ``input_tokens``, ``output_tokens``,
            ``total_tokens``, and ``provider`` keys.
        """
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenUsage":
        """Deserialize from a dictionary.

        Args:
            data: A dictionary as produced by :meth:`to_dict`.

        Returns:
            A new :class:`TokenUsage` instance.
        """
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            provider=data.get("provider", "ollama"),
        )

    def __repr__(self) -> str:
        return (
            f"TokenUsage(input_tokens={self.input_tokens}, "
            f"output_tokens={self.output_tokens}, "
            f"provider={self.provider!r})"
        )


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class TokenTracker:
    """Session-scoped token usage tracker.

    Accumulates :class:`TokenUsage` records across the conversation
    session and provides summary statistics and cost estimation.

    Args:
        input_cost_per_token: Cost per input token in USD (Claude only).
        output_cost_per_token: Cost per output token in USD (Claude only).
    """

    def __init__(
        self,
        input_cost_per_token: float = _DEFAULT_INPUT_COST_PER_TOKEN,
        output_cost_per_token: float = _DEFAULT_OUTPUT_COST_PER_TOKEN,
    ) -> None:
        self._records: list[TokenUsage] = []
        self._input_cost_per_token: float = input_cost_per_token
        self._output_cost_per_token: float = output_cost_per_token

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, usage: TokenUsage) -> None:
        """Record a single message exchange's token usage.

        Args:
            usage: The :class:`TokenUsage` for one LLM round-trip.
        """
        self._records.append(usage)

    def record_from_ollama(self, response: dict[str, Any]) -> TokenUsage:
        """Extract and record token usage from an Ollama response chunk.

        Ollama reports ``prompt_eval_count`` (input tokens) and
        ``eval_count`` (output tokens) in the final streaming chunk
        (the one where ``"done": true``).

        Args:
            response: The assembled response dict from
                :func:`~local_cli.agent.collect_streaming_response`.

        Returns:
            The recorded :class:`TokenUsage`.
        """
        input_tokens = response.get("prompt_eval_count", 0) or 0
        output_tokens = response.get("eval_count", 0) or 0
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider="ollama",
        )
        self.record(usage)
        return usage

    def record_from_claude(self, response: dict[str, Any]) -> TokenUsage:
        """Extract and record token usage from a Claude API response.

        Claude returns token counts in the response's ``usage`` field
        with ``input_tokens`` and ``output_tokens`` keys.

        Args:
            response: The assembled response dict from
                the Claude provider.

        Returns:
            The recorded :class:`TokenUsage`.
        """
        usage_data = response.get("usage", {})
        input_tokens = usage_data.get("input_tokens", 0) or 0
        output_tokens = usage_data.get("output_tokens", 0) or 0
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider="claude",
        )
        self.record(usage)
        return usage

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def records(self) -> list[TokenUsage]:
        """All recorded token usage entries (read-only copy)."""
        return list(self._records)

    @property
    def message_count(self) -> int:
        """Number of recorded exchanges."""
        return len(self._records)

    @property
    def total_input_tokens(self) -> int:
        """Cumulative input tokens across all recorded exchanges."""
        return sum(r.input_tokens for r in self._records)

    @property
    def total_output_tokens(self) -> int:
        """Cumulative output tokens across all recorded exchanges."""
        return sum(r.output_tokens for r in self._records)

    @property
    def total_tokens(self) -> int:
        """Cumulative total tokens (input + output) across all exchanges."""
        return self.total_input_tokens + self.total_output_tokens

    def estimated_cost(self) -> float | None:
        """Estimate cumulative USD cost for Claude API usage.

        Only considers records where ``provider == "claude"``.  Returns
        ``None`` if there are no Claude records (i.e. pure Ollama usage).

        Returns:
            Estimated cost in USD, or ``None`` if no Claude usage.
        """
        claude_records = [r for r in self._records if r.provider == "claude"]
        if not claude_records:
            return None

        total_input = sum(r.input_tokens for r in claude_records)
        total_output = sum(r.output_tokens for r in claude_records)

        cost = (
            total_input * self._input_cost_per_token
            + total_output * self._output_cost_per_token
        )
        return cost

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_summary(self) -> str:
        """Format a compact one-line summary of session token usage.

        Returns:
            A string like ``"Tokens: 1,234 in / 567 out (1,801 total)"``,
            or ``"No token usage recorded."`` if the tracker is empty.
        """
        if not self._records:
            return "No token usage recorded."

        parts = [
            f"Tokens: {self.total_input_tokens:,} in / "
            f"{self.total_output_tokens:,} out "
            f"({self.total_tokens:,} total)"
        ]

        cost = self.estimated_cost()
        if cost is not None:
            parts.append(f" | Est. cost: ${cost:.4f}")
        else:
            parts.append(" | Cost: N/A (local)")

        return "".join(parts)

    def format_table(self) -> str:
        """Format a detailed table of per-message token usage.

        Returns:
            A multi-line string table with columns for message number,
            provider, input tokens, output tokens, total, and cost.
            Includes a totals row at the bottom.
        """
        if not self._records:
            return "No token usage recorded."

        # Header.
        lines: list[str] = []
        lines.append(
            f"{'#':>3}  {'Provider':<8}  {'Input':>8}  "
            f"{'Output':>8}  {'Total':>8}  {'Cost':>10}"
        )
        lines.append("-" * len(lines[0]))

        # Per-message rows.
        for i, record in enumerate(self._records, start=1):
            if record.provider == "claude":
                cost_input = record.input_tokens * self._input_cost_per_token
                cost_output = (
                    record.output_tokens * self._output_cost_per_token
                )
                cost_str = f"${cost_input + cost_output:.4f}"
            else:
                cost_str = "N/A"

            lines.append(
                f"{i:>3}  {record.provider:<8}  "
                f"{record.input_tokens:>8,}  {record.output_tokens:>8,}  "
                f"{record.total_tokens:>8,}  {cost_str:>10}"
            )

        # Totals row.
        lines.append("-" * len(lines[0]))
        cost = self.estimated_cost()
        cost_str = f"${cost:.4f}" if cost is not None else "N/A"
        lines.append(
            f"{'':>3}  {'TOTAL':<8}  "
            f"{self.total_input_tokens:>8,}  {self.total_output_tokens:>8,}  "
            f"{self.total_tokens:>8,}  {cost_str:>10}"
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all recorded token usage.

        Called when the user runs ``/clear`` to reset the session.
        """
        self._records.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tracker state to a dictionary.

        Returns:
            A dictionary with ``records``, ``total_input_tokens``,
            ``total_output_tokens``, and ``total_tokens`` keys.
        """
        return {
            "records": [r.to_dict() for r in self._records],
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        input_cost_per_token: float = _DEFAULT_INPUT_COST_PER_TOKEN,
        output_cost_per_token: float = _DEFAULT_OUTPUT_COST_PER_TOKEN,
    ) -> "TokenTracker":
        """Deserialize from a dictionary.

        Args:
            data: A dictionary as produced by :meth:`to_dict`.
            input_cost_per_token: Cost per input token in USD.
            output_cost_per_token: Cost per output token in USD.

        Returns:
            A new :class:`TokenTracker` with restored records.
        """
        tracker = cls(
            input_cost_per_token=input_cost_per_token,
            output_cost_per_token=output_cost_per_token,
        )
        for record_data in data.get("records", []):
            tracker.record(TokenUsage.from_dict(record_data))
        return tracker
