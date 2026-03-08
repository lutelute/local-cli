"""Tests for local_cli.token_tracker module."""

import unittest

from local_cli.token_tracker import (
    TokenTracker,
    TokenUsage,
    _DEFAULT_INPUT_COST_PER_TOKEN,
    _DEFAULT_OUTPUT_COST_PER_TOKEN,
)


# ---------------------------------------------------------------------------
# TokenUsage tests
# ---------------------------------------------------------------------------


class TestTokenUsage(unittest.TestCase):
    """Tests for the TokenUsage data class."""

    def test_default_values(self) -> None:
        """TokenUsage defaults to zero tokens and ollama provider."""
        usage = TokenUsage()
        self.assertEqual(usage.input_tokens, 0)
        self.assertEqual(usage.output_tokens, 0)
        self.assertEqual(usage.provider, "ollama")

    def test_custom_values(self) -> None:
        """TokenUsage stores provided values."""
        usage = TokenUsage(input_tokens=100, output_tokens=50, provider="claude")
        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.output_tokens, 50)
        self.assertEqual(usage.provider, "claude")

    def test_total_tokens(self) -> None:
        """total_tokens returns the sum of input and output."""
        usage = TokenUsage(input_tokens=200, output_tokens=300)
        self.assertEqual(usage.total_tokens, 500)

    def test_total_tokens_zero(self) -> None:
        """total_tokens returns 0 when both are zero."""
        usage = TokenUsage()
        self.assertEqual(usage.total_tokens, 0)

    def test_to_dict(self) -> None:
        """to_dict() produces the expected dictionary structure."""
        usage = TokenUsage(input_tokens=10, output_tokens=20, provider="claude")
        result = usage.to_dict()
        self.assertEqual(result, {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "provider": "claude",
        })

    def test_from_dict(self) -> None:
        """from_dict() reconstructs a TokenUsage from a dictionary."""
        data = {
            "input_tokens": 42,
            "output_tokens": 58,
            "provider": "ollama",
        }
        usage = TokenUsage.from_dict(data)
        self.assertEqual(usage.input_tokens, 42)
        self.assertEqual(usage.output_tokens, 58)
        self.assertEqual(usage.provider, "ollama")

    def test_from_dict_missing_keys(self) -> None:
        """from_dict() defaults missing keys gracefully."""
        usage = TokenUsage.from_dict({})
        self.assertEqual(usage.input_tokens, 0)
        self.assertEqual(usage.output_tokens, 0)
        self.assertEqual(usage.provider, "ollama")

    def test_roundtrip(self) -> None:
        """to_dict() and from_dict() round-trip correctly."""
        original = TokenUsage(input_tokens=123, output_tokens=456, provider="claude")
        restored = TokenUsage.from_dict(original.to_dict())
        self.assertEqual(restored.input_tokens, original.input_tokens)
        self.assertEqual(restored.output_tokens, original.output_tokens)
        self.assertEqual(restored.provider, original.provider)

    def test_repr(self) -> None:
        """__repr__ returns a readable string."""
        usage = TokenUsage(input_tokens=10, output_tokens=20, provider="ollama")
        r = repr(usage)
        self.assertIn("TokenUsage", r)
        self.assertIn("10", r)
        self.assertIn("20", r)
        self.assertIn("ollama", r)


# ---------------------------------------------------------------------------
# TokenTracker basic tests
# ---------------------------------------------------------------------------


class TestTokenTrackerBasic(unittest.TestCase):
    """Tests for basic TokenTracker functionality."""

    def test_empty_tracker(self) -> None:
        """New tracker has zero counts and no records."""
        tracker = TokenTracker()
        self.assertEqual(tracker.message_count, 0)
        self.assertEqual(tracker.total_input_tokens, 0)
        self.assertEqual(tracker.total_output_tokens, 0)
        self.assertEqual(tracker.total_tokens, 0)
        self.assertEqual(tracker.records, [])

    def test_record_single_usage(self) -> None:
        """Recording a single usage increments all counters."""
        tracker = TokenTracker()
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        tracker.record(usage)
        self.assertEqual(tracker.message_count, 1)
        self.assertEqual(tracker.total_input_tokens, 100)
        self.assertEqual(tracker.total_output_tokens, 50)
        self.assertEqual(tracker.total_tokens, 150)

    def test_record_multiple_usages(self) -> None:
        """Multiple recordings are accumulated correctly."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50))
        tracker.record(TokenUsage(input_tokens=200, output_tokens=100))
        tracker.record(TokenUsage(input_tokens=300, output_tokens=150))
        self.assertEqual(tracker.message_count, 3)
        self.assertEqual(tracker.total_input_tokens, 600)
        self.assertEqual(tracker.total_output_tokens, 300)
        self.assertEqual(tracker.total_tokens, 900)

    def test_records_returns_copy(self) -> None:
        """records property returns a copy, not the internal list."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=10, output_tokens=5))
        records = tracker.records
        records.clear()  # mutate the copy
        # Original should be unaffected.
        self.assertEqual(tracker.message_count, 1)

    def test_clear(self) -> None:
        """clear() removes all recorded usages."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50))
        tracker.record(TokenUsage(input_tokens=200, output_tokens=100))
        tracker.clear()
        self.assertEqual(tracker.message_count, 0)
        self.assertEqual(tracker.total_tokens, 0)
        self.assertEqual(tracker.records, [])


# ---------------------------------------------------------------------------
# Ollama recording tests
# ---------------------------------------------------------------------------


class TestRecordFromOllama(unittest.TestCase):
    """Tests for TokenTracker.record_from_ollama()."""

    def test_extracts_token_counts(self) -> None:
        """Extracts prompt_eval_count and eval_count from Ollama response."""
        tracker = TokenTracker()
        response = {
            "done": True,
            "prompt_eval_count": 150,
            "eval_count": 75,
            "model": "qwen3:8b",
        }
        usage = tracker.record_from_ollama(response)
        self.assertEqual(usage.input_tokens, 150)
        self.assertEqual(usage.output_tokens, 75)
        self.assertEqual(usage.provider, "ollama")
        self.assertEqual(tracker.message_count, 1)

    def test_handles_missing_fields(self) -> None:
        """Missing token fields default to zero."""
        tracker = TokenTracker()
        response = {"done": True, "model": "qwen3:8b"}
        usage = tracker.record_from_ollama(response)
        self.assertEqual(usage.input_tokens, 0)
        self.assertEqual(usage.output_tokens, 0)

    def test_handles_none_values(self) -> None:
        """None token values are treated as zero."""
        tracker = TokenTracker()
        response = {
            "done": True,
            "prompt_eval_count": None,
            "eval_count": None,
        }
        usage = tracker.record_from_ollama(response)
        self.assertEqual(usage.input_tokens, 0)
        self.assertEqual(usage.output_tokens, 0)

    def test_empty_response(self) -> None:
        """Empty response dict yields zero tokens."""
        tracker = TokenTracker()
        usage = tracker.record_from_ollama({})
        self.assertEqual(usage.input_tokens, 0)
        self.assertEqual(usage.output_tokens, 0)
        self.assertEqual(usage.provider, "ollama")


# ---------------------------------------------------------------------------
# Claude recording tests
# ---------------------------------------------------------------------------


class TestRecordFromClaude(unittest.TestCase):
    """Tests for TokenTracker.record_from_claude()."""

    def test_extracts_token_counts(self) -> None:
        """Extracts input_tokens and output_tokens from Claude response."""
        tracker = TokenTracker()
        response = {
            "usage": {
                "input_tokens": 500,
                "output_tokens": 200,
            },
            "model": "claude-sonnet-4-20250514",
        }
        usage = tracker.record_from_claude(response)
        self.assertEqual(usage.input_tokens, 500)
        self.assertEqual(usage.output_tokens, 200)
        self.assertEqual(usage.provider, "claude")
        self.assertEqual(tracker.message_count, 1)

    def test_handles_missing_usage(self) -> None:
        """Missing usage field defaults to zero tokens."""
        tracker = TokenTracker()
        response = {"model": "claude-sonnet-4-20250514"}
        usage = tracker.record_from_claude(response)
        self.assertEqual(usage.input_tokens, 0)
        self.assertEqual(usage.output_tokens, 0)

    def test_handles_none_values_in_usage(self) -> None:
        """None values in usage dict are treated as zero."""
        tracker = TokenTracker()
        response = {
            "usage": {
                "input_tokens": None,
                "output_tokens": None,
            },
        }
        usage = tracker.record_from_claude(response)
        self.assertEqual(usage.input_tokens, 0)
        self.assertEqual(usage.output_tokens, 0)

    def test_empty_usage_dict(self) -> None:
        """Empty usage dict yields zero tokens."""
        tracker = TokenTracker()
        response = {"usage": {}}
        usage = tracker.record_from_claude(response)
        self.assertEqual(usage.input_tokens, 0)
        self.assertEqual(usage.output_tokens, 0)


# ---------------------------------------------------------------------------
# Cost estimation tests
# ---------------------------------------------------------------------------


class TestEstimatedCost(unittest.TestCase):
    """Tests for TokenTracker.estimated_cost()."""

    def test_returns_none_for_ollama_only(self) -> None:
        """Returns None when all records are from Ollama."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50, provider="ollama"))
        self.assertIsNone(tracker.estimated_cost())

    def test_returns_none_for_empty_tracker(self) -> None:
        """Returns None when no records exist."""
        tracker = TokenTracker()
        self.assertIsNone(tracker.estimated_cost())

    def test_calculates_cost_for_claude(self) -> None:
        """Correctly calculates cost based on Claude token counts."""
        tracker = TokenTracker(
            input_cost_per_token=0.000003,
            output_cost_per_token=0.000015,
        )
        tracker.record(
            TokenUsage(input_tokens=1000, output_tokens=500, provider="claude")
        )
        cost = tracker.estimated_cost()
        self.assertIsNotNone(cost)
        # 1000 * 0.000003 + 500 * 0.000015 = 0.003 + 0.0075 = 0.0105
        self.assertAlmostEqual(cost, 0.0105, places=6)

    def test_ignores_ollama_in_cost_calculation(self) -> None:
        """Ollama records are excluded from cost calculation."""
        tracker = TokenTracker(
            input_cost_per_token=0.000003,
            output_cost_per_token=0.000015,
        )
        tracker.record(
            TokenUsage(input_tokens=1000, output_tokens=500, provider="ollama")
        )
        tracker.record(
            TokenUsage(input_tokens=100, output_tokens=50, provider="claude")
        )
        cost = tracker.estimated_cost()
        self.assertIsNotNone(cost)
        # Only Claude: 100 * 0.000003 + 50 * 0.000015 = 0.0003 + 0.00075 = 0.00105
        self.assertAlmostEqual(cost, 0.00105, places=6)

    def test_accumulates_across_claude_records(self) -> None:
        """Cost accumulates across multiple Claude records."""
        tracker = TokenTracker(
            input_cost_per_token=0.000003,
            output_cost_per_token=0.000015,
        )
        tracker.record(
            TokenUsage(input_tokens=100, output_tokens=50, provider="claude")
        )
        tracker.record(
            TokenUsage(input_tokens=200, output_tokens=100, provider="claude")
        )
        cost = tracker.estimated_cost()
        # (100+200) * 0.000003 + (50+100) * 0.000015
        # = 300 * 0.000003 + 150 * 0.000015
        # = 0.0009 + 0.00225 = 0.00315
        self.assertAlmostEqual(cost, 0.00315, places=6)

    def test_custom_cost_rates(self) -> None:
        """Custom per-token cost rates are applied correctly."""
        tracker = TokenTracker(
            input_cost_per_token=0.00001,
            output_cost_per_token=0.00005,
        )
        tracker.record(
            TokenUsage(input_tokens=1000, output_tokens=500, provider="claude")
        )
        cost = tracker.estimated_cost()
        # 1000 * 0.00001 + 500 * 0.00005 = 0.01 + 0.025 = 0.035
        self.assertAlmostEqual(cost, 0.035, places=6)


# ---------------------------------------------------------------------------
# Formatting tests
# ---------------------------------------------------------------------------


class TestFormatSummary(unittest.TestCase):
    """Tests for TokenTracker.format_summary()."""

    def test_empty_tracker(self) -> None:
        """Empty tracker shows 'No token usage recorded.'."""
        tracker = TokenTracker()
        self.assertEqual(tracker.format_summary(), "No token usage recorded.")

    def test_ollama_summary(self) -> None:
        """Ollama-only tracker shows 'Cost: N/A (local)'."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50, provider="ollama"))
        summary = tracker.format_summary()
        self.assertIn("100", summary)
        self.assertIn("50", summary)
        self.assertIn("150", summary)
        self.assertIn("N/A", summary)

    def test_claude_summary_includes_cost(self) -> None:
        """Claude tracker shows estimated cost."""
        tracker = TokenTracker(
            input_cost_per_token=0.000003,
            output_cost_per_token=0.000015,
        )
        tracker.record(
            TokenUsage(input_tokens=1000, output_tokens=500, provider="claude")
        )
        summary = tracker.format_summary()
        self.assertIn("1,000", summary)
        self.assertIn("500", summary)
        self.assertIn("Est. cost:", summary)
        self.assertIn("$", summary)

    def test_mixed_providers_shows_cost(self) -> None:
        """Mixed Ollama+Claude shows cost from Claude portion only."""
        tracker = TokenTracker(
            input_cost_per_token=0.000003,
            output_cost_per_token=0.000015,
        )
        tracker.record(
            TokenUsage(input_tokens=500, output_tokens=200, provider="ollama")
        )
        tracker.record(
            TokenUsage(input_tokens=100, output_tokens=50, provider="claude")
        )
        summary = tracker.format_summary()
        # Total tokens includes both providers.
        self.assertIn("600", summary)  # 500 + 100 input
        self.assertIn("250", summary)  # 200 + 50 output
        self.assertIn("Est. cost:", summary)


class TestFormatTable(unittest.TestCase):
    """Tests for TokenTracker.format_table()."""

    def test_empty_tracker(self) -> None:
        """Empty tracker shows 'No token usage recorded.'."""
        tracker = TokenTracker()
        self.assertEqual(tracker.format_table(), "No token usage recorded.")

    def test_single_ollama_record(self) -> None:
        """Single Ollama record renders correctly in table."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50, provider="ollama"))
        table = tracker.format_table()
        self.assertIn("ollama", table)
        self.assertIn("100", table)
        self.assertIn("50", table)
        self.assertIn("150", table)
        self.assertIn("N/A", table)
        self.assertIn("TOTAL", table)

    def test_table_has_header(self) -> None:
        """Table includes a header row with column names."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=10, output_tokens=5))
        table = tracker.format_table()
        self.assertIn("Provider", table)
        self.assertIn("Input", table)
        self.assertIn("Output", table)
        self.assertIn("Total", table)
        self.assertIn("Cost", table)

    def test_claude_record_shows_cost(self) -> None:
        """Claude record shows dollar amount in cost column."""
        tracker = TokenTracker(
            input_cost_per_token=0.000003,
            output_cost_per_token=0.000015,
        )
        tracker.record(
            TokenUsage(input_tokens=1000, output_tokens=500, provider="claude")
        )
        table = tracker.format_table()
        self.assertIn("claude", table)
        self.assertIn("$", table)
        self.assertIn("TOTAL", table)

    def test_multiple_records(self) -> None:
        """Multiple records each get their own row."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50, provider="ollama"))
        tracker.record(TokenUsage(input_tokens=200, output_tokens=100, provider="ollama"))
        table = tracker.format_table()
        lines = table.strip().split("\n")
        # Header + separator + 2 data rows + separator + total row = 6 lines.
        self.assertEqual(len(lines), 6)


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization(unittest.TestCase):
    """Tests for TokenTracker serialization (to_dict / from_dict)."""

    def test_empty_tracker_to_dict(self) -> None:
        """Empty tracker serializes to zero counts."""
        tracker = TokenTracker()
        data = tracker.to_dict()
        self.assertEqual(data["records"], [])
        self.assertEqual(data["total_input_tokens"], 0)
        self.assertEqual(data["total_output_tokens"], 0)
        self.assertEqual(data["total_tokens"], 0)

    def test_tracker_to_dict(self) -> None:
        """Tracker with records serializes all data."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50, provider="ollama"))
        tracker.record(TokenUsage(input_tokens=200, output_tokens=100, provider="claude"))
        data = tracker.to_dict()
        self.assertEqual(len(data["records"]), 2)
        self.assertEqual(data["total_input_tokens"], 300)
        self.assertEqual(data["total_output_tokens"], 150)
        self.assertEqual(data["total_tokens"], 450)

    def test_roundtrip(self) -> None:
        """to_dict() and from_dict() round-trip preserves all data."""
        tracker = TokenTracker(
            input_cost_per_token=0.00001,
            output_cost_per_token=0.00005,
        )
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50, provider="ollama"))
        tracker.record(TokenUsage(input_tokens=200, output_tokens=100, provider="claude"))

        data = tracker.to_dict()
        restored = TokenTracker.from_dict(
            data,
            input_cost_per_token=0.00001,
            output_cost_per_token=0.00005,
        )

        self.assertEqual(restored.message_count, 2)
        self.assertEqual(restored.total_input_tokens, 300)
        self.assertEqual(restored.total_output_tokens, 150)
        self.assertEqual(restored.total_tokens, 450)

        # Verify individual records.
        records = restored.records
        self.assertEqual(records[0].provider, "ollama")
        self.assertEqual(records[1].provider, "claude")

    def test_from_dict_empty(self) -> None:
        """from_dict() handles empty data dict gracefully."""
        tracker = TokenTracker.from_dict({})
        self.assertEqual(tracker.message_count, 0)
        self.assertEqual(tracker.total_tokens, 0)

    def test_from_dict_with_custom_rates(self) -> None:
        """from_dict() accepts custom cost rates."""
        data = {
            "records": [
                {"input_tokens": 100, "output_tokens": 50, "provider": "claude"},
            ],
        }
        tracker = TokenTracker.from_dict(
            data,
            input_cost_per_token=0.00002,
            output_cost_per_token=0.00006,
        )
        cost = tracker.estimated_cost()
        # 100 * 0.00002 + 50 * 0.00006 = 0.002 + 0.003 = 0.005
        self.assertAlmostEqual(cost, 0.005, places=6)


# ---------------------------------------------------------------------------
# Default cost constants tests
# ---------------------------------------------------------------------------


class TestDefaultConstants(unittest.TestCase):
    """Tests for module-level default constants."""

    def test_default_input_cost_per_token(self) -> None:
        """Default input cost per token is positive."""
        self.assertGreater(_DEFAULT_INPUT_COST_PER_TOKEN, 0)

    def test_default_output_cost_per_token(self) -> None:
        """Default output cost per token is positive."""
        self.assertGreater(_DEFAULT_OUTPUT_COST_PER_TOKEN, 0)

    def test_output_cost_greater_than_input(self) -> None:
        """Output tokens are more expensive than input tokens."""
        self.assertGreater(
            _DEFAULT_OUTPUT_COST_PER_TOKEN,
            _DEFAULT_INPUT_COST_PER_TOKEN,
        )


if __name__ == "__main__":
    unittest.main()
