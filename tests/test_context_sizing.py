"""Tests for adaptive context sizing (context_sizing.py).

qwen3.5:9b reports a native context of 262,144; the historical default
pinned every model to 8,192.  The resolver picks
min(model max, RAM tier, 32k) — never below the old default, never
above what the model actually supports.
"""

import unittest
from unittest.mock import MagicMock, patch

from local_cli.context_sizing import (
    _max_ctx_cache,
    model_max_context,
    resolve_num_ctx,
)


def _client(context_length: int | None = 262144) -> MagicMock:
    client = MagicMock()
    model_info = {"general.architecture": "qwen35"}
    if context_length is not None:
        model_info["qwen35.context_length"] = context_length
    client.show_model.return_value = {"model_info": model_info}
    return client


class TestResolveNumCtx(unittest.TestCase):
    def setUp(self) -> None:
        _max_ctx_cache.clear()

    def test_explicit_config_wins_without_lookup(self) -> None:
        client = _client()
        self.assertEqual(resolve_num_ctx(client, "m", 12345), 12345)
        client.show_model.assert_not_called()

    def test_big_machine_gets_ceiling(self) -> None:
        with patch(
            "local_cli.context_sizing._system_ram_gb", return_value=64.0,
        ):
            self.assertEqual(resolve_num_ctx(_client(), "m", 0), 32768)

    def test_mid_machine_gets_16k(self) -> None:
        with patch(
            "local_cli.context_sizing._system_ram_gb", return_value=16.0,
        ):
            self.assertEqual(resolve_num_ctx(_client(), "m", 0), 16384)

    def test_small_or_unknown_ram_keeps_floor(self) -> None:
        for ram in (8.0, 0.0):
            _max_ctx_cache.clear()
            with patch(
                "local_cli.context_sizing._system_ram_gb", return_value=ram,
            ):
                self.assertEqual(resolve_num_ctx(_client(), "m", 0), 8192)

    def test_model_smaller_than_floor_is_respected(self) -> None:
        with patch(
            "local_cli.context_sizing._system_ram_gb", return_value=64.0,
        ):
            self.assertEqual(
                resolve_num_ctx(_client(context_length=4096), "m", 0), 4096,
            )

    def test_model_between_floor_and_cap(self) -> None:
        with patch(
            "local_cli.context_sizing._system_ram_gb", return_value=64.0,
        ):
            self.assertEqual(
                resolve_num_ctx(_client(context_length=20000), "m", 0),
                20000,
            )

    def test_lookup_failure_falls_back_to_floor(self) -> None:
        client = MagicMock()
        client.show_model.side_effect = OSError("ollama down")
        self.assertEqual(resolve_num_ctx(client, "m", 0), 8192)


class TestModelMaxContext(unittest.TestCase):
    def setUp(self) -> None:
        _max_ctx_cache.clear()

    def test_reads_arch_prefixed_key(self) -> None:
        self.assertEqual(model_max_context(_client(), "m"), 262144)

    def test_cached_after_first_lookup(self) -> None:
        client = _client()
        model_max_context(client, "m")
        model_max_context(client, "m")
        self.assertEqual(client.show_model.call_count, 1)

    def test_failure_not_cached(self) -> None:
        client = MagicMock()
        client.show_model.side_effect = OSError("down")
        self.assertEqual(model_max_context(client, "m"), 0)
        client.show_model.side_effect = None
        client.show_model.return_value = {
            "model_info": {"llama.context_length": 131072},
        }
        self.assertEqual(model_max_context(client, "m"), 131072)

    def test_missing_key_returns_zero(self) -> None:
        self.assertEqual(
            model_max_context(_client(context_length=None), "m"), 0,
        )


if __name__ == "__main__":
    unittest.main()
