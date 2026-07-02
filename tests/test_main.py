"""Tests for local_cli.__main__ — the decomposed startup sequence.

The startup steps were previously one untested 300-line main(); these
tests exercise each step in isolation with mocks.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from local_cli.__main__ import (
    _apply_arg_aliases,
    _build_tools,
    _dispatch_alternate_modes,
    _init_orchestrator,
    _init_rag,
    _load_registry,
    _maybe_select_model,
    _resolve_initial_mode,
    _show_update_notice,
    _start_update_check,
)
from local_cli.config import Config


def _args(**kwargs) -> SimpleNamespace:
    """An argparse-like namespace with the given flags set."""
    return SimpleNamespace(**kwargs)


class TestApplyArgAliases(unittest.TestCase):
    """CLI args whose names differ from config keys are mapped."""

    def test_brain_model_maps_to_orchestrator_model(self) -> None:
        config = Config()
        _apply_arg_aliases(_args(brain_model="qwen3:30b"), config)
        self.assertEqual(config.orchestrator_model, "qwen3:30b")

    def test_registry_file_maps_to_model_registry_file(self) -> None:
        config = Config()
        _apply_arg_aliases(_args(registry_file="/tmp/reg.json"), config)
        self.assertEqual(config.model_registry_file, "/tmp/reg.json")

    def test_absent_args_leave_config_unchanged(self) -> None:
        config = Config()
        original = config.orchestrator_model
        _apply_arg_aliases(_args(), config)
        self.assertEqual(config.orchestrator_model, original)


class TestDispatchAlternateModes(unittest.TestCase):
    """Non-REPL modes run and return True; otherwise False."""

    def test_no_flags_returns_false(self) -> None:
        self.assertFalse(_dispatch_alternate_modes(_args(), Config()))

    def test_server_mode(self) -> None:
        with patch("local_cli.server.run_server") as run_server:
            handled = _dispatch_alternate_modes(_args(server=True), Config())
        self.assertTrue(handled)
        run_server.assert_called_once()

    def test_web_monitor_mode(self) -> None:
        config = Config()
        with patch("local_cli.web_monitor.run_web_monitor") as run_web:
            handled = _dispatch_alternate_modes(
                _args(web_monitor=True, web_port=7071), config,
            )
        self.assertTrue(handled)
        run_web.assert_called_once_with(config=config, port=7071)

    def test_update_mode_no_updates(self) -> None:
        with patch(
            "local_cli.updater.check_for_updates",
            return_value=(False, "Up to date."),
        ):
            handled = _dispatch_alternate_modes(_args(update=True), Config())
        self.assertTrue(handled)


class TestBuildTools(unittest.TestCase):
    """Risky-command confirmation is wired unless --yes was passed."""

    def test_confirmation_wired_by_default(self) -> None:
        config = Config()
        config.auto_approve = False
        tools = _build_tools(config)
        bash = next(t for t in tools if t.name == "bash")
        self.assertIsNotNone(getattr(bash, "_confirm", None))

    def test_auto_approve_keeps_plain_bash(self) -> None:
        config = Config()
        config.auto_approve = True
        tools = _build_tools(config)
        bash = next(t for t in tools if t.name == "bash")
        self.assertIsNone(getattr(bash, "_confirm", None))


class TestLoadRegistry(unittest.TestCase):
    """Registry loading is optional and failure-tolerant."""

    def test_unconfigured_returns_none(self) -> None:
        config = Config()
        config.model_registry_file = ""
        self.assertIsNone(_load_registry(config))

    def test_missing_file_warns_and_returns_none(self) -> None:
        config = Config()
        config.model_registry_file = "/nonexistent/registry.json"
        with patch("sys.stderr") as stderr:
            result = _load_registry(config)
        self.assertIsNone(result)
        self.assertTrue(stderr.write.called)


class TestInitOrchestrator(unittest.TestCase):
    """Provider initialization with fallback semantics."""

    def test_working_provider_returned(self) -> None:
        config = Config()
        with patch("local_cli.__main__.Orchestrator") as orch_cls:
            orch = orch_cls.return_value
            orch.get_active_provider.return_value = MagicMock()
            result = _init_orchestrator(config, None)
        self.assertIs(result, orch)

    def test_non_ollama_failure_falls_back(self) -> None:
        config = Config()
        config.provider = "claude"
        with patch("local_cli.__main__.Orchestrator") as orch_cls:
            orch = orch_cls.return_value
            orch.get_active_provider.side_effect = ValueError("no API key")
            with patch("sys.stderr"):
                result = _init_orchestrator(config, None)
        orch.switch_provider.assert_called_once_with("ollama")
        self.assertIs(result, orch)

    def test_total_failure_exits(self) -> None:
        config = Config()
        config.provider = "claude"
        with patch("local_cli.__main__.Orchestrator") as orch_cls:
            orch = orch_cls.return_value
            orch.get_active_provider.side_effect = ValueError("no API key")
            orch.switch_provider.side_effect = ValueError("ollama down")
            with patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    _init_orchestrator(config, None)

    def test_ollama_failure_exits(self) -> None:
        config = Config()
        config.provider = "ollama"
        with patch("local_cli.__main__.Orchestrator") as orch_cls:
            orch = orch_cls.return_value
            orch.get_active_provider.side_effect = ValueError("down")
            with patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    _init_orchestrator(config, None)


class TestInitRag(unittest.TestCase):
    """RAG initialization is opt-in and failure-tolerant."""

    def test_disabled_returns_none_and_default_topk(self) -> None:
        engine, topk = _init_rag(_args(rag=False), MagicMock())
        self.assertIsNone(engine)
        self.assertEqual(topk, 5)

    def test_failure_warns_and_returns_none(self) -> None:
        with patch(
            "local_cli.__main__.RAGEngine",
            side_effect=RuntimeError("no embeddings"),
        ):
            with patch("sys.stderr"):
                engine, topk = _init_rag(
                    _args(rag=True, rag_path=".", rag_model="m", rag_topk=3),
                    MagicMock(),
                )
        self.assertIsNone(engine)
        self.assertEqual(topk, 3)


class TestResolveInitialMode(unittest.TestCase):
    """Initial REPL mode from flags and config."""

    def test_default_from_config(self) -> None:
        config = Config()
        config.default_mode = "agent"
        self.assertEqual(_resolve_initial_mode(_args(), config), "agent")

    def test_ideate_flag(self) -> None:
        config = Config()
        self.assertEqual(
            _resolve_initial_mode(_args(ideate=True), config), "ideate",
        )

    def test_plan_flag_forces_agent(self) -> None:
        config = Config()
        config.default_mode = "ideate"
        self.assertEqual(
            _resolve_initial_mode(_args(plan=True), config), "agent",
        )


class TestMaybeSelectModel(unittest.TestCase):
    """--select-model runs the picker and validates the choice."""

    def test_not_requested_is_noop(self) -> None:
        config = Config()
        original = config.model
        _maybe_select_model(_args(select_model=False), config, MagicMock())
        self.assertEqual(config.model, original)

    def test_selection_updates_model(self) -> None:
        config = Config()
        with patch(
            "local_cli.__main__.select_model_interactive",
            return_value="qwen3:8b",
        ):
            _maybe_select_model(_args(select_model=True), config, MagicMock())
        self.assertEqual(config.model, "qwen3:8b")

    def test_invalid_selection_exits(self) -> None:
        config = Config()
        with patch(
            "local_cli.__main__.select_model_interactive",
            return_value="bad name!!",
        ):
            with patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    _maybe_select_model(
                        _args(select_model=True), config, MagicMock(),
                    )


class TestUpdateCheck(unittest.TestCase):
    """Background update check and notice."""

    def test_notice_shown_when_update_available(self) -> None:
        with patch(
            "local_cli.updater.check_for_updates",
            return_value=(True, "v2 available"),
        ):
            thread, result = _start_update_check()
            thread.join(timeout=2)
        with patch("sys.stderr") as stderr:
            _show_update_notice(thread, result)
        written = "".join(c.args[0] for c in stderr.write.call_args_list)
        self.assertIn("Update available", written)

    def test_no_notice_when_up_to_date(self) -> None:
        with patch(
            "local_cli.updater.check_for_updates",
            return_value=(False, "Up to date."),
        ):
            thread, result = _start_update_check()
            thread.join(timeout=2)
        with patch("sys.stderr") as stderr:
            _show_update_notice(thread, result)
        self.assertFalse(stderr.write.called)

    def test_auto_update_installs_when_enabled(self) -> None:
        """With auto_update, an available update is installed, not just shown."""
        with patch(
            "local_cli.updater.check_for_updates",
            return_value=(True, "v2 available"),
        ):
            thread, result = _start_update_check()
            thread.join(timeout=2)
        with patch(
            "local_cli.updater.perform_update",
            return_value=(True, "Updated to v0.11.0"),
        ) as perform:
            with patch("sys.stderr") as stderr:
                _show_update_notice(thread, result, auto_update=True)
        perform.assert_called_once()
        written = "".join(c.args[0] for c in stderr.write.call_args_list)
        self.assertIn("installing automatically", written)
        self.assertIn("Restart", written)

    def test_auto_update_off_only_notifies(self) -> None:
        """Without auto_update, perform_update is never called."""
        with patch(
            "local_cli.updater.check_for_updates",
            return_value=(True, "v2 available"),
        ):
            thread, result = _start_update_check()
            thread.join(timeout=2)
        with patch("local_cli.updater.perform_update") as perform:
            with patch("sys.stderr"):
                _show_update_notice(thread, result, auto_update=False)
        perform.assert_not_called()

    def test_auto_update_failure_does_not_crash(self) -> None:
        """A failing auto-update is caught and reported, not raised."""
        with patch(
            "local_cli.updater.check_for_updates",
            return_value=(True, "v2 available"),
        ):
            thread, result = _start_update_check()
            thread.join(timeout=2)
        with patch(
            "local_cli.updater.perform_update",
            side_effect=RuntimeError("network down"),
        ):
            with patch("sys.stderr") as stderr:
                _show_update_notice(thread, result, auto_update=True)
        written = "".join(c.args[0] for c in stderr.write.call_args_list)
        self.assertIn("Auto-update failed", written)


if __name__ == "__main__":
    unittest.main()
