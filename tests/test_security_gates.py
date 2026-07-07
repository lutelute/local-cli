"""Tests for the security boundaries around one-shot behaviour.

Audit findings (2026-07-07): the GUI (desktop) path constructed
``BashTool()`` with no confirm callback, so risky commands (sudo,
recursive rm, kill, ...) ran unconfirmed; sub-agents likewise.  The
system prompt had no security section, and injected project-instruction
files carried no authority boundary — a malicious AGENTS.md in a cloned
repo could steer a small model.

These tests pin the fixes: GUI confirm flow (deny on timeout),
sub-agent risky refusal, the SECURITY prompt section, and the
instruction-injection boundary.
"""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from local_cli.project_instructions import build_instruction_message
from local_cli.prompts import build_system_prompt
from local_cli.tools import get_default_tools, get_sub_agent_tools
from tests.test_server import _make_server


class TestSubAgentBashPolicy(unittest.TestCase):
    def _bash(self, tools: list) -> object:
        return next(t for t in tools if t.name == "bash")

    def test_sub_agent_refuses_risky_commands(self) -> None:
        bash = self._bash(get_sub_agent_tools())
        # $((6*7)) would expand to 42 only if the command actually ran;
        # the decline message merely echoes the unexpanded text.
        result = bash.execute(command="sudo echo $((6*7))")
        self.assertIn("Command declined (not run)", result)
        self.assertNotIn("42", result)

    def test_sub_agent_still_runs_normal_commands(self) -> None:
        bash = self._bash(get_sub_agent_tools())
        result = bash.execute(command="echo subagent-ok")
        self.assertIn("subagent-ok", result)

    def test_default_tools_bash_has_no_confirm(self) -> None:
        """The factory stays neutral; frontends wire their own gate."""
        bash = self._bash(get_default_tools())
        self.assertIsNone(bash._confirm)


class TestGuiConfirm(unittest.TestCase):
    def _server(self):
        server = _make_server(provider=None, tools=[])
        server._config = SimpleNamespace(auto_approve=False)
        return server

    def test_auto_approve_skips_the_dialog(self) -> None:
        server = self._server()
        server._config.auto_approve = True
        sent: list[dict] = []
        with patch("local_cli.server._send", side_effect=sent.append):
            self.assertTrue(server._gui_confirm("sudo echo hi"))
        self.assertEqual(sent, [])

    def test_approved_by_gui(self) -> None:
        server = self._server()
        sent: list[dict] = []

        def send_and_approve(msg: dict) -> None:
            sent.append(msg)
            # Simulate the main stdin loop routing the response back.
            server._confirm_result = True
            server._confirm_event.set()

        with patch("local_cli.server._send", side_effect=send_and_approve):
            self.assertTrue(server._gui_confirm("sudo echo hi"))
        (request,) = sent
        self.assertEqual(request["type"], "confirm_request")
        self.assertEqual(request["command"], "sudo echo hi")
        self.assertIn("confirm_id", request)

    def test_denied_by_gui(self) -> None:
        server = self._server()

        def send_and_deny(msg: dict) -> None:
            server._confirm_result = False
            server._confirm_event.set()

        with patch("local_cli.server._send", side_effect=send_and_deny):
            self.assertFalse(server._gui_confirm("kill -9 123"))

    def test_timeout_denies(self) -> None:
        """Nobody watching means the risky command must NOT run."""
        server = self._server()
        with patch("local_cli.server._send"), \
             patch("local_cli.server._CONFIRM_TIMEOUT_S", 0.05):
            self.assertFalse(server._gui_confirm("sudo rm -r build"))

    def test_confirm_response_routing_sets_event(self) -> None:
        """The stdin loop shape: response resolves the waiting thread."""
        server = self._server()
        sent: list[dict] = []
        results: list[bool] = []
        with patch("local_cli.server._send", side_effect=sent.append):
            thread = threading.Thread(
                target=lambda: results.append(
                    server._gui_confirm("sudo echo hi"),
                ),
            )
            thread.start()
            for _ in range(200):
                if sent:
                    break
                threading.Event().wait(0.01)
            # What run() does for req_type == "confirm_response":
            server._confirm_result = True
            server._confirm_event.set()
            thread.join(timeout=5)
        self.assertEqual(results, [True])


class TestPromptSecuritySection(unittest.TestCase):
    def test_system_prompt_has_security_rules(self) -> None:
        prompt = build_system_prompt(get_default_tools())
        self.assertIn("SECURITY:", prompt)
        self.assertIn("DATA", prompt)
        self.assertIn("secrets", prompt)
        self.assertIn("least destructive", prompt)

    def test_instruction_injection_carries_boundary(self) -> None:
        msg = build_instruction_message("AGENTS.md", "use tabs")
        self.assertIn("conventions ONLY", msg["content"])
        self.assertIn("do NOT comply", msg["content"])


if __name__ == "__main__":
    unittest.main()
