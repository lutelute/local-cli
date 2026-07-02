"""Tests for local_cli.prompts — shared prompt and injection builders."""

import unittest

from local_cli.prompts import build_skill_messages, build_system_prompt
from local_cli.tools.base import Tool


class _DummyTool(Tool):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "A dummy tool."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs: object) -> str:
        return "ok"


class _FakeSkill:
    def __init__(self, name: str, content: str) -> None:
        self.name = name
        self.content = content


class _FakeSkillsLoader:
    def __init__(self, skills: list[_FakeSkill]) -> None:
        self._skills = skills
        self.queries: list[str] = []

    def get_matching_skills(self, text: str) -> list[_FakeSkill]:
        self.queries.append(text)
        return self._skills


class _BrokenSkillsLoader:
    def get_matching_skills(self, text: str) -> list:
        raise RuntimeError("skills directory unreadable")


class TestBuildSystemPrompt(unittest.TestCase):
    """Tests for the shared system prompt builder."""

    def test_lists_tools(self) -> None:
        prompt = build_system_prompt([_DummyTool()])
        self.assertIn("dummy: A dummy tool.", prompt)
        self.assertIn("AVAILABLE TOOLS", prompt)

    def test_main_role_has_no_sub_agent_section(self) -> None:
        prompt = build_system_prompt([_DummyTool()])
        self.assertNotIn("SUB-AGENT MODE", prompt)

    def test_sub_agent_role_appends_section(self) -> None:
        prompt = build_system_prompt([_DummyTool()], role="sub_agent")
        self.assertIn("SUB-AGENT MODE", prompt)
        self.assertIn("returned to the main agent", prompt)
        # The full main prompt is still present (not the old stub).
        self.assertIn("AVAILABLE TOOLS", prompt)
        self.assertIn("TASK TRACKING", prompt)


class TestBuildSkillMessages(unittest.TestCase):
    """Tests for the shared skill-injection builder."""

    def test_matching_skills_become_system_messages(self) -> None:
        loader = _FakeSkillsLoader([
            _FakeSkill("django-api", "Use DRF serializers."),
            _FakeSkill("review", "Check for SQL injection."),
        ])
        messages = build_skill_messages(loader, "build a django REST API")

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("--- SKILL: django-api ---", messages[0]["content"])
        self.assertIn("Use DRF serializers.", messages[0]["content"])
        self.assertIn("--- END SKILL ---", messages[0]["content"])
        self.assertEqual(loader.queries, ["build a django REST API"])

    def test_none_loader_returns_empty(self) -> None:
        self.assertEqual(build_skill_messages(None, "anything"), [])

    def test_no_matches_returns_empty(self) -> None:
        loader = _FakeSkillsLoader([])
        self.assertEqual(build_skill_messages(loader, "hello"), [])

    def test_loader_failure_is_swallowed(self) -> None:
        self.assertEqual(
            build_skill_messages(_BrokenSkillsLoader(), "hello"), [],
        )


if __name__ == "__main__":
    unittest.main()
