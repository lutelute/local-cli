"""Todo tool for structured task tracking.

Provides a TodoWrite tool that lets the agent maintain a persistent
checklist of tasks across the conversation. This helps smaller models
(and larger ones too) stay on track for multi-step tasks by providing
explicit state tracking.

Inspired by Claude Code's TodoWrite pattern. The todo list is stored
in memory per session and persists across agent loop iterations.

Usage from the agent:

    # Create initial todo list
    todo_write(todos=[
        {"content": "Create game 1: tetris", "status": "pending"},
        {"content": "Create game 2: snake", "status": "pending"},
        ...
    ])

    # Update status as work progresses
    todo_write(todos=[
        {"content": "Create game 1: tetris", "status": "completed"},
        {"content": "Create game 2: snake", "status": "in_progress"},
        ...
    ])
"""

from typing import Any

from local_cli.tools.base import Tool


_VALID_STATUSES = {"pending", "in_progress", "completed"}


class TodoWriteTool(Tool):
    """Maintain a structured todo list for multi-step tasks.

    The todo list is shared across all calls to this tool within a
    single session. Each call replaces the entire list, so the agent
    must pass the full updated list every time.
    """

    def __init__(self) -> None:
        # Session-scoped state (one instance per JsonLineServer or
        # web_monitor session).
        self._todos: list[dict[str, str]] = []

    @property
    def cacheable(self) -> bool:
        # Writing todos has side effects (updates internal state).
        return False

    @property
    def name(self) -> str:
        return "todo_write"

    @property
    def description(self) -> str:
        return (
            "Create and maintain a structured task list for multi-step work. "
            "Use this tool when a task has 3+ distinct steps or when the user "
            "asks for multiple deliverables (e.g., 'make 10 games', 'refactor "
            "these 5 files'). Pass the entire updated list each time. "
            "Each todo has a 'content' (task description) and 'status' "
            "(one of: pending, in_progress, completed). Mark exactly one "
            "todo as 'in_progress' at a time, and mark completed ones "
            "immediately after finishing them. This keeps you organized "
            "and prevents forgetting or duplicating work."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": (
                        "The complete updated todo list. Each item must "
                        "have 'content' (string) and 'status' (one of: "
                        "pending, in_progress, completed)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The task description.",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "Current status of this task.",
                            },
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["todos"],
        }

    def execute(self, **kwargs: Any) -> str:
        """Update the todo list and return a formatted summary.

        Validates the input and replaces the internal todo list with
        the new one. Returns a human-readable summary showing progress
        and the current in-progress task.
        """
        todos = kwargs.get("todos")
        if not isinstance(todos, list):
            return "Error: 'todos' must be a list."

        # Validate and normalize each todo.
        validated: list[dict[str, str]] = []
        for i, t in enumerate(todos):
            if not isinstance(t, dict):
                return f"Error: todo[{i}] must be an object."
            content = t.get("content", "")
            status = t.get("status", "")
            if not isinstance(content, str) or not content.strip():
                return f"Error: todo[{i}].content must be a non-empty string."
            if status not in _VALID_STATUSES:
                return (
                    f"Error: todo[{i}].status must be one of "
                    f"{sorted(_VALID_STATUSES)}, got {status!r}."
                )
            validated.append({"content": content.strip(), "status": status})

        # Count in-progress — warn if more than one (not enforced).
        in_progress = sum(1 for t in validated if t["status"] == "in_progress")

        self._todos = validated

        # Build summary.
        total = len(validated)
        completed = sum(1 for t in validated if t["status"] == "completed")
        pending = sum(1 for t in validated if t["status"] == "pending")

        lines = [f"Todo list updated ({completed}/{total} completed):"]
        for i, t in enumerate(validated, 1):
            icon = {
                "completed": "[x]",
                "in_progress": "[>]",
                "pending": "[ ]",
            }[t["status"]]
            lines.append(f"  {icon} {i}. {t['content']}")

        if in_progress > 1:
            lines.append(
                f"\nWarning: {in_progress} tasks marked in_progress. "
                "Only one task should be in_progress at a time."
            )

        if pending == 0 and completed == total and total > 0:
            lines.append("\nAll tasks completed.")

        return "\n".join(lines)

    @property
    def current_todos(self) -> list[dict[str, str]]:
        """Return a copy of the current todo list (for external inspection)."""
        return list(self._todos)
