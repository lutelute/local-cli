"""Tests for TodoWriteTool."""

from local_cli.tools.todo_tool import TodoWriteTool


def test_name_and_description():
    tool = TodoWriteTool()
    assert tool.name == "todo_write"
    assert "task list" in tool.description.lower()


def test_parameters_schema():
    tool = TodoWriteTool()
    p = tool.parameters
    assert p["type"] == "object"
    assert "todos" in p["properties"]
    assert p["required"] == ["todos"]


def test_create_initial_list():
    tool = TodoWriteTool()
    result = tool.execute(todos=[
        {"content": "Task 1", "status": "pending"},
        {"content": "Task 2", "status": "pending"},
        {"content": "Task 3", "status": "pending"},
    ])
    assert "0/3 completed" in result
    assert "[ ] 1. Task 1" in result
    assert "[ ] 2. Task 2" in result
    assert "[ ] 3. Task 3" in result


def test_update_status_to_in_progress():
    tool = TodoWriteTool()
    tool.execute(todos=[
        {"content": "Task 1", "status": "pending"},
        {"content": "Task 2", "status": "pending"},
    ])
    result = tool.execute(todos=[
        {"content": "Task 1", "status": "in_progress"},
        {"content": "Task 2", "status": "pending"},
    ])
    assert "[>] 1. Task 1" in result
    assert "[ ] 2. Task 2" in result


def test_complete_all_tasks():
    tool = TodoWriteTool()
    result = tool.execute(todos=[
        {"content": "Task 1", "status": "completed"},
        {"content": "Task 2", "status": "completed"},
    ])
    assert "2/2 completed" in result
    assert "[x] 1. Task 1" in result
    assert "[x] 2. Task 2" in result
    assert "All tasks completed" in result


def test_warn_multiple_in_progress():
    tool = TodoWriteTool()
    result = tool.execute(todos=[
        {"content": "Task 1", "status": "in_progress"},
        {"content": "Task 2", "status": "in_progress"},
    ])
    assert "2 tasks marked in_progress" in result


def test_invalid_status():
    tool = TodoWriteTool()
    result = tool.execute(todos=[
        {"content": "Task 1", "status": "done"},  # invalid
    ])
    assert result.startswith("Error:")
    assert "status must be one of" in result


def test_empty_content():
    tool = TodoWriteTool()
    result = tool.execute(todos=[
        {"content": "", "status": "pending"},
    ])
    assert result.startswith("Error:")
    assert "content" in result


def test_todos_not_a_list():
    tool = TodoWriteTool()
    result = tool.execute(todos="not a list")
    assert result.startswith("Error:")


def test_todo_item_not_a_dict():
    tool = TodoWriteTool()
    result = tool.execute(todos=["string instead of dict"])
    assert result.startswith("Error:")


def test_state_persists_across_calls():
    tool = TodoWriteTool()
    tool.execute(todos=[
        {"content": "Task 1", "status": "pending"},
        {"content": "Task 2", "status": "pending"},
    ])
    assert len(tool.current_todos) == 2

    tool.execute(todos=[
        {"content": "Task 1", "status": "completed"},
        {"content": "Task 2", "status": "in_progress"},
        {"content": "Task 3", "status": "pending"},
    ])
    assert len(tool.current_todos) == 3
    assert tool.current_todos[0]["status"] == "completed"


def test_state_isolated_per_instance():
    """Each tool instance should have its own state."""
    tool1 = TodoWriteTool()
    tool2 = TodoWriteTool()

    tool1.execute(todos=[{"content": "A", "status": "pending"}])
    assert len(tool1.current_todos) == 1
    assert len(tool2.current_todos) == 0


def test_to_ollama_tool_format():
    tool = TodoWriteTool()
    ollama = tool.to_ollama_tool()
    assert ollama["type"] == "function"
    assert ollama["function"]["name"] == "todo_write"
    assert "parameters" in ollama["function"]


def test_content_stripped():
    tool = TodoWriteTool()
    tool.execute(todos=[
        {"content": "  Task with whitespace  ", "status": "pending"},
    ])
    assert tool.current_todos[0]["content"] == "Task with whitespace"
