"""Tests for local_cli.model_registry module."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from local_cli.model_registry import (
    ModelRegistry,
    RegistryEntry,
    TaskType,
    _MAX_REGISTRY_SIZE,
    _VALID_PROVIDERS,
)


class TestTaskTypeEnum(unittest.TestCase):
    """Tests for TaskType enum."""

    def test_all_task_types_defined(self) -> None:
        """All required task types are defined."""
        expected = {
            "code_generation",
            "planning",
            "review",
            "documentation",
            "general",
        }
        actual = {t.value for t in TaskType}
        self.assertEqual(actual, expected)

    def test_enum_values_are_strings(self) -> None:
        """All TaskType values are lowercase strings."""
        for t in TaskType:
            self.assertIsInstance(t.value, str)
            self.assertEqual(t.value, t.value.lower())

    def test_enum_lookup_by_value(self) -> None:
        """TaskType members can be looked up by string value."""
        self.assertEqual(TaskType("code_generation"), TaskType.CODE_GENERATION)
        self.assertEqual(TaskType("planning"), TaskType.PLANNING)
        self.assertEqual(TaskType("general"), TaskType.GENERAL)


class TestRegistryEntry(unittest.TestCase):
    """Tests for RegistryEntry dataclass."""

    def test_creation(self) -> None:
        """RegistryEntry can be created with all fields."""
        entry = RegistryEntry(provider="ollama", model="qwen3:8b", priority=1)
        self.assertEqual(entry.provider, "ollama")
        self.assertEqual(entry.model, "qwen3:8b")
        self.assertEqual(entry.priority, 1)

    def test_equality(self) -> None:
        """Two entries with the same values are equal."""
        e1 = RegistryEntry(provider="ollama", model="qwen3:8b", priority=1)
        e2 = RegistryEntry(provider="ollama", model="qwen3:8b", priority=1)
        self.assertEqual(e1, e2)

    def test_inequality(self) -> None:
        """Entries with different values are not equal."""
        e1 = RegistryEntry(provider="ollama", model="qwen3:8b", priority=1)
        e2 = RegistryEntry(provider="claude", model="qwen3:8b", priority=1)
        self.assertNotEqual(e1, e2)


class TestModelRegistryInit(unittest.TestCase):
    """Tests for ModelRegistry construction."""

    def test_defaults(self) -> None:
        """Default registry uses ollama/qwen3:8b."""
        registry = ModelRegistry()
        self.assertEqual(registry.default_provider, "ollama")
        self.assertEqual(registry.default_model, "qwen3:8b")

    def test_custom_defaults(self) -> None:
        """Registry accepts custom default provider and model."""
        registry = ModelRegistry(
            default_provider="claude",
            default_model="claude-sonnet-4-5",
        )
        self.assertEqual(registry.default_provider, "claude")
        self.assertEqual(registry.default_model, "claude-sonnet-4-5")

    def test_empty_routes(self) -> None:
        """A new registry has no routes."""
        registry = ModelRegistry()
        self.assertEqual(registry.list_routes(), {})


class TestGetDefault(unittest.TestCase):
    """Tests for ModelRegistry.get_default()."""

    def test_returns_tuple(self) -> None:
        """get_default returns (provider, model) tuple."""
        registry = ModelRegistry()
        result = registry.get_default()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_returns_configured_defaults(self) -> None:
        """get_default returns the configured default pair."""
        registry = ModelRegistry(
            default_provider="claude",
            default_model="claude-haiku-4-5",
        )
        self.assertEqual(registry.get_default(), ("claude", "claude-haiku-4-5"))


class TestGetModelForTask(unittest.TestCase):
    """Tests for ModelRegistry.get_model_for_task()."""

    def test_fallback_to_default(self) -> None:
        """When no routes exist, falls back to default."""
        registry = ModelRegistry()
        result = registry.get_model_for_task(TaskType.CODE_GENERATION)
        self.assertEqual(result, ("ollama", "qwen3:8b"))

    def test_returns_highest_priority(self) -> None:
        """Returns the entry with priority=1 (highest)."""
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.CODE_GENERATION, "claude", "claude-sonnet-4-5", priority=2
        )
        registry.update_task_route(
            TaskType.CODE_GENERATION, "ollama", "qwen3:8b", priority=1
        )
        result = registry.get_model_for_task(TaskType.CODE_GENERATION)
        self.assertEqual(result, ("ollama", "qwen3:8b"))

    def test_different_task_types(self) -> None:
        """Different task types return different models."""
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.CODE_GENERATION, "ollama", "qwen3:8b", priority=1
        )
        registry.update_task_route(
            TaskType.PLANNING, "claude", "claude-sonnet-4-5", priority=1
        )

        code_result = registry.get_model_for_task(TaskType.CODE_GENERATION)
        plan_result = registry.get_model_for_task(TaskType.PLANNING)

        self.assertEqual(code_result, ("ollama", "qwen3:8b"))
        self.assertEqual(plan_result, ("claude", "claude-sonnet-4-5"))

    def test_unmapped_task_falls_back(self) -> None:
        """A task type with no mapping falls back to default."""
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.CODE_GENERATION, "ollama", "qwen3:8b", priority=1
        )
        # REVIEW has no mapping.
        result = registry.get_model_for_task(TaskType.REVIEW)
        self.assertEqual(result, ("ollama", "qwen3:8b"))


class TestGetModelsForTask(unittest.TestCase):
    """Tests for ModelRegistry.get_models_for_task()."""

    def test_returns_all_entries_sorted(self) -> None:
        """Returns all entries sorted by priority."""
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.CODE_GENERATION, "claude", "claude-sonnet-4-5", priority=2
        )
        registry.update_task_route(
            TaskType.CODE_GENERATION, "ollama", "qwen3:8b", priority=1
        )

        result = registry.get_models_for_task(TaskType.CODE_GENERATION)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], ("ollama", "qwen3:8b"))
        self.assertEqual(result[1], ("claude", "claude-sonnet-4-5"))

    def test_fallback_to_default_list(self) -> None:
        """Returns a list with just the default when no mapping exists."""
        registry = ModelRegistry()
        result = registry.get_models_for_task(TaskType.GENERAL)
        self.assertEqual(result, [("ollama", "qwen3:8b")])


class TestUpdateTaskRoute(unittest.TestCase):
    """Tests for ModelRegistry.update_task_route()."""

    def test_add_new_route(self) -> None:
        """Adding a new route creates a new entry."""
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.CODE_GENERATION, "ollama", "qwen3:8b", priority=1
        )
        result = registry.get_model_for_task(TaskType.CODE_GENERATION)
        self.assertEqual(result, ("ollama", "qwen3:8b"))

    def test_update_existing_priority(self) -> None:
        """Updating an existing provider+model entry changes its priority."""
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.CODE_GENERATION, "ollama", "qwen3:8b", priority=1
        )
        registry.update_task_route(
            TaskType.CODE_GENERATION, "claude", "claude-sonnet-4-5", priority=2
        )
        # Update ollama priority to 3 (lower than claude's 2).
        registry.update_task_route(
            TaskType.CODE_GENERATION, "ollama", "qwen3:8b", priority=3
        )

        result = registry.get_model_for_task(TaskType.CODE_GENERATION)
        self.assertEqual(result, ("claude", "claude-sonnet-4-5"))

    def test_multiple_entries_sorted(self) -> None:
        """Multiple entries for one task type are sorted by priority."""
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.REVIEW, "ollama", "qwen3:8b", priority=3
        )
        registry.update_task_route(
            TaskType.REVIEW, "claude", "claude-haiku-4-5", priority=1
        )
        registry.update_task_route(
            TaskType.REVIEW, "claude", "claude-sonnet-4-5", priority=2
        )

        models = registry.get_models_for_task(TaskType.REVIEW)
        self.assertEqual(len(models), 3)
        self.assertEqual(models[0], ("claude", "claude-haiku-4-5"))
        self.assertEqual(models[1], ("claude", "claude-sonnet-4-5"))
        self.assertEqual(models[2], ("ollama", "qwen3:8b"))

    def test_invalid_provider_raises(self) -> None:
        """Invalid provider name raises ValueError."""
        registry = ModelRegistry()
        with self.assertRaises(ValueError) as ctx:
            registry.update_task_route(
                TaskType.GENERAL, "openai", "gpt-4", priority=1
            )
        self.assertIn("Invalid provider", str(ctx.exception))

    def test_invalid_model_name_raises(self) -> None:
        """Invalid model name raises ValueError."""
        registry = ModelRegistry()
        with self.assertRaises(ValueError) as ctx:
            registry.update_task_route(
                TaskType.GENERAL, "ollama", "", priority=1
            )
        self.assertIn("Invalid model name", str(ctx.exception))

    def test_invalid_model_name_injection_raises(self) -> None:
        """Model names with shell injection characters are rejected."""
        registry = ModelRegistry()
        with self.assertRaises(ValueError):
            registry.update_task_route(
                TaskType.GENERAL, "ollama", "model; rm -rf /", priority=1
            )

    def test_invalid_priority_zero_raises(self) -> None:
        """Priority of 0 raises ValueError."""
        registry = ModelRegistry()
        with self.assertRaises(ValueError) as ctx:
            registry.update_task_route(
                TaskType.GENERAL, "ollama", "qwen3:8b", priority=0
            )
        self.assertIn("positive integer", str(ctx.exception))

    def test_invalid_priority_negative_raises(self) -> None:
        """Negative priority raises ValueError."""
        registry = ModelRegistry()
        with self.assertRaises(ValueError):
            registry.update_task_route(
                TaskType.GENERAL, "ollama", "qwen3:8b", priority=-1
            )

    def test_invalid_priority_string_raises(self) -> None:
        """Non-integer priority raises ValueError."""
        registry = ModelRegistry()
        with self.assertRaises(ValueError):
            registry.update_task_route(
                TaskType.GENERAL, "ollama", "qwen3:8b", priority="high"  # type: ignore[arg-type]
            )

    def test_default_priority_is_one(self) -> None:
        """Default priority is 1 when not specified."""
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.GENERAL, "ollama", "qwen3:8b"
        )
        routes = registry.list_routes()
        self.assertEqual(routes["general"][0]["priority"], 1)


class TestRemoveTaskRoute(unittest.TestCase):
    """Tests for ModelRegistry.remove_task_route()."""

    def test_remove_existing(self) -> None:
        """Removing an existing entry returns True."""
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.CODE_GENERATION, "ollama", "qwen3:8b", priority=1
        )
        result = registry.remove_task_route(
            TaskType.CODE_GENERATION, "ollama", "qwen3:8b"
        )
        self.assertTrue(result)
        # Should fall back to default now.
        self.assertEqual(
            registry.get_model_for_task(TaskType.CODE_GENERATION),
            registry.get_default(),
        )

    def test_remove_nonexistent(self) -> None:
        """Removing a non-existent entry returns False."""
        registry = ModelRegistry()
        result = registry.remove_task_route(
            TaskType.CODE_GENERATION, "ollama", "qwen3:8b"
        )
        self.assertFalse(result)

    def test_remove_one_of_multiple(self) -> None:
        """Removing one entry preserves others."""
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.CODE_GENERATION, "ollama", "qwen3:8b", priority=1
        )
        registry.update_task_route(
            TaskType.CODE_GENERATION, "claude", "claude-sonnet-4-5", priority=2
        )
        registry.remove_task_route(
            TaskType.CODE_GENERATION, "ollama", "qwen3:8b"
        )

        models = registry.get_models_for_task(TaskType.CODE_GENERATION)
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0], ("claude", "claude-sonnet-4-5"))


class TestListRoutes(unittest.TestCase):
    """Tests for ModelRegistry.list_routes()."""

    def test_empty_registry(self) -> None:
        """Empty registry returns empty dict."""
        registry = ModelRegistry()
        self.assertEqual(registry.list_routes(), {})

    def test_returns_dict_with_string_keys(self) -> None:
        """Routes dict uses task type string values as keys."""
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.CODE_GENERATION, "ollama", "qwen3:8b", priority=1
        )
        routes = registry.list_routes()
        self.assertIn("code_generation", routes)
        self.assertNotIn(TaskType.CODE_GENERATION, routes)

    def test_entry_format(self) -> None:
        """Each entry has provider, model, and priority keys."""
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.GENERAL, "ollama", "qwen3:8b", priority=1
        )
        routes = registry.list_routes()
        entry = routes["general"][0]
        self.assertEqual(entry["provider"], "ollama")
        self.assertEqual(entry["model"], "qwen3:8b")
        self.assertEqual(entry["priority"], 1)

    def test_multiple_task_types(self) -> None:
        """Multiple task types appear as separate keys."""
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.CODE_GENERATION, "ollama", "qwen3:8b", priority=1
        )
        registry.update_task_route(
            TaskType.PLANNING, "claude", "claude-sonnet-4-5", priority=1
        )
        routes = registry.list_routes()
        self.assertEqual(len(routes), 2)
        self.assertIn("code_generation", routes)
        self.assertIn("planning", routes)


class TestSaveRegistry(unittest.TestCase):
    """Tests for ModelRegistry.save()."""

    def test_save_creates_file(self) -> None:
        """save() creates a JSON file at the specified path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "registry.json")
            registry = ModelRegistry()
            registry.save(path)
            self.assertTrue(os.path.exists(path))

    def test_save_valid_json(self) -> None:
        """Saved file contains valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "registry.json")
            registry = ModelRegistry()
            registry.update_task_route(
                TaskType.CODE_GENERATION, "ollama", "qwen3:8b", priority=1
            )
            registry.save(path)

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["default_provider"], "ollama")
            self.assertEqual(data["default_model"], "qwen3:8b")
            self.assertIn("task_routing", data)

    def test_save_includes_routes(self) -> None:
        """Saved JSON includes all configured routes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "registry.json")
            registry = ModelRegistry()
            registry.update_task_route(
                TaskType.CODE_GENERATION, "ollama", "qwen3:8b", priority=1
            )
            registry.update_task_route(
                TaskType.PLANNING, "claude", "claude-sonnet-4-5", priority=1
            )
            registry.save(path)

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            routing = data["task_routing"]
            self.assertIn("code_generation", routing)
            self.assertIn("planning", routing)

    def test_save_creates_parent_directories(self) -> None:
        """save() creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "registry.json")
            registry = ModelRegistry()
            registry.save(path)
            self.assertTrue(os.path.exists(path))


class TestLoadRegistry(unittest.TestCase):
    """Tests for ModelRegistry.load()."""

    def test_load_basic(self) -> None:
        """Load a basic registry from JSON."""
        data = {
            "default_provider": "ollama",
            "default_model": "qwen3:8b",
            "task_routing": {
                "code_generation": [
                    {"provider": "ollama", "model": "qwen3:8b", "priority": 1},
                ],
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name

        try:
            registry = ModelRegistry.load(path)
            self.assertEqual(registry.default_provider, "ollama")
            self.assertEqual(registry.default_model, "qwen3:8b")
            result = registry.get_model_for_task(TaskType.CODE_GENERATION)
            self.assertEqual(result, ("ollama", "qwen3:8b"))
        finally:
            os.unlink(path)

    def test_load_multiple_routes(self) -> None:
        """Load registry with multiple task types and priorities."""
        data = {
            "default_provider": "ollama",
            "default_model": "qwen3:8b",
            "task_routing": {
                "code_generation": [
                    {"provider": "ollama", "model": "qwen3:8b", "priority": 1},
                    {"provider": "claude", "model": "claude-sonnet-4-5", "priority": 2},
                ],
                "planning": [
                    {"provider": "claude", "model": "claude-sonnet-4-5", "priority": 1},
                    {"provider": "ollama", "model": "qwen3:8b", "priority": 2},
                ],
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name

        try:
            registry = ModelRegistry.load(path)

            code_result = registry.get_model_for_task(TaskType.CODE_GENERATION)
            self.assertEqual(code_result, ("ollama", "qwen3:8b"))

            plan_result = registry.get_model_for_task(TaskType.PLANNING)
            self.assertEqual(plan_result, ("claude", "claude-sonnet-4-5"))

            # Verify priority ordering.
            code_models = registry.get_models_for_task(TaskType.CODE_GENERATION)
            self.assertEqual(code_models[0], ("ollama", "qwen3:8b"))
            self.assertEqual(code_models[1], ("claude", "claude-sonnet-4-5"))
        finally:
            os.unlink(path)

    def test_load_nonexistent_file(self) -> None:
        """Loading a non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            ModelRegistry.load("/tmp/does_not_exist_registry.json")

    def test_load_invalid_json(self) -> None:
        """Loading invalid JSON raises ValueError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not valid json {{{")
            path = f.name

        try:
            with self.assertRaises(ValueError) as ctx:
                ModelRegistry.load(path)
            self.assertIn("Invalid JSON", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_load_non_object_json(self) -> None:
        """Loading JSON that is not an object raises ValueError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump([1, 2, 3], f)
            path = f.name

        try:
            with self.assertRaises(ValueError) as ctx:
                ModelRegistry.load(path)
            self.assertIn("JSON object", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_load_defaults_when_missing_keys(self) -> None:
        """Missing keys use default values."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({}, f)
            path = f.name

        try:
            registry = ModelRegistry.load(path)
            self.assertEqual(registry.default_provider, "ollama")
            self.assertEqual(registry.default_model, "qwen3:8b")
        finally:
            os.unlink(path)

    def test_load_skips_unknown_task_types(self) -> None:
        """Unknown task type keys in routing are silently skipped."""
        data = {
            "default_provider": "ollama",
            "default_model": "qwen3:8b",
            "task_routing": {
                "unknown_task": [
                    {"provider": "ollama", "model": "qwen3:8b", "priority": 1},
                ],
                "code_generation": [
                    {"provider": "ollama", "model": "qwen3:8b", "priority": 1},
                ],
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name

        try:
            registry = ModelRegistry.load(path)
            routes = registry.list_routes()
            self.assertNotIn("unknown_task", routes)
            self.assertIn("code_generation", routes)
        finally:
            os.unlink(path)

    def test_load_skips_invalid_entries(self) -> None:
        """Invalid entries within valid task types are skipped."""
        data = {
            "default_provider": "ollama",
            "default_model": "qwen3:8b",
            "task_routing": {
                "code_generation": [
                    {"provider": "ollama", "model": "qwen3:8b", "priority": 1},
                    {"provider": "invalid_provider", "model": "x", "priority": 1},
                    {"provider": "ollama", "model": "", "priority": 1},
                    {"provider": "ollama", "model": "valid:tag", "priority": 0},
                    {"provider": "ollama", "model": "valid:tag", "priority": -1},
                    "not a dict",
                    {"missing": "fields"},
                ],
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name

        try:
            registry = ModelRegistry.load(path)
            models = registry.get_models_for_task(TaskType.CODE_GENERATION)
            # Only the first valid entry should survive.
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0], ("ollama", "qwen3:8b"))
        finally:
            os.unlink(path)

    def test_load_symlink_rejected(self) -> None:
        """Symlinked registry files are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            real_path = os.path.join(tmpdir, "real.json")
            link_path = os.path.join(tmpdir, "link.json")

            with open(real_path, "w") as f:
                json.dump({"default_provider": "ollama"}, f)
            os.symlink(real_path, link_path)

            with self.assertRaises(ValueError) as ctx:
                ModelRegistry.load(link_path)
            self.assertIn("symlink", str(ctx.exception))

    def test_load_oversized_file_rejected(self) -> None:
        """Registry files exceeding 64KB are rejected."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            # Write more than 64KB.
            f.write("{" + " " * (_MAX_REGISTRY_SIZE + 100) + "}")
            path = f.name

        try:
            with self.assertRaises(ValueError) as ctx:
                ModelRegistry.load(path)
            self.assertIn("too large", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_load_non_dict_task_routing(self) -> None:
        """Non-dict task_routing is handled gracefully."""
        data = {
            "default_provider": "ollama",
            "default_model": "qwen3:8b",
            "task_routing": "not a dict",
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name

        try:
            registry = ModelRegistry.load(path)
            self.assertEqual(registry.list_routes(), {})
        finally:
            os.unlink(path)

    def test_load_non_list_entries(self) -> None:
        """Non-list entries for a task type are skipped."""
        data = {
            "default_provider": "ollama",
            "default_model": "qwen3:8b",
            "task_routing": {
                "code_generation": "not a list",
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name

        try:
            registry = ModelRegistry.load(path)
            self.assertEqual(registry.list_routes(), {})
        finally:
            os.unlink(path)

    def test_load_sorts_by_priority(self) -> None:
        """Loaded entries are sorted by priority (ascending)."""
        data = {
            "default_provider": "ollama",
            "default_model": "qwen3:8b",
            "task_routing": {
                "code_generation": [
                    {"provider": "claude", "model": "claude-sonnet-4-5", "priority": 3},
                    {"provider": "ollama", "model": "qwen3:8b", "priority": 1},
                    {"provider": "claude", "model": "claude-haiku-4-5", "priority": 2},
                ],
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name

        try:
            registry = ModelRegistry.load(path)
            models = registry.get_models_for_task(TaskType.CODE_GENERATION)
            self.assertEqual(models[0], ("ollama", "qwen3:8b"))
            self.assertEqual(models[1], ("claude", "claude-haiku-4-5"))
            self.assertEqual(models[2], ("claude", "claude-sonnet-4-5"))
        finally:
            os.unlink(path)


class TestSaveLoadRoundTrip(unittest.TestCase):
    """Integration tests for save/load round-trip."""

    def test_round_trip_preserves_data(self) -> None:
        """Save then load preserves all registry data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "registry.json")

            original = ModelRegistry(
                default_provider="claude",
                default_model="claude-sonnet-4-5",
            )
            original.update_task_route(
                TaskType.CODE_GENERATION, "ollama", "qwen3:8b", priority=1
            )
            original.update_task_route(
                TaskType.CODE_GENERATION, "claude", "claude-sonnet-4-5", priority=2
            )
            original.update_task_route(
                TaskType.PLANNING, "claude", "claude-sonnet-4-5", priority=1
            )
            original.update_task_route(
                TaskType.REVIEW, "claude", "claude-haiku-4-5", priority=1
            )
            original.update_task_route(
                TaskType.DOCUMENTATION, "ollama", "qwen3:8b", priority=1
            )
            original.update_task_route(
                TaskType.GENERAL, "ollama", "qwen3:8b", priority=1
            )
            original.save(path)

            loaded = ModelRegistry.load(path)

            self.assertEqual(loaded.default_provider, "claude")
            self.assertEqual(loaded.default_model, "claude-sonnet-4-5")
            self.assertEqual(loaded.list_routes(), original.list_routes())

    def test_round_trip_empty_registry(self) -> None:
        """Empty registry survives save/load round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "registry.json")

            original = ModelRegistry()
            original.save(path)

            loaded = ModelRegistry.load(path)
            self.assertEqual(loaded.default_provider, "ollama")
            self.assertEqual(loaded.default_model, "qwen3:8b")
            self.assertEqual(loaded.list_routes(), {})

    def test_round_trip_with_all_task_types(self) -> None:
        """All task types survive round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "registry.json")

            original = ModelRegistry()
            for task_type in TaskType:
                original.update_task_route(
                    task_type, "ollama", "qwen3:8b", priority=1
                )
            original.save(path)

            loaded = ModelRegistry.load(path)
            for task_type in TaskType:
                result = loaded.get_model_for_task(task_type)
                self.assertEqual(result, ("ollama", "qwen3:8b"))


class TestRegistrySpecFormat(unittest.TestCase):
    """Tests that the registry matches the spec format from the spec."""

    def test_full_spec_format_loads(self) -> None:
        """The exact JSON format from the spec loads correctly."""
        spec_json = {
            "default_provider": "ollama",
            "default_model": "qwen3:8b",
            "task_routing": {
                "code_generation": [
                    {"provider": "ollama", "model": "qwen3:8b", "priority": 1},
                    {"provider": "claude", "model": "claude-sonnet-4-5", "priority": 2},
                ],
                "planning": [
                    {"provider": "claude", "model": "claude-sonnet-4-5", "priority": 1},
                    {"provider": "ollama", "model": "qwen3:8b", "priority": 2},
                ],
                "review": [
                    {"provider": "claude", "model": "claude-haiku-4-5", "priority": 1},
                ],
                "documentation": [
                    {"provider": "ollama", "model": "qwen3:8b", "priority": 1},
                ],
                "general": [
                    {"provider": "ollama", "model": "qwen3:8b", "priority": 1},
                ],
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(spec_json, f)
            path = f.name

        try:
            registry = ModelRegistry.load(path)

            # Verify routing.
            self.assertEqual(
                registry.get_model_for_task(TaskType.CODE_GENERATION),
                ("ollama", "qwen3:8b"),
            )
            self.assertEqual(
                registry.get_model_for_task(TaskType.PLANNING),
                ("claude", "claude-sonnet-4-5"),
            )
            self.assertEqual(
                registry.get_model_for_task(TaskType.REVIEW),
                ("claude", "claude-haiku-4-5"),
            )
            self.assertEqual(
                registry.get_model_for_task(TaskType.DOCUMENTATION),
                ("ollama", "qwen3:8b"),
            )
            self.assertEqual(
                registry.get_model_for_task(TaskType.GENERAL),
                ("ollama", "qwen3:8b"),
            )
        finally:
            os.unlink(path)


class TestValidProviders(unittest.TestCase):
    """Tests for valid provider constants."""

    def test_valid_providers_include_ollama(self) -> None:
        """ollama is a valid provider."""
        self.assertIn("ollama", _VALID_PROVIDERS)

    def test_valid_providers_include_claude(self) -> None:
        """claude is a valid provider."""
        self.assertIn("claude", _VALID_PROVIDERS)

    def test_valid_providers_is_frozenset(self) -> None:
        """_VALID_PROVIDERS is immutable."""
        self.assertIsInstance(_VALID_PROVIDERS, frozenset)


class TestModelNameValidation(unittest.TestCase):
    """Tests that model name validation is applied."""

    def test_valid_model_names_accepted(self) -> None:
        """Valid model names are accepted in update_task_route."""
        valid_names = [
            "qwen3:8b",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "all-minilm",
            "library/model:latest",
            "phi4-mini",
        ]
        for i, name in enumerate(valid_names, start=1):
            # Use a fresh registry for each to avoid priority conflicts.
            registry = ModelRegistry()
            registry.update_task_route(
                TaskType.GENERAL, "ollama", name, priority=1
            )
            result = registry.get_model_for_task(TaskType.GENERAL)
            self.assertEqual(result[1], name, f"Failed for model: {name}")

    def test_invalid_model_names_rejected(self) -> None:
        """Invalid model names are rejected."""
        registry = ModelRegistry()
        invalid_names = [
            "",
            " spaces ",
            "model;injection",
            "$(whoami)",
            "`command`",
        ]
        for name in invalid_names:
            with self.assertRaises(ValueError, msg=f"Should reject: {name!r}"):
                registry.update_task_route(
                    TaskType.GENERAL, "ollama", name, priority=1
                )


class TestRuntimeUpdates(unittest.TestCase):
    """Tests for runtime modification of registry."""

    def test_add_then_query(self) -> None:
        """Adding a route at runtime is immediately queryable."""
        registry = ModelRegistry()
        self.assertEqual(
            registry.get_model_for_task(TaskType.CODE_GENERATION),
            ("ollama", "qwen3:8b"),
        )

        registry.update_task_route(
            TaskType.CODE_GENERATION, "claude", "claude-sonnet-4-5", priority=1
        )
        self.assertEqual(
            registry.get_model_for_task(TaskType.CODE_GENERATION),
            ("claude", "claude-sonnet-4-5"),
        )

    def test_update_priority_changes_result(self) -> None:
        """Changing priority at runtime changes the top result."""
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.GENERAL, "ollama", "qwen3:8b", priority=1
        )
        registry.update_task_route(
            TaskType.GENERAL, "claude", "claude-sonnet-4-5", priority=2
        )
        self.assertEqual(
            registry.get_model_for_task(TaskType.GENERAL),
            ("ollama", "qwen3:8b"),
        )

        # Swap priorities.
        registry.update_task_route(
            TaskType.GENERAL, "ollama", "qwen3:8b", priority=2
        )
        registry.update_task_route(
            TaskType.GENERAL, "claude", "claude-sonnet-4-5", priority=1
        )
        self.assertEqual(
            registry.get_model_for_task(TaskType.GENERAL),
            ("claude", "claude-sonnet-4-5"),
        )

    def test_remove_then_fallback(self) -> None:
        """After removing all routes, task falls back to default."""
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.REVIEW, "claude", "claude-haiku-4-5", priority=1
        )
        registry.remove_task_route(
            TaskType.REVIEW, "claude", "claude-haiku-4-5"
        )
        self.assertEqual(
            registry.get_model_for_task(TaskType.REVIEW),
            registry.get_default(),
        )


if __name__ == "__main__":
    unittest.main()
