"""Tests for local_cli.orchestrator module."""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from local_cli.config import Config
from local_cli.model_registry import ModelRegistry, TaskType
from local_cli.orchestrator import Orchestrator
from local_cli.providers.base import (
    LLMProvider,
    ProviderConnectionError,
    ProviderRequestError,
    ProviderStreamError,
)


def _make_config(**overrides: object) -> Config:
    """Create a Config with defaults suitable for testing.

    All tests use a non-existent config file to avoid loading any
    real configuration from the filesystem.  Uses ``types.SimpleNamespace``
    so that ``vars()`` correctly returns the attributes.
    """
    namespace = types.SimpleNamespace(
        model=overrides.get("model", "qwen3:8b"),
        provider=overrides.get("provider", "ollama"),
        orchestrator_model=overrides.get("orchestrator_model", ""),
        ollama_host=overrides.get("ollama_host", "http://localhost:11434"),
    )
    return Config(cli_args=namespace, config_file="/dev/null")


def _make_mock_provider(name: str = "ollama") -> MagicMock:
    """Create a mock LLMProvider."""
    provider = MagicMock(spec=LLMProvider)
    provider.name = name
    return provider


class TestOrchestratorInit(unittest.TestCase):
    """Tests for Orchestrator construction."""

    def test_accepts_config(self) -> None:
        """Orchestrator can be created with a Config."""
        config = _make_config()
        orch = Orchestrator(config)
        self.assertIsNotNone(orch)

    def test_active_provider_from_config(self) -> None:
        """Active provider name comes from config.provider."""
        config = _make_config(provider="claude")
        orch = Orchestrator(config)
        self.assertEqual(orch.get_active_provider_name(), "claude")

    def test_default_brain_model_from_config_model(self) -> None:
        """Brain model defaults to config.model when orchestrator_model is empty."""
        config = _make_config(model="qwen3:8b", orchestrator_model="")
        orch = Orchestrator(config)
        self.assertEqual(orch.get_brain_model(), "qwen3:8b")

    def test_brain_model_from_orchestrator_model(self) -> None:
        """Brain model uses orchestrator_model when set."""
        config = _make_config(
            model="qwen3:8b",
            orchestrator_model="claude-sonnet-4-5",
        )
        orch = Orchestrator(config)
        self.assertEqual(orch.get_brain_model(), "claude-sonnet-4-5")

    def test_accepts_optional_registry(self) -> None:
        """Orchestrator accepts an optional ModelRegistry."""
        config = _make_config()
        registry = ModelRegistry()
        orch = Orchestrator(config, registry=registry)
        self.assertIs(orch.registry, registry)

    def test_no_registry_by_default(self) -> None:
        """Orchestrator has no registry by default."""
        config = _make_config()
        orch = Orchestrator(config)
        self.assertIsNone(orch.registry)

    def test_empty_provider_cache(self) -> None:
        """Provider cache starts empty."""
        config = _make_config()
        orch = Orchestrator(config)
        self.assertEqual(orch._providers, {})


class TestGetActiveProviderName(unittest.TestCase):
    """Tests for Orchestrator.get_active_provider_name()."""

    def test_returns_default_provider(self) -> None:
        """Returns the config default provider name."""
        config = _make_config(provider="ollama")
        orch = Orchestrator(config)
        self.assertEqual(orch.get_active_provider_name(), "ollama")

    def test_reflects_switch(self) -> None:
        """Returns the switched provider name after switch_provider."""
        config = _make_config(provider="ollama")
        orch = Orchestrator(config)

        mock_provider = _make_mock_provider("claude")
        orch._providers["claude"] = mock_provider

        orch.switch_provider("claude")
        self.assertEqual(orch.get_active_provider_name(), "claude")


class TestGetProvider(unittest.TestCase):
    """Tests for Orchestrator.get_provider()."""

    def test_creates_ollama_provider(self) -> None:
        """get_provider('ollama') creates an OllamaProvider."""
        config = _make_config()
        orch = Orchestrator(config)

        with patch("local_cli.providers.get_provider") as mock_get:
            mock_prov = _make_mock_provider("ollama")
            mock_get.return_value = mock_prov

            result = orch.get_provider("ollama")

            self.assertIs(result, mock_prov)
            mock_get.assert_called_once_with(
                "ollama", base_url="http://localhost:11434"
            )

    def test_creates_claude_provider(self) -> None:
        """get_provider('claude') creates a ClaudeProvider."""
        config = _make_config()
        orch = Orchestrator(config)

        with patch("local_cli.providers.get_provider") as mock_get:
            mock_prov = _make_mock_provider("claude")
            mock_get.return_value = mock_prov

            result = orch.get_provider("claude")

            self.assertIs(result, mock_prov)
            mock_get.assert_called_once_with("claude")

    def test_caches_provider(self) -> None:
        """Provider instances are cached by name."""
        config = _make_config()
        orch = Orchestrator(config)

        with patch("local_cli.providers.get_provider") as mock_get:
            mock_prov = _make_mock_provider("ollama")
            mock_get.return_value = mock_prov

            first = orch.get_provider("ollama")
            second = orch.get_provider("ollama")

            self.assertIs(first, second)
            # Factory called only once.
            mock_get.assert_called_once()

    def test_different_providers_cached_separately(self) -> None:
        """Different provider names produce different cached instances."""
        config = _make_config()
        orch = Orchestrator(config)

        mock_ollama = _make_mock_provider("ollama")
        mock_claude = _make_mock_provider("claude")

        orch._providers["ollama"] = mock_ollama
        orch._providers["claude"] = mock_claude

        self.assertIs(orch.get_provider("ollama"), mock_ollama)
        self.assertIs(orch.get_provider("claude"), mock_claude)

    def test_none_returns_active_provider(self) -> None:
        """get_provider(None) returns the active provider."""
        config = _make_config(provider="ollama")
        orch = Orchestrator(config)

        mock_prov = _make_mock_provider("ollama")
        orch._providers["ollama"] = mock_prov

        result = orch.get_provider(None)
        self.assertIs(result, mock_prov)

    def test_unknown_provider_raises(self) -> None:
        """Unknown provider name raises ValueError."""
        config = _make_config()
        orch = Orchestrator(config)

        with self.assertRaises(ValueError) as ctx:
            orch.get_provider("openai")
        self.assertIn("Unknown provider", str(ctx.exception))


class TestGetActiveProvider(unittest.TestCase):
    """Tests for Orchestrator.get_active_provider()."""

    def test_returns_active_provider_instance(self) -> None:
        """get_active_provider returns the active provider."""
        config = _make_config(provider="ollama")
        orch = Orchestrator(config)

        mock_prov = _make_mock_provider("ollama")
        orch._providers["ollama"] = mock_prov

        result = orch.get_active_provider()
        self.assertIs(result, mock_prov)

    def test_creates_provider_if_not_cached(self) -> None:
        """get_active_provider creates the provider if not cached."""
        config = _make_config(provider="ollama")
        orch = Orchestrator(config)

        with patch("local_cli.providers.get_provider") as mock_get:
            mock_prov = _make_mock_provider("ollama")
            mock_get.return_value = mock_prov

            result = orch.get_active_provider()

            self.assertIs(result, mock_prov)


class TestSwitchProvider(unittest.TestCase):
    """Tests for Orchestrator.switch_provider()."""

    def test_switch_changes_active(self) -> None:
        """switch_provider changes the active provider name."""
        config = _make_config(provider="ollama")
        orch = Orchestrator(config)

        mock_prov = _make_mock_provider("claude")
        orch._providers["claude"] = mock_prov

        orch.switch_provider("claude")
        self.assertEqual(orch.get_active_provider_name(), "claude")

    def test_switch_returns_new_provider(self) -> None:
        """switch_provider returns the new provider instance."""
        config = _make_config(provider="ollama")
        orch = Orchestrator(config)

        mock_prov = _make_mock_provider("claude")
        orch._providers["claude"] = mock_prov

        result = orch.switch_provider("claude")
        self.assertIs(result, mock_prov)

    def test_switch_creates_provider_if_needed(self) -> None:
        """switch_provider creates the provider if not cached."""
        config = _make_config(provider="ollama")
        orch = Orchestrator(config)

        with patch("local_cli.providers.get_provider") as mock_get:
            mock_prov = _make_mock_provider("claude")
            mock_get.return_value = mock_prov

            result = orch.switch_provider("claude")

            self.assertIs(result, mock_prov)
            self.assertEqual(orch.get_active_provider_name(), "claude")

    def test_switch_to_same_provider(self) -> None:
        """Switching to the already-active provider is a no-op."""
        config = _make_config(provider="ollama")
        orch = Orchestrator(config)

        mock_prov = _make_mock_provider("ollama")
        orch._providers["ollama"] = mock_prov

        result = orch.switch_provider("ollama")
        self.assertIs(result, mock_prov)
        self.assertEqual(orch.get_active_provider_name(), "ollama")

    def test_switch_failure_falls_back(self) -> None:
        """On switch failure, falls back to the other provider."""
        config = _make_config(provider="ollama")
        orch = Orchestrator(config)

        # Claude fails, ollama succeeds.
        mock_ollama = _make_mock_provider("ollama")
        orch._providers["ollama"] = mock_ollama

        with patch("local_cli.providers.get_provider") as mock_get:
            mock_get.side_effect = ValueError("No API key")

            result = orch.switch_provider("claude")

            # Should fall back to ollama.
            self.assertIs(result, mock_ollama)
            self.assertEqual(orch.get_active_provider_name(), "ollama")

    def test_switch_failure_warns_on_stderr(self) -> None:
        """On switch failure, a warning is printed to stderr."""
        config = _make_config(provider="ollama")
        orch = Orchestrator(config)

        mock_ollama = _make_mock_provider("ollama")
        orch._providers["ollama"] = mock_ollama

        with patch("local_cli.providers.get_provider") as mock_get:
            mock_get.side_effect = ValueError("No API key")

            with patch("sys.stderr") as mock_stderr:
                orch.switch_provider("claude")
                # Verify warnings were written.
                self.assertTrue(mock_stderr.write.called)

    def test_switch_both_fail_raises(self) -> None:
        """If both providers fail, raises ValueError."""
        config = _make_config(provider="ollama")
        orch = Orchestrator(config)

        with patch("local_cli.providers.get_provider") as mock_get:
            mock_get.side_effect = ValueError("No providers available")

            with self.assertRaises(ValueError) as ctx:
                orch.switch_provider("claude")
            self.assertIn("Cannot switch", str(ctx.exception))
            self.assertIn("fallback", str(ctx.exception))

    def test_switch_from_claude_fallback_to_ollama(self) -> None:
        """When switching from ollama to claude fails, falls back to ollama."""
        config = _make_config(provider="ollama")
        orch = Orchestrator(config)

        mock_ollama = _make_mock_provider("ollama")
        orch._providers["ollama"] = mock_ollama

        with patch("local_cli.providers.get_provider") as mock_get:
            mock_get.side_effect = ValueError("Claude API key missing")

            result = orch.switch_provider("claude")

            self.assertEqual(orch.get_active_provider_name(), "ollama")
            self.assertIs(result, mock_ollama)

    def test_switch_from_ollama_fallback_to_claude(self) -> None:
        """When switching to ollama fails, falls back to claude."""
        config = _make_config(provider="claude")
        orch = Orchestrator(config)

        mock_claude = _make_mock_provider("claude")
        orch._providers["claude"] = mock_claude

        with patch("local_cli.providers.get_provider") as mock_get:
            mock_get.side_effect = ValueError("Ollama not available")

            result = orch.switch_provider("ollama")

            self.assertEqual(orch.get_active_provider_name(), "claude")
            self.assertIs(result, mock_claude)


class TestBrainModel(unittest.TestCase):
    """Tests for brain model management."""

    def test_get_brain_model_default(self) -> None:
        """Brain model defaults to config.model."""
        config = _make_config(model="qwen3:8b")
        orch = Orchestrator(config)
        self.assertEqual(orch.get_brain_model(), "qwen3:8b")

    def test_set_brain_model(self) -> None:
        """set_brain_model updates the brain model."""
        config = _make_config()
        orch = Orchestrator(config)

        orch.set_brain_model("claude-sonnet-4-5")
        self.assertEqual(orch.get_brain_model(), "claude-sonnet-4-5")

    def test_set_brain_model_overwrites(self) -> None:
        """set_brain_model replaces the previous brain model."""
        config = _make_config(orchestrator_model="model-a")
        orch = Orchestrator(config)
        self.assertEqual(orch.get_brain_model(), "model-a")

        orch.set_brain_model("model-b")
        self.assertEqual(orch.get_brain_model(), "model-b")

    def test_set_brain_model_empty_raises(self) -> None:
        """set_brain_model rejects empty model names."""
        config = _make_config()
        orch = Orchestrator(config)

        with self.assertRaises(ValueError) as ctx:
            orch.set_brain_model("")
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_get_brain_model_after_set(self) -> None:
        """get_brain_model reflects the most recent set_brain_model."""
        config = _make_config()
        orch = Orchestrator(config)

        orch.set_brain_model("phi4-mini")
        self.assertEqual(orch.get_brain_model(), "phi4-mini")


class TestGetModelForTask(unittest.TestCase):
    """Tests for Orchestrator.get_model_for_task()."""

    def test_with_registry(self) -> None:
        """Uses registry when available."""
        config = _make_config()
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.CODE_GENERATION, "ollama", "qwen3:8b", priority=1
        )
        orch = Orchestrator(config, registry=registry)

        result = orch.get_model_for_task(TaskType.CODE_GENERATION)
        self.assertEqual(result, ("ollama", "qwen3:8b"))

    def test_with_registry_different_tasks(self) -> None:
        """Different task types return different models from registry."""
        config = _make_config()
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.CODE_GENERATION, "ollama", "qwen3:8b", priority=1
        )
        registry.update_task_route(
            TaskType.PLANNING, "claude", "claude-sonnet-4-5", priority=1
        )
        orch = Orchestrator(config, registry=registry)

        code_result = orch.get_model_for_task(TaskType.CODE_GENERATION)
        plan_result = orch.get_model_for_task(TaskType.PLANNING)

        self.assertEqual(code_result, ("ollama", "qwen3:8b"))
        self.assertEqual(plan_result, ("claude", "claude-sonnet-4-5"))

    def test_without_registry_uses_config(self) -> None:
        """Without registry, falls back to active provider + config model."""
        config = _make_config(provider="ollama", model="qwen3:8b")
        orch = Orchestrator(config)

        result = orch.get_model_for_task(TaskType.CODE_GENERATION)
        self.assertEqual(result, ("ollama", "qwen3:8b"))

    def test_without_registry_reflects_provider_switch(self) -> None:
        """Without registry, task routing reflects provider switch."""
        config = _make_config(provider="ollama", model="qwen3:8b")
        orch = Orchestrator(config)

        # Pre-populate provider cache.
        mock_claude = _make_mock_provider("claude")
        orch._providers["claude"] = mock_claude

        orch.switch_provider("claude")
        result = orch.get_model_for_task(TaskType.PLANNING)
        self.assertEqual(result, ("claude", "qwen3:8b"))

    def test_registry_fallback_to_default(self) -> None:
        """Registry with no mapping falls back to registry default."""
        config = _make_config()
        registry = ModelRegistry(
            default_provider="ollama",
            default_model="qwen3:8b",
        )
        orch = Orchestrator(config, registry=registry)

        # No routes configured -> falls back to registry default.
        result = orch.get_model_for_task(TaskType.REVIEW)
        self.assertEqual(result, ("ollama", "qwen3:8b"))


class TestGetProviderForTask(unittest.TestCase):
    """Tests for Orchestrator.get_provider_for_task()."""

    def test_returns_provider_and_model(self) -> None:
        """get_provider_for_task returns (provider, model) tuple."""
        config = _make_config()
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.CODE_GENERATION, "ollama", "qwen3:8b", priority=1
        )
        orch = Orchestrator(config, registry=registry)

        mock_prov = _make_mock_provider("ollama")
        orch._providers["ollama"] = mock_prov

        provider, model = orch.get_provider_for_task(TaskType.CODE_GENERATION)
        self.assertIs(provider, mock_prov)
        self.assertEqual(model, "qwen3:8b")

    def test_fallback_on_provider_failure(self) -> None:
        """Falls back to active provider when preferred provider fails."""
        config = _make_config(provider="ollama", model="qwen3:8b")
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.PLANNING, "claude", "claude-sonnet-4-5", priority=1
        )
        orch = Orchestrator(config, registry=registry)

        # Ollama is available but Claude is not.
        mock_ollama = _make_mock_provider("ollama")
        orch._providers["ollama"] = mock_ollama

        with patch("local_cli.providers.get_provider") as mock_get:
            mock_get.side_effect = ValueError("No API key")

            provider, model = orch.get_provider_for_task(TaskType.PLANNING)

            # Should fall back to active provider (ollama).
            self.assertIs(provider, mock_ollama)
            self.assertEqual(model, "qwen3:8b")

    def test_fallback_warns_on_stderr(self) -> None:
        """Fallback prints a warning to stderr."""
        config = _make_config(provider="ollama", model="qwen3:8b")
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.PLANNING, "claude", "claude-sonnet-4-5", priority=1
        )
        orch = Orchestrator(config, registry=registry)

        mock_ollama = _make_mock_provider("ollama")
        orch._providers["ollama"] = mock_ollama

        with patch("local_cli.providers.get_provider") as mock_get:
            mock_get.side_effect = ValueError("No API key")

            with patch("sys.stderr") as mock_stderr:
                orch.get_provider_for_task(TaskType.PLANNING)
                self.assertTrue(mock_stderr.write.called)

    def test_no_registry_uses_active_provider(self) -> None:
        """Without registry, uses active provider."""
        config = _make_config(provider="ollama", model="qwen3:8b")
        orch = Orchestrator(config)

        mock_prov = _make_mock_provider("ollama")
        orch._providers["ollama"] = mock_prov

        provider, model = orch.get_provider_for_task(TaskType.GENERAL)
        self.assertIs(provider, mock_prov)
        self.assertEqual(model, "qwen3:8b")


class TestRegistryProperty(unittest.TestCase):
    """Tests for Orchestrator.registry property."""

    def test_get_registry(self) -> None:
        """registry property returns the configured registry."""
        config = _make_config()
        registry = ModelRegistry()
        orch = Orchestrator(config, registry=registry)
        self.assertIs(orch.registry, registry)

    def test_get_registry_none(self) -> None:
        """registry property returns None when not configured."""
        config = _make_config()
        orch = Orchestrator(config)
        self.assertIsNone(orch.registry)

    def test_set_registry(self) -> None:
        """registry can be set at runtime."""
        config = _make_config()
        orch = Orchestrator(config)
        self.assertIsNone(orch.registry)

        registry = ModelRegistry()
        orch.registry = registry
        self.assertIs(orch.registry, registry)

    def test_clear_registry(self) -> None:
        """registry can be cleared to None."""
        config = _make_config()
        registry = ModelRegistry()
        orch = Orchestrator(config, registry=registry)
        self.assertIsNotNone(orch.registry)

        orch.registry = None
        self.assertIsNone(orch.registry)

    def test_replace_registry(self) -> None:
        """registry can be replaced with a different instance."""
        config = _make_config()
        registry1 = ModelRegistry(default_model="model-a")
        registry2 = ModelRegistry(default_model="model-b")
        orch = Orchestrator(config, registry=registry1)

        orch.registry = registry2
        self.assertIs(orch.registry, registry2)


class TestOrchestratorConformance(unittest.TestCase):
    """Tests verifying orchestrator fulfills spec requirements."""

    def test_manages_provider_instances(self) -> None:
        """Orchestrator manages and caches provider instances."""
        config = _make_config()
        orch = Orchestrator(config)

        mock_prov = _make_mock_provider("ollama")
        orch._providers["ollama"] = mock_prov

        # Getting provider by name returns cached instance.
        self.assertIs(orch.get_provider("ollama"), mock_prov)

    def test_consults_registry(self) -> None:
        """Orchestrator consults model registry for task routing."""
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.REVIEW, "claude", "claude-haiku-4-5", priority=1
        )

        config = _make_config()
        orch = Orchestrator(config, registry=registry)

        result = orch.get_model_for_task(TaskType.REVIEW)
        self.assertEqual(result, ("claude", "claude-haiku-4-5"))

    def test_supports_runtime_provider_switching(self) -> None:
        """Orchestrator supports switching providers at runtime."""
        config = _make_config(provider="ollama")
        orch = Orchestrator(config)

        mock_ollama = _make_mock_provider("ollama")
        mock_claude = _make_mock_provider("claude")
        orch._providers["ollama"] = mock_ollama
        orch._providers["claude"] = mock_claude

        self.assertEqual(orch.get_active_provider_name(), "ollama")
        orch.switch_provider("claude")
        self.assertEqual(orch.get_active_provider_name(), "claude")
        orch.switch_provider("ollama")
        self.assertEqual(orch.get_active_provider_name(), "ollama")

    def test_supports_runtime_brain_switching(self) -> None:
        """Orchestrator supports switching the brain model at runtime."""
        config = _make_config(model="qwen3:8b")
        orch = Orchestrator(config)

        self.assertEqual(orch.get_brain_model(), "qwen3:8b")
        orch.set_brain_model("claude-sonnet-4-5")
        self.assertEqual(orch.get_brain_model(), "claude-sonnet-4-5")
        orch.set_brain_model("phi4-mini")
        self.assertEqual(orch.get_brain_model(), "phi4-mini")


class TestProviderCreation(unittest.TestCase):
    """Tests for provider creation with correct parameters."""

    def test_ollama_provider_gets_base_url(self) -> None:
        """Ollama provider is created with the configured host URL."""
        config = _make_config(ollama_host="http://custom:9999")
        orch = Orchestrator(config)

        with patch("local_cli.providers.get_provider") as mock_get:
            mock_prov = _make_mock_provider("ollama")
            mock_get.return_value = mock_prov

            orch.get_provider("ollama")

            mock_get.assert_called_once_with(
                "ollama", base_url="http://custom:9999"
            )

    def test_claude_provider_no_extra_args(self) -> None:
        """Claude provider is created without extra arguments."""
        config = _make_config()
        orch = Orchestrator(config)

        with patch("local_cli.providers.get_provider") as mock_get:
            mock_prov = _make_mock_provider("claude")
            mock_get.return_value = mock_prov

            orch.get_provider("claude")

            mock_get.assert_called_once_with("claude")


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error paths."""

    def test_get_provider_for_task_both_fail(self) -> None:
        """When both preferred and active providers fail, raises."""
        config = _make_config(provider="ollama")
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.PLANNING, "claude", "claude-sonnet-4-5", priority=1
        )
        orch = Orchestrator(config, registry=registry)

        with patch("local_cli.providers.get_provider") as mock_get:
            mock_get.side_effect = ValueError("All providers unavailable")

            # get_provider_for_task tries preferred (claude), then falls
            # back to active (ollama). Both use get_provider which fails.
            with self.assertRaises(ValueError):
                orch.get_provider_for_task(TaskType.PLANNING)

    def test_switch_back_uses_cache(self) -> None:
        """Switching back to a previously used provider uses the cache."""
        config = _make_config(provider="ollama")
        orch = Orchestrator(config)

        mock_ollama = _make_mock_provider("ollama")
        mock_claude = _make_mock_provider("claude")
        orch._providers["ollama"] = mock_ollama
        orch._providers["claude"] = mock_claude

        # Switch to claude and back.
        orch.switch_provider("claude")
        result = orch.switch_provider("ollama")

        self.assertIs(result, mock_ollama)
        self.assertEqual(orch.get_active_provider_name(), "ollama")

    def test_registry_update_affects_routing(self) -> None:
        """Runtime registry updates immediately affect task routing."""
        config = _make_config()
        registry = ModelRegistry()
        orch = Orchestrator(config, registry=registry)

        # No route -> default.
        result1 = orch.get_model_for_task(TaskType.CODE_GENERATION)
        self.assertEqual(result1, ("ollama", "qwen3:8b"))

        # Add route -> specific mapping.
        registry.update_task_route(
            TaskType.CODE_GENERATION, "claude", "claude-sonnet-4-5", priority=1
        )
        result2 = orch.get_model_for_task(TaskType.CODE_GENERATION)
        self.assertEqual(result2, ("claude", "claude-sonnet-4-5"))

    def test_set_registry_changes_routing(self) -> None:
        """Setting a new registry changes task routing behavior."""
        config = _make_config(provider="ollama", model="qwen3:8b")
        orch = Orchestrator(config)

        # Without registry -> config defaults.
        result1 = orch.get_model_for_task(TaskType.PLANNING)
        self.assertEqual(result1, ("ollama", "qwen3:8b"))

        # Set registry with different routing.
        registry = ModelRegistry()
        registry.update_task_route(
            TaskType.PLANNING, "claude", "claude-sonnet-4-5", priority=1
        )
        orch.registry = registry

        result2 = orch.get_model_for_task(TaskType.PLANNING)
        self.assertEqual(result2, ("claude", "claude-sonnet-4-5"))


if __name__ == "__main__":
    unittest.main()
