"""Tests for local_cli.model_manager module."""

import unittest
from unittest.mock import MagicMock, call, patch

from local_cli.model_manager import (
    ModelManager,
    ModelManagerError,
    ModelOperationInProgressError,
    ProgressCallback,
)
from local_cli.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaRequestError,
    OllamaStreamError,
)


def _make_client() -> MagicMock:
    """Create a mock OllamaClient."""
    return MagicMock(spec=OllamaClient)


class TestModelManagerInit(unittest.TestCase):
    """Tests for ModelManager construction."""

    def test_accepts_ollama_client(self) -> None:
        """ModelManager can be created with an OllamaClient."""
        client = _make_client()
        manager = ModelManager(client)
        self.assertIs(manager._client, client)

    def test_lock_initially_false(self) -> None:
        """Operation lock starts as False."""
        client = _make_client()
        manager = ModelManager(client)
        self.assertFalse(manager._operation_in_progress)


class TestInstallModel(unittest.TestCase):
    """Tests for ModelManager.install_model()."""

    def setUp(self) -> None:
        self.client = _make_client()
        self.manager = ModelManager(self.client)

    def test_calls_pull_model(self) -> None:
        """install_model delegates to client.pull_model."""
        self.client.pull_model.return_value = iter([
            {"status": "pulling manifest"},
            {"status": "downloading", "completed": 500, "total": 1000},
            {"status": "success"},
        ])

        self.manager.install_model("qwen3:8b")

        self.client.pull_model.assert_called_once_with("qwen3:8b")

    def test_progress_callback_invoked(self) -> None:
        """Progress callback receives status, completed, total."""
        self.client.pull_model.return_value = iter([
            {"status": "pulling manifest"},
            {"status": "downloading", "completed": 500, "total": 1000},
            {"status": "success"},
        ])

        callback = MagicMock()
        self.manager.install_model("qwen3:8b", progress_callback=callback)

        expected_calls = [
            call("pulling manifest", None, None),
            call("downloading", 500, 1000),
            call("success", None, None),
        ]
        callback.assert_has_calls(expected_calls)
        self.assertEqual(callback.call_count, 3)

    def test_no_callback_ok(self) -> None:
        """install_model works without a progress callback."""
        self.client.pull_model.return_value = iter([
            {"status": "success"},
        ])

        # Should not raise.
        self.manager.install_model("qwen3:8b")

    def test_invalid_model_name_raises_value_error(self) -> None:
        """install_model rejects invalid model names."""
        with self.assertRaises(ValueError) as ctx:
            self.manager.install_model("")
        self.assertIn("Invalid model name", str(ctx.exception))

    def test_invalid_model_name_with_special_chars(self) -> None:
        """install_model rejects model names with special characters."""
        with self.assertRaises(ValueError):
            self.manager.install_model("model; rm -rf /")

    def test_connection_error_raises_manager_error(self) -> None:
        """Connection errors are wrapped in ModelManagerError."""
        self.client.pull_model.side_effect = OllamaConnectionError(
            "Connection refused"
        )

        with self.assertRaises(ModelManagerError) as ctx:
            self.manager.install_model("qwen3:8b")
        self.assertIn("Failed to connect", str(ctx.exception))

    def test_stream_error_raises_manager_error(self) -> None:
        """Stream errors are wrapped in ModelManagerError."""
        def _failing_stream(name):
            yield {"status": "pulling manifest"}
            raise OllamaStreamError("Model not found")

        self.client.pull_model.side_effect = _failing_stream

        with self.assertRaises(ModelManagerError) as ctx:
            self.manager.install_model("qwen3:8b")
        self.assertIn("Error during installation", str(ctx.exception))

    def test_lock_released_on_success(self) -> None:
        """Lock is released after successful installation."""
        self.client.pull_model.return_value = iter([{"status": "success"}])

        self.manager.install_model("qwen3:8b")

        self.assertFalse(self.manager._operation_in_progress)

    def test_lock_released_on_error(self) -> None:
        """Lock is released even when installation fails."""
        self.client.pull_model.side_effect = OllamaConnectionError("fail")

        with self.assertRaises(ModelManagerError):
            self.manager.install_model("qwen3:8b")

        self.assertFalse(self.manager._operation_in_progress)

    def test_lock_prevents_concurrent_install(self) -> None:
        """Cannot start install while another operation is in progress."""
        self.manager._operation_in_progress = True

        with self.assertRaises(ModelOperationInProgressError) as ctx:
            self.manager.install_model("qwen3:8b")
        self.assertIn("already in progress", str(ctx.exception))

    def test_progress_callback_with_missing_fields(self) -> None:
        """Callback handles chunks with missing completed/total."""
        self.client.pull_model.return_value = iter([
            {"status": "verifying sha256 digest"},
        ])

        callback = MagicMock()
        self.manager.install_model("qwen3:8b", progress_callback=callback)

        callback.assert_called_once_with("verifying sha256 digest", None, None)

    def test_progress_callback_with_empty_status(self) -> None:
        """Callback handles chunks with empty status."""
        self.client.pull_model.return_value = iter([
            {"completed": 100, "total": 200},
        ])

        callback = MagicMock()
        self.manager.install_model("qwen3:8b", progress_callback=callback)

        callback.assert_called_once_with("", 100, 200)


class TestUpdateModel(unittest.TestCase):
    """Tests for ModelManager.update_model()."""

    def setUp(self) -> None:
        self.client = _make_client()
        self.manager = ModelManager(self.client)

    def test_delegates_to_install(self) -> None:
        """update_model delegates to install_model (re-pull)."""
        self.client.pull_model.return_value = iter([{"status": "success"}])

        self.manager.update_model("qwen3:8b")

        self.client.pull_model.assert_called_once_with("qwen3:8b")

    def test_passes_progress_callback(self) -> None:
        """update_model forwards the progress callback."""
        self.client.pull_model.return_value = iter([
            {"status": "downloading", "completed": 50, "total": 100},
        ])

        callback = MagicMock()
        self.manager.update_model("qwen3:8b", progress_callback=callback)

        callback.assert_called_once_with("downloading", 50, 100)

    def test_invalid_model_name_raises(self) -> None:
        """update_model rejects invalid model names."""
        with self.assertRaises(ValueError):
            self.manager.update_model("")

    def test_lock_prevents_concurrent_update(self) -> None:
        """Cannot start update while another operation is in progress."""
        self.manager._operation_in_progress = True

        with self.assertRaises(ModelOperationInProgressError):
            self.manager.update_model("qwen3:8b")


class TestDeleteModel(unittest.TestCase):
    """Tests for ModelManager.delete_model()."""

    def setUp(self) -> None:
        self.client = _make_client()
        self.manager = ModelManager(self.client)

    def test_calls_client_delete(self) -> None:
        """delete_model delegates to client.delete_model."""
        self.manager.delete_model("phi4-mini")

        self.client.delete_model.assert_called_once_with("phi4-mini")

    def test_invalid_model_name_raises(self) -> None:
        """delete_model rejects invalid model names."""
        with self.assertRaises(ValueError) as ctx:
            self.manager.delete_model("")
        self.assertIn("Invalid model name", str(ctx.exception))

    def test_connection_error_raises_manager_error(self) -> None:
        """Connection errors are wrapped in ModelManagerError."""
        self.client.delete_model.side_effect = OllamaConnectionError(
            "Connection refused"
        )

        with self.assertRaises(ModelManagerError) as ctx:
            self.manager.delete_model("phi4-mini")
        self.assertIn("Failed to connect", str(ctx.exception))

    def test_request_error_raises_manager_error(self) -> None:
        """Request errors (e.g. model not found) are wrapped."""
        self.client.delete_model.side_effect = OllamaRequestError(
            "model not found"
        )

        with self.assertRaises(ModelManagerError) as ctx:
            self.manager.delete_model("nonexistent:latest")
        self.assertIn("Failed to delete", str(ctx.exception))

    def test_lock_released_on_success(self) -> None:
        """Lock is released after successful deletion."""
        self.manager.delete_model("phi4-mini")

        self.assertFalse(self.manager._operation_in_progress)

    def test_lock_released_on_error(self) -> None:
        """Lock is released even when deletion fails."""
        self.client.delete_model.side_effect = OllamaRequestError("fail")

        with self.assertRaises(ModelManagerError):
            self.manager.delete_model("phi4-mini")

        self.assertFalse(self.manager._operation_in_progress)

    def test_lock_prevents_concurrent_delete(self) -> None:
        """Cannot start delete while another operation is in progress."""
        self.manager._operation_in_progress = True

        with self.assertRaises(ModelOperationInProgressError):
            self.manager.delete_model("phi4-mini")

    def test_client_not_called_on_invalid_name(self) -> None:
        """Client is not called when model name validation fails."""
        with self.assertRaises(ValueError):
            self.manager.delete_model("../etc/passwd")

        self.client.delete_model.assert_not_called()


class TestGetModelInfo(unittest.TestCase):
    """Tests for ModelManager.get_model_info()."""

    def setUp(self) -> None:
        self.client = _make_client()
        self.manager = ModelManager(self.client)

    def test_calls_show_model(self) -> None:
        """get_model_info delegates to client.show_model."""
        expected = {
            "modelfile": "FROM qwen3:8b",
            "parameters": "temperature 0.7",
            "details": {"family": "qwen3", "parameter_size": "8B"},
            "capabilities": ["completion", "tools"],
        }
        self.client.show_model.return_value = expected

        result = self.manager.get_model_info("qwen3:8b")

        self.assertEqual(result, expected)
        self.client.show_model.assert_called_once_with("qwen3:8b")

    def test_invalid_model_name_raises(self) -> None:
        """get_model_info rejects invalid model names."""
        with self.assertRaises(ValueError) as ctx:
            self.manager.get_model_info("")
        self.assertIn("Invalid model name", str(ctx.exception))

    def test_connection_error_raises_manager_error(self) -> None:
        """Connection errors are wrapped in ModelManagerError."""
        self.client.show_model.side_effect = OllamaConnectionError(
            "Connection refused"
        )

        with self.assertRaises(ModelManagerError) as ctx:
            self.manager.get_model_info("qwen3:8b")
        self.assertIn("Failed to connect", str(ctx.exception))

    def test_request_error_raises_manager_error(self) -> None:
        """Request errors are wrapped in ModelManagerError."""
        self.client.show_model.side_effect = OllamaRequestError(
            "model not found"
        )

        with self.assertRaises(ModelManagerError) as ctx:
            self.manager.get_model_info("nonexistent:latest")
        self.assertIn("Failed to get info", str(ctx.exception))

    def test_does_not_require_lock(self) -> None:
        """get_model_info does not use the operation lock."""
        self.client.show_model.return_value = {"details": {}}

        # Should succeed even with lock set.
        self.manager._operation_in_progress = True
        result = self.manager.get_model_info("qwen3:8b")

        self.assertEqual(result, {"details": {}})

    def test_returns_raw_dict(self) -> None:
        """get_model_info returns the raw dict from show_model."""
        info = {
            "license": "MIT",
            "template": "{{ .Prompt }}",
            "details": {},
        }
        self.client.show_model.return_value = info

        result = self.manager.get_model_info("qwen3:8b")

        self.assertIs(result, info)

    def test_model_without_capabilities(self) -> None:
        """get_model_info works for models without capabilities."""
        info = {"details": {"family": "custom"}}
        self.client.show_model.return_value = info

        result = self.manager.get_model_info("custom-model:latest")

        self.assertNotIn("capabilities", result)


class TestListRunning(unittest.TestCase):
    """Tests for ModelManager.list_running()."""

    def setUp(self) -> None:
        self.client = _make_client()
        self.manager = ModelManager(self.client)

    def test_calls_list_running_models(self) -> None:
        """list_running delegates to client.list_running_models."""
        expected = [
            {"name": "qwen3:8b", "size": 5_200_000_000},
            {"name": "phi4-mini", "size": 2_500_000_000},
        ]
        self.client.list_running_models.return_value = expected

        result = self.manager.list_running()

        self.assertEqual(result, expected)
        self.client.list_running_models.assert_called_once()

    def test_empty_list(self) -> None:
        """list_running returns empty list when no models are loaded."""
        self.client.list_running_models.return_value = []

        result = self.manager.list_running()

        self.assertEqual(result, [])

    def test_connection_error_raises_manager_error(self) -> None:
        """Connection errors are wrapped in ModelManagerError."""
        self.client.list_running_models.side_effect = OllamaConnectionError(
            "Connection refused"
        )

        with self.assertRaises(ModelManagerError) as ctx:
            self.manager.list_running()
        self.assertIn("Failed to connect", str(ctx.exception))

    def test_request_error_raises_manager_error(self) -> None:
        """Request errors are wrapped in ModelManagerError."""
        self.client.list_running_models.side_effect = OllamaRequestError(
            "Internal error"
        )

        with self.assertRaises(ModelManagerError) as ctx:
            self.manager.list_running()
        self.assertIn("Failed to list running models", str(ctx.exception))

    def test_does_not_require_lock(self) -> None:
        """list_running does not use the operation lock."""
        self.client.list_running_models.return_value = []

        # Should succeed even with lock set.
        self.manager._operation_in_progress = True
        result = self.manager.list_running()

        self.assertEqual(result, [])


class TestIsAvailable(unittest.TestCase):
    """Tests for ModelManager.is_available()."""

    def setUp(self) -> None:
        self.client = _make_client()
        self.manager = ModelManager(self.client)

    def test_model_found(self) -> None:
        """is_available returns True when model is in the list."""
        self.client.list_models.return_value = [
            {"name": "qwen3:8b", "size": 5_200_000_000},
            {"name": "phi4-mini", "size": 2_500_000_000},
        ]

        self.assertTrue(self.manager.is_available("qwen3:8b"))

    def test_model_not_found(self) -> None:
        """is_available returns False when model is not in the list."""
        self.client.list_models.return_value = [
            {"name": "qwen3:8b", "size": 5_200_000_000},
        ]

        self.assertFalse(self.manager.is_available("phi4-mini"))

    def test_empty_model_list(self) -> None:
        """is_available returns False when no models are installed."""
        self.client.list_models.return_value = []

        self.assertFalse(self.manager.is_available("qwen3:8b"))

    def test_invalid_model_name_raises(self) -> None:
        """is_available rejects invalid model names."""
        with self.assertRaises(ValueError):
            self.manager.is_available("")

    def test_connection_error_raises_manager_error(self) -> None:
        """Connection errors are wrapped in ModelManagerError."""
        self.client.list_models.side_effect = OllamaConnectionError(
            "Connection refused"
        )

        with self.assertRaises(ModelManagerError) as ctx:
            self.manager.is_available("qwen3:8b")
        self.assertIn("Failed to connect", str(ctx.exception))

    def test_does_not_require_lock(self) -> None:
        """is_available does not use the operation lock."""
        self.client.list_models.return_value = [
            {"name": "qwen3:8b", "size": 5_200_000_000},
        ]

        # Should succeed even with lock set.
        self.manager._operation_in_progress = True
        self.assertTrue(self.manager.is_available("qwen3:8b"))

    def test_exact_name_match(self) -> None:
        """is_available uses exact name matching, not substring."""
        self.client.list_models.return_value = [
            {"name": "qwen3:8b", "size": 5_200_000_000},
        ]

        self.assertFalse(self.manager.is_available("qwen3"))
        self.assertFalse(self.manager.is_available("qwen3:8"))


class TestExceptionHierarchy(unittest.TestCase):
    """Tests for model manager exception hierarchy."""

    def test_model_manager_error_is_exception(self) -> None:
        """ModelManagerError inherits from Exception."""
        self.assertTrue(issubclass(ModelManagerError, Exception))

    def test_operation_in_progress_is_manager_error(self) -> None:
        """ModelOperationInProgressError inherits from ModelManagerError."""
        self.assertTrue(
            issubclass(ModelOperationInProgressError, ModelManagerError)
        )

    def test_catch_manager_error_catches_operation_in_progress(self) -> None:
        """except ModelManagerError also catches ModelOperationInProgressError."""
        with self.assertRaises(ModelManagerError):
            raise ModelOperationInProgressError("test")


class TestLockBehavior(unittest.TestCase):
    """Tests for the operation lock mechanism."""

    def setUp(self) -> None:
        self.client = _make_client()
        self.manager = ModelManager(self.client)

    def test_install_acquires_and_releases_lock(self) -> None:
        """install_model acquires lock during operation and releases after."""
        lock_states: list[bool] = []

        def _track_lock(name):
            lock_states.append(self.manager._operation_in_progress)
            return iter([{"status": "success"}])

        self.client.pull_model.side_effect = _track_lock

        self.manager.install_model("qwen3:8b")

        # Lock should have been True during pull_model call.
        self.assertEqual(lock_states, [True])
        # Lock should be released after.
        self.assertFalse(self.manager._operation_in_progress)

    def test_delete_acquires_and_releases_lock(self) -> None:
        """delete_model acquires lock during operation and releases after."""
        lock_states: list[bool] = []

        def _track_lock(name):
            lock_states.append(self.manager._operation_in_progress)

        self.client.delete_model.side_effect = _track_lock

        self.manager.delete_model("phi4-mini")

        self.assertEqual(lock_states, [True])
        self.assertFalse(self.manager._operation_in_progress)

    def test_install_blocks_delete(self) -> None:
        """Cannot delete while install is in progress."""
        self.manager._operation_in_progress = True

        with self.assertRaises(ModelOperationInProgressError):
            self.manager.delete_model("phi4-mini")

    def test_delete_blocks_install(self) -> None:
        """Cannot install while delete is in progress."""
        self.manager._operation_in_progress = True

        with self.assertRaises(ModelOperationInProgressError):
            self.manager.install_model("qwen3:8b")

    def test_install_then_delete_works(self) -> None:
        """Sequential install then delete works fine."""
        self.client.pull_model.return_value = iter([{"status": "success"}])

        self.manager.install_model("qwen3:8b")
        self.manager.delete_model("qwen3:8b")

        self.client.pull_model.assert_called_once()
        self.client.delete_model.assert_called_once()

    def test_lock_released_on_keyboard_interrupt(self) -> None:
        """Lock is released even on KeyboardInterrupt during pull."""
        def _interrupt(name):
            yield {"status": "pulling"}
            raise KeyboardInterrupt()

        self.client.pull_model.side_effect = _interrupt

        with self.assertRaises(KeyboardInterrupt):
            self.manager.install_model("qwen3:8b")

        self.assertFalse(self.manager._operation_in_progress)


class TestModelNameValidation(unittest.TestCase):
    """Tests for model name validation across all methods."""

    def setUp(self) -> None:
        self.client = _make_client()
        self.manager = ModelManager(self.client)

    def test_valid_simple_name(self) -> None:
        """Simple model name like 'phi4-mini' is accepted."""
        self.client.pull_model.return_value = iter([{"status": "success"}])
        self.manager.install_model("phi4-mini")
        self.client.pull_model.assert_called_once()

    def test_valid_name_with_tag(self) -> None:
        """Model name with tag like 'qwen3:8b' is accepted."""
        self.client.pull_model.return_value = iter([{"status": "success"}])
        self.manager.install_model("qwen3:8b")
        self.client.pull_model.assert_called_once()

    def test_valid_name_with_namespace(self) -> None:
        """Model name with namespace like 'library/model:latest' is accepted."""
        self.client.pull_model.return_value = iter([{"status": "success"}])
        self.manager.install_model("library/model:latest")
        self.client.pull_model.assert_called_once()

    def test_empty_name_rejected_by_install(self) -> None:
        """Empty model name is rejected by install_model."""
        with self.assertRaises(ValueError):
            self.manager.install_model("")

    def test_empty_name_rejected_by_delete(self) -> None:
        """Empty model name is rejected by delete_model."""
        with self.assertRaises(ValueError):
            self.manager.delete_model("")

    def test_empty_name_rejected_by_info(self) -> None:
        """Empty model name is rejected by get_model_info."""
        with self.assertRaises(ValueError):
            self.manager.get_model_info("")

    def test_empty_name_rejected_by_is_available(self) -> None:
        """Empty model name is rejected by is_available."""
        with self.assertRaises(ValueError):
            self.manager.is_available("")

    def test_validation_before_lock_acquired(self) -> None:
        """Model name validation happens before lock acquisition."""
        # If validation raised after lock was acquired, the lock wouldn't
        # be in the expected state. Validation should raise before locking.
        with self.assertRaises(ValueError):
            self.manager.install_model("")

        self.assertFalse(self.manager._operation_in_progress)


if __name__ == "__main__":
    unittest.main()
