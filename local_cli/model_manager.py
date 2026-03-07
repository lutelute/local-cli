"""High-level model management for local-cli.

Provides a ``ModelManager`` class that wraps the low-level
``OllamaClient`` model endpoints with higher-level lifecycle operations:
install (pull with progress), update (re-pull), delete, info query,
running model inspection, and availability checks.

Progress callbacks are supported for long-running operations such as
model installation, allowing callers to display download progress.
"""

from typing import Any, Callable

from local_cli.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaRequestError,
    OllamaStreamError,
)
from local_cli.security import validate_model_name

# Type alias for progress callbacks.
# (status: str, completed: int | None, total: int | None) -> None
ProgressCallback = Callable[[str, int | None, int | None], None]


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ModelManagerError(Exception):
    """Raised when a model management operation fails."""


class ModelOperationInProgressError(ModelManagerError):
    """Raised when another model operation is already in progress."""


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class ModelManager:
    """High-level model management wrapping ``OllamaClient``.

    Provides lifecycle operations for Ollama models: install, update,
    delete, info, running status, and availability checks.  A simple
    lock flag prevents concurrent model operations.

    Args:
        client: An ``OllamaClient`` instance for communicating with the
            Ollama REST API.
    """

    def __init__(self, client: OllamaClient) -> None:
        self._client: OllamaClient = client
        self._operation_in_progress: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_name(self, model: str) -> None:
        """Validate a model name.

        Args:
            model: Model name to validate.

        Raises:
            ValueError: If the model name is invalid.
        """
        if not validate_model_name(model):
            raise ValueError(f"Invalid model name: {model!r}")

    def _acquire_lock(self) -> None:
        """Acquire the operation lock.

        Raises:
            ModelOperationInProgressError: If another operation is
                already in progress.
        """
        if self._operation_in_progress:
            raise ModelOperationInProgressError(
                "Another model operation is already in progress"
            )
        self._operation_in_progress = True

    def _release_lock(self) -> None:
        """Release the operation lock."""
        self._operation_in_progress = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def install_model(
        self,
        name: str,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Install (pull) a model from the Ollama registry.

        Downloads the model with streaming progress updates.  If a
        *progress_callback* is provided, it is called for each progress
        update with ``(status, completed, total)``.

        Args:
            name: Model name to install (e.g. ``"qwen3:8b"``).
            progress_callback: Optional callback invoked with
                ``(status: str, completed: int | None, total: int | None)``
                for each progress update.

        Raises:
            ValueError: If the model name is invalid.
            ModelOperationInProgressError: If another operation is in
                progress.
            ModelManagerError: If the installation fails due to a
                connection or stream error.
        """
        self._validate_name(name)
        self._acquire_lock()

        try:
            for chunk in self._client.pull_model(name):
                if progress_callback is not None:
                    status = chunk.get("status", "")
                    completed = chunk.get("completed")
                    total = chunk.get("total")
                    progress_callback(status, completed, total)
        except OllamaConnectionError as exc:
            raise ModelManagerError(
                f"Failed to connect to Ollama while installing {name!r}: {exc}"
            ) from exc
        except OllamaStreamError as exc:
            raise ModelManagerError(
                f"Error during installation of {name!r}: {exc}"
            ) from exc
        finally:
            self._release_lock()

    def update_model(
        self,
        name: str,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Update (re-pull) an installed model.

        This is equivalent to ``install_model()`` -- pulling an already
        installed model updates it to the latest version.

        Args:
            name: Model name to update (e.g. ``"qwen3:8b"``).
            progress_callback: Optional callback invoked with
                ``(status: str, completed: int | None, total: int | None)``
                for each progress update.

        Raises:
            ValueError: If the model name is invalid.
            ModelOperationInProgressError: If another operation is in
                progress.
            ModelManagerError: If the update fails due to a connection
                or stream error.
        """
        self.install_model(name, progress_callback=progress_callback)

    def delete_model(self, name: str) -> None:
        """Delete a model from Ollama.

        Args:
            name: Model name to delete (e.g. ``"phi4-mini"``).

        Raises:
            ValueError: If the model name is invalid.
            ModelOperationInProgressError: If another operation is in
                progress.
            ModelManagerError: If the deletion fails due to a connection
                or request error.
        """
        self._validate_name(name)
        self._acquire_lock()

        try:
            self._client.delete_model(name)
        except OllamaConnectionError as exc:
            raise ModelManagerError(
                f"Failed to connect to Ollama while deleting {name!r}: {exc}"
            ) from exc
        except OllamaRequestError as exc:
            raise ModelManagerError(
                f"Failed to delete model {name!r}: {exc}"
            ) from exc
        finally:
            self._release_lock()

    def get_model_info(self, name: str) -> dict[str, Any]:
        """Get detailed information about a model.

        Returns model metadata including template, parameters, license,
        details, and capabilities (if the model advertises them).

        Args:
            name: Model name to inspect (e.g. ``"qwen3:8b"``).

        Returns:
            A dict with model details from the Ollama ``/api/show``
            endpoint.

        Raises:
            ValueError: If the model name is invalid.
            ModelManagerError: If the info query fails due to a
                connection or request error.
        """
        self._validate_name(name)

        try:
            return self._client.show_model(name)
        except OllamaConnectionError as exc:
            raise ModelManagerError(
                f"Failed to connect to Ollama while querying {name!r}: {exc}"
            ) from exc
        except OllamaRequestError as exc:
            raise ModelManagerError(
                f"Failed to get info for model {name!r}: {exc}"
            ) from exc

    def list_running(self) -> list[dict[str, Any]]:
        """List models currently loaded in VRAM.

        Returns:
            A list of running model info dicts from the Ollama
            ``/api/ps`` endpoint.

        Raises:
            ModelManagerError: If the query fails due to a connection
                or request error.
        """
        try:
            return self._client.list_running_models()
        except OllamaConnectionError as exc:
            raise ModelManagerError(
                f"Failed to connect to Ollama: {exc}"
            ) from exc
        except OllamaRequestError as exc:
            raise ModelManagerError(
                f"Failed to list running models: {exc}"
            ) from exc

    def is_available(self, name: str) -> bool:
        """Check whether a model is available locally.

        Queries the list of installed models and checks if the given
        model name is present.

        Args:
            name: Model name to check (e.g. ``"qwen3:8b"``).

        Returns:
            True if the model is available, False otherwise.

        Raises:
            ValueError: If the model name is invalid.
            ModelManagerError: If the query fails due to a connection
                error.
        """
        self._validate_name(name)

        try:
            models = self._client.list_models()
        except OllamaConnectionError as exc:
            raise ModelManagerError(
                f"Failed to connect to Ollama: {exc}"
            ) from exc

        return any(m.get("name") == name for m in models)
