"""Config-driven model registry for local-cli.

Maps task types to model+provider combinations with priority ordering
and fallback support.  The registry is loaded from and saved to a JSON
configuration file.
"""

import enum
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from local_cli.security import validate_model_name

# Maximum registry file size in bytes (64KB).
_MAX_REGISTRY_SIZE = 64 * 1024

# Valid provider names.
_VALID_PROVIDERS = frozenset({"ollama", "claude"})


class TaskType(enum.Enum):
    """Supported task types for model routing."""

    CODE_GENERATION = "code_generation"
    PLANNING = "planning"
    REVIEW = "review"
    DOCUMENTATION = "documentation"
    GENERAL = "general"


@dataclass
class RegistryEntry:
    """A single model+provider routing entry.

    Attributes:
        provider: Provider name (e.g. ``"ollama"`` or ``"claude"``).
        model: Model name (e.g. ``"qwen3:8b"``).
        priority: Priority ordering (1 = highest priority, used first).
    """

    provider: str
    model: str
    priority: int


class ModelRegistry:
    """Config-driven registry mapping task types to model+provider combos.

    The registry supports:
    - Loading from / saving to a JSON configuration file.
    - Looking up the best model for a given task type (by priority).
    - Falling back to the default model when no specific mapping exists.
    - Runtime updates to task routing entries.

    Args:
        default_provider: Default provider name (e.g. ``"ollama"``).
        default_model: Default model name (e.g. ``"qwen3:8b"``).
    """

    def __init__(
        self,
        default_provider: str = "ollama",
        default_model: str = "qwen3:8b",
    ) -> None:
        self._default_provider: str = default_provider
        self._default_model: str = default_model
        self._routes: dict[TaskType, list[RegistryEntry]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def default_provider(self) -> str:
        """The default provider name."""
        return self._default_provider

    @property
    def default_model(self) -> str:
        """The default model name."""
        return self._default_model

    def get_default(self) -> tuple[str, str]:
        """Return the default (provider, model) pair.

        Returns:
            A tuple of ``(provider, model)``.
        """
        return (self._default_provider, self._default_model)

    def get_model_for_task(self, task_type: TaskType) -> tuple[str, str]:
        """Look up the best model for a given task type.

        Returns the highest-priority (lowest priority number) entry for
        the specified task type.  If no mapping exists for the task type,
        falls back to the default model.

        Args:
            task_type: The type of task to route.

        Returns:
            A tuple of ``(provider, model)``.
        """
        entries = self._routes.get(task_type)
        if not entries:
            return self.get_default()

        # Entries are kept sorted by priority (ascending).
        best = entries[0]
        return (best.provider, best.model)

    def get_models_for_task(self, task_type: TaskType) -> list[tuple[str, str]]:
        """Return all model+provider combos for a task type, ordered by priority.

        Args:
            task_type: The type of task to route.

        Returns:
            A list of ``(provider, model)`` tuples sorted by priority
            (highest priority first).  Returns a list with just the
            default if no specific mapping exists.
        """
        entries = self._routes.get(task_type)
        if not entries:
            return [self.get_default()]

        return [(e.provider, e.model) for e in entries]

    def update_task_route(
        self,
        task_type: TaskType,
        provider: str,
        model: str,
        priority: int = 1,
    ) -> None:
        """Add or update a routing entry for a task type.

        If an entry with the same provider+model already exists for the
        task type, its priority is updated.  Otherwise a new entry is
        appended.  Entries are re-sorted by priority after modification.

        Args:
            task_type: The task type to route.
            provider: Provider name (must be in ``_VALID_PROVIDERS``).
            model: Model name (validated via ``validate_model_name``).
            priority: Priority ordering (1 = highest).

        Raises:
            ValueError: If the provider name is invalid, the model name
                is invalid, or the priority is not a positive integer.
        """
        if provider not in _VALID_PROVIDERS:
            raise ValueError(
                f"Invalid provider: {provider!r}. "
                f"Must be one of: {', '.join(sorted(_VALID_PROVIDERS))}"
            )

        if not validate_model_name(model):
            raise ValueError(f"Invalid model name: {model!r}")

        if not isinstance(priority, int) or priority < 1:
            raise ValueError(
                f"Priority must be a positive integer, got: {priority!r}"
            )

        entries = self._routes.setdefault(task_type, [])

        # Check for existing entry with same provider+model.
        for entry in entries:
            if entry.provider == provider and entry.model == model:
                entry.priority = priority
                entries.sort(key=lambda e: e.priority)
                return

        entries.append(RegistryEntry(provider=provider, model=model, priority=priority))
        entries.sort(key=lambda e: e.priority)

    def remove_task_route(
        self,
        task_type: TaskType,
        provider: str,
        model: str,
    ) -> bool:
        """Remove a specific routing entry for a task type.

        Args:
            task_type: The task type to modify.
            provider: Provider name of the entry to remove.
            model: Model name of the entry to remove.

        Returns:
            True if an entry was removed, False if not found.
        """
        entries = self._routes.get(task_type)
        if not entries:
            return False

        original_len = len(entries)
        self._routes[task_type] = [
            e for e in entries
            if not (e.provider == provider and e.model == model)
        ]

        # Clean up empty lists.
        if not self._routes[task_type]:
            del self._routes[task_type]

        return len(self._routes.get(task_type, [])) < original_len

    def list_routes(self) -> dict[str, list[dict[str, object]]]:
        """Return all routing entries as a plain dictionary.

        Returns:
            A dictionary mapping task type values (strings) to lists of
            entry dictionaries, each containing ``provider``, ``model``,
            and ``priority`` keys.
        """
        result: dict[str, list[dict[str, object]]] = {}
        for task_type, entries in self._routes.items():
            result[task_type.value] = [asdict(e) for e in entries]
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the registry to a JSON file.

        Args:
            path: File path to write the registry JSON to.
        """
        data: dict[str, object] = {
            "default_provider": self._default_provider,
            "default_model": self._default_model,
            "task_routing": self.list_routes(),
        }

        file_path = Path(path).expanduser()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        text = json.dumps(data, indent=2, ensure_ascii=False)
        file_path.write_text(text + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "ModelRegistry":
        """Load a registry from a JSON file.

        Validates the structure and contents of the registry file.
        Invalid entries are silently skipped so that partially valid
        registries can still be used.

        Security: Rejects symlinks and oversized files (>64KB).

        Args:
            path: File path to the registry JSON file.

        Returns:
            A populated ``ModelRegistry`` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed as valid JSON or
                has an invalid top-level structure.
        """
        file_path = Path(path).expanduser()

        if not file_path.exists():
            raise FileNotFoundError(f"Registry file not found: {path}")

        # Reject symlinks for security.
        if file_path.is_symlink():
            raise ValueError(f"Registry file is a symlink: {path}")

        # Reject oversized files.
        try:
            file_size = file_path.stat().st_size
        except OSError as exc:
            raise ValueError(f"Cannot stat registry file: {exc}") from exc

        if file_size > _MAX_REGISTRY_SIZE:
            raise ValueError(
                f"Registry file too large: {file_size} bytes "
                f"(max {_MAX_REGISTRY_SIZE})"
            )

        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"Cannot read registry file: {exc}") from exc

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid JSON in registry file: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("Registry file must contain a JSON object")

        # Extract defaults.
        default_provider = str(data.get("default_provider", "ollama"))
        default_model = str(data.get("default_model", "qwen3:8b"))

        registry = cls(
            default_provider=default_provider,
            default_model=default_model,
        )

        # Parse task routing entries.
        task_routing = data.get("task_routing", {})
        if not isinstance(task_routing, dict):
            return registry

        # Build a lookup from task type value to TaskType enum.
        task_type_map: dict[str, TaskType] = {t.value: t for t in TaskType}

        for task_name, entries in task_routing.items():
            task_type = task_type_map.get(task_name)
            if task_type is None:
                # Unknown task type -- skip silently.
                continue

            if not isinstance(entries, list):
                continue

            for entry_data in entries:
                if not isinstance(entry_data, dict):
                    continue

                provider = entry_data.get("provider")
                model = entry_data.get("model")
                priority = entry_data.get("priority")

                # Validate each field.
                if not isinstance(provider, str):
                    continue
                if provider not in _VALID_PROVIDERS:
                    continue
                if not isinstance(model, str):
                    continue
                if not validate_model_name(model):
                    continue
                if not isinstance(priority, int) or priority < 1:
                    continue

                registry._routes.setdefault(task_type, []).append(
                    RegistryEntry(provider=provider, model=model, priority=priority)
                )

        # Sort all entries by priority.
        for entries in registry._routes.values():
            entries.sort(key=lambda e: e.priority)

        return registry
