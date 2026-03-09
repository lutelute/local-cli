"""Knowledge persistence for local-cli.

Provides :class:`KnowledgeStore` for saving, loading, listing, and deleting
persistent knowledge items.  Each item is stored as a subdirectory
containing a ``metadata.json`` file and one or more markdown artifact files.
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KnowledgeError(Exception):
    """Base exception for knowledge operations."""


class KnowledgeNotFoundError(KnowledgeError):
    """Raised when a referenced knowledge item does not exist."""


# ---------------------------------------------------------------------------
# KnowledgeStore
# ---------------------------------------------------------------------------


class KnowledgeStore:
    """Manages persistent knowledge items as JSON metadata + markdown artifacts.

    Knowledge items are stored in subdirectories of a configurable base
    directory (default ``.agents/knowledge/``).  Each item directory contains
    a ``metadata.json`` file describing the item and one or more markdown
    artifact files.

    Directory layout::

        .agents/knowledge/
            item-name/
                metadata.json
                README.md
                notes.md

    Args:
        knowledge_dir: Base directory for knowledge storage.  Created
            automatically on first save if it does not exist.
    """

    # Metadata filename within each item directory.
    _METADATA_FILE = "metadata.json"

    # Default artifact filename when saving a single content string.
    _DEFAULT_ARTIFACT = "README.md"

    def __init__(self, knowledge_dir: str = ".agents/knowledge") -> None:
        self._knowledge_dir = Path(knowledge_dir).expanduser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_item(
        self,
        name: str,
        description: str = "",
        content: str = "",
        tags: list[str] | None = None,
    ) -> dict:
        """Save a knowledge item to disk.

        Creates (or overwrites) a knowledge item directory containing
        ``metadata.json`` and a default markdown artifact.

        Args:
            name: Unique item name (used as the directory name).
            description: Brief description of the knowledge item.
            content: Markdown content to store as the default artifact.
            tags: Optional list of tag strings for categorisation.

        Returns:
            The metadata dictionary that was persisted.

        Raises:
            KnowledgeError: If the item cannot be written.
        """
        if not name or not name.strip():
            raise KnowledgeError("Knowledge item name must not be empty.")

        name = name.strip()
        item_dir = self._item_dir(name)
        item_dir.mkdir(parents=True, exist_ok=True)

        artifacts = [self._DEFAULT_ARTIFACT] if content else []

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        metadata: dict = {
            "name": name,
            "description": description,
            "created": now,
            "tags": tags if tags is not None else [],
            "artifacts": artifacts,
        }

        # Write metadata atomically.
        self._write_json(item_dir / self._METADATA_FILE, metadata)

        # Write default artifact if content provided.
        if content:
            self._write_text(item_dir / self._DEFAULT_ARTIFACT, content)

        return metadata

    def load_item(self, name: str) -> dict:
        """Load a knowledge item's metadata and artifact contents.

        Args:
            name: The item name to load.

        Returns:
            A dictionary containing the metadata fields plus an
            ``"artifacts_content"`` mapping of filename to content string.

        Raises:
            KnowledgeNotFoundError: If the item does not exist.
            KnowledgeError: If the item cannot be read.
        """
        name = name.strip()
        item_dir = self._item_dir(name)
        metadata_path = item_dir / self._METADATA_FILE

        if not metadata_path.exists():
            raise KnowledgeNotFoundError(
                f"Knowledge item '{name}' not found."
            )

        metadata = self._read_json(metadata_path)

        # Load artifact contents.
        artifacts_content: dict[str, str] = {}
        for artifact_name in metadata.get("artifacts", []):
            artifact_path = item_dir / artifact_name
            if artifact_path.exists():
                try:
                    artifacts_content[artifact_name] = artifact_path.read_text(
                        encoding="utf-8"
                    )
                except OSError:
                    # Skip unreadable artifacts gracefully.
                    continue

        result = dict(metadata)
        result["artifacts_content"] = artifacts_content
        return result

    def list_items(self) -> list[dict]:
        """Return metadata for all knowledge items, sorted by name.

        Items with missing or corrupt metadata are silently skipped.

        Returns:
            A list of metadata dictionaries, sorted alphabetically by name.
        """
        if not self._knowledge_dir.is_dir():
            return []

        items: list[dict] = []
        try:
            for entry in self._knowledge_dir.iterdir():
                if entry.is_dir():
                    metadata_path = entry / self._METADATA_FILE
                    if metadata_path.exists():
                        try:
                            metadata = self._read_json(metadata_path)
                            items.append(metadata)
                        except KnowledgeError:
                            # Skip corrupt metadata gracefully.
                            continue
        except OSError:
            return []

        items.sort(key=lambda m: m.get("name", ""))
        return items

    def delete_item(self, name: str) -> None:
        """Delete a knowledge item and its directory.

        Args:
            name: The item name to delete.

        Raises:
            KnowledgeNotFoundError: If the item does not exist.
            KnowledgeError: If the item cannot be deleted.
        """
        name = name.strip()
        item_dir = self._item_dir(name)

        if not item_dir.is_dir():
            raise KnowledgeNotFoundError(
                f"Knowledge item '{name}' not found."
            )

        try:
            shutil.rmtree(str(item_dir))
        except OSError as exc:
            raise KnowledgeError(
                f"Failed to delete knowledge item '{name}': {exc}"
            )

    def add_artifact(
        self,
        name: str,
        artifact_name: str,
        content: str,
    ) -> dict:
        """Add an artifact file to an existing knowledge item.

        Args:
            name: The knowledge item name.
            artifact_name: Filename for the artifact (e.g. ``"notes.md"``).
            content: Artifact content string.

        Returns:
            The updated metadata dictionary.

        Raises:
            KnowledgeNotFoundError: If the item does not exist.
            KnowledgeError: If the artifact cannot be written.
        """
        name = name.strip()
        item_dir = self._item_dir(name)
        metadata_path = item_dir / self._METADATA_FILE

        if not metadata_path.exists():
            raise KnowledgeNotFoundError(
                f"Knowledge item '{name}' not found."
            )

        metadata = self._read_json(metadata_path)

        # Write artifact file.
        self._write_text(item_dir / artifact_name, content)

        # Update metadata artifacts list.
        if artifact_name not in metadata.get("artifacts", []):
            metadata.setdefault("artifacts", []).append(artifact_name)
            self._write_json(metadata_path, metadata)

        return metadata

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _item_dir(self, name: str) -> Path:
        """Return the directory path for a given knowledge item name."""
        return self._knowledge_dir / name

    def _write_json(self, path: Path, data: dict) -> None:
        """Write a JSON file atomically (temp file + rename).

        Args:
            path: Target file path.
            data: Dictionary to serialise as JSON.

        Raises:
            KnowledgeError: If the file cannot be written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent),
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, indent=2)
                    fh.write("\n")
                os.replace(tmp_path, str(path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as exc:
            raise KnowledgeError(f"Failed to write '{path}': {exc}")

    def _write_text(self, path: Path, content: str) -> None:
        """Write a text file atomically (temp file + rename).

        Args:
            path: Target file path.
            content: Text content to write.

        Raises:
            KnowledgeError: If the file cannot be written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent),
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(content)
                os.replace(tmp_path, str(path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as exc:
            raise KnowledgeError(f"Failed to write '{path}': {exc}")

    def _read_json(self, path: Path) -> dict:
        """Read and parse a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            The parsed dictionary.

        Raises:
            KnowledgeError: If the file cannot be read or parsed.
        """
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise KnowledgeError(f"Failed to read '{path}': {exc}")

        try:
            data = json.loads(content)
        except (json.JSONDecodeError, ValueError) as exc:
            raise KnowledgeError(f"Invalid JSON in '{path}': {exc}")

        if not isinstance(data, dict):
            raise KnowledgeError(
                f"Expected JSON object in '{path}', got {type(data).__name__}."
            )

        return data
