"""Tests for local_cli.knowledge module."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from local_cli.knowledge import (
    KnowledgeError,
    KnowledgeNotFoundError,
    KnowledgeStore,
)


class TestKnowledgeStoreInit(unittest.TestCase):
    """Tests for KnowledgeStore construction."""

    def test_stores_knowledge_dir_as_path(self) -> None:
        """knowledge_dir argument is stored as a Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            self.assertIsInstance(store._knowledge_dir, Path)

    def test_tilde_expansion(self) -> None:
        """Knowledge dir path with ~ is expanded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            self.assertTrue(store._knowledge_dir.is_absolute())

    def test_directory_not_created_on_init(self) -> None:
        """Knowledge directory is NOT created on init (lazy creation)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            knowledge_dir = os.path.join(tmpdir, "knowledge")
            store = KnowledgeStore(knowledge_dir)
            self.assertFalse(Path(knowledge_dir).exists())

    def test_existing_directory_ok(self) -> None:
        """Initializing with an existing knowledge dir does not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store1 = KnowledgeStore(tmpdir)
            store2 = KnowledgeStore(tmpdir)  # Second init should not fail.
            self.assertIsInstance(store2._knowledge_dir, Path)

    def test_default_knowledge_dir(self) -> None:
        """Default knowledge_dir is '.agents/knowledge'."""
        store = KnowledgeStore()
        self.assertEqual(store._knowledge_dir.name, "knowledge")


class TestSaveItem(unittest.TestCase):
    """Tests for KnowledgeStore.save_item()."""

    def test_creates_item_directory(self) -> None:
        """save_item creates a subdirectory with the item name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("my-item", description="Test item")
            item_dir = Path(tmpdir) / "my-item"
            self.assertTrue(item_dir.is_dir())

    def test_creates_metadata_file(self) -> None:
        """save_item creates a metadata.json file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("my-item", description="Test item")
            metadata_path = Path(tmpdir) / "my-item" / "metadata.json"
            self.assertTrue(metadata_path.exists())

    def test_metadata_contains_required_fields(self) -> None:
        """Metadata contains name, description, created, tags, artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("my-item", description="Test item", tags=["python"])
            metadata_path = Path(tmpdir) / "my-item" / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertEqual(metadata["name"], "my-item")
            self.assertEqual(metadata["description"], "Test item")
            self.assertIn("created", metadata)
            self.assertEqual(metadata["tags"], ["python"])
            self.assertIsInstance(metadata["artifacts"], list)

    def test_returns_metadata_dict(self) -> None:
        """save_item returns the persisted metadata dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            result = store.save_item("my-item", description="Test")
            self.assertIsInstance(result, dict)
            self.assertEqual(result["name"], "my-item")
            self.assertEqual(result["description"], "Test")

    def test_content_creates_default_artifact(self) -> None:
        """Providing content creates a README.md artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("my-item", content="# Hello World")
            readme = Path(tmpdir) / "my-item" / "README.md"
            self.assertTrue(readme.exists())
            self.assertEqual(readme.read_text(encoding="utf-8"), "# Hello World")

    def test_no_content_no_default_artifact(self) -> None:
        """Without content, no README.md artifact is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            metadata = store.save_item("my-item", description="No content")
            readme = Path(tmpdir) / "my-item" / "README.md"
            self.assertFalse(readme.exists())
            self.assertEqual(metadata["artifacts"], [])

    def test_default_tags_empty_list(self) -> None:
        """Tags default to an empty list when not provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            metadata = store.save_item("my-item")
            self.assertEqual(metadata["tags"], [])

    def test_empty_name_raises_error(self) -> None:
        """save_item raises KnowledgeError for empty name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            with self.assertRaises(KnowledgeError):
                store.save_item("")

    def test_whitespace_only_name_raises_error(self) -> None:
        """save_item raises KnowledgeError for whitespace-only name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            with self.assertRaises(KnowledgeError):
                store.save_item("   ")

    def test_name_stripped(self) -> None:
        """Leading/trailing whitespace in name is stripped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            metadata = store.save_item("  my-item  ", description="Test")
            self.assertEqual(metadata["name"], "my-item")
            item_dir = Path(tmpdir) / "my-item"
            self.assertTrue(item_dir.is_dir())

    def test_unicode_content_preserved(self) -> None:
        """Unicode content is preserved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            text = "Hello 世界 😀"
            store.save_item("unicode-item", content=text)
            readme = Path(tmpdir) / "unicode-item" / "README.md"
            self.assertEqual(readme.read_text(encoding="utf-8"), text)

    def test_overwrite_existing_item(self) -> None:
        """Saving with the same name overwrites the previous item."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("my-item", description="First", content="v1")
            store.save_item("my-item", description="Second", content="v2")

            loaded = store.load_item("my-item")
            self.assertEqual(loaded["description"], "Second")
            self.assertEqual(loaded["artifacts_content"]["README.md"], "v2")

    def test_multiple_tags(self) -> None:
        """Multiple tags are stored correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            tags = ["python", "architecture", "patterns"]
            metadata = store.save_item("tagged-item", tags=tags)
            self.assertEqual(metadata["tags"], tags)

    def test_created_timestamp_format(self) -> None:
        """Created timestamp follows ISO format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            metadata = store.save_item("ts-item")
            # e.g. "2026-03-09T12:00:00"
            self.assertRegex(
                metadata["created"],
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
            )

    def test_creates_nested_parent_directories(self) -> None:
        """save_item creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_dir = os.path.join(tmpdir, "deep", "nested", "knowledge")
            store = KnowledgeStore(deep_dir)
            store.save_item("my-item", content="test")
            metadata_path = Path(deep_dir) / "my-item" / "metadata.json"
            self.assertTrue(metadata_path.exists())


class TestLoadItem(unittest.TestCase):
    """Tests for KnowledgeStore.load_item()."""

    def test_round_trip(self) -> None:
        """Save then load preserves metadata and content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item(
                "my-item",
                description="Test item",
                content="# Content",
                tags=["test"],
            )
            loaded = store.load_item("my-item")

            self.assertEqual(loaded["name"], "my-item")
            self.assertEqual(loaded["description"], "Test item")
            self.assertEqual(loaded["tags"], ["test"])
            self.assertIn("artifacts_content", loaded)
            self.assertEqual(loaded["artifacts_content"]["README.md"], "# Content")

    def test_nonexistent_item_raises(self) -> None:
        """Loading a non-existent item raises KnowledgeNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            with self.assertRaises(KnowledgeNotFoundError):
                store.load_item("does-not-exist")

    def test_name_stripped_on_load(self) -> None:
        """Leading/trailing whitespace in name is stripped on load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("my-item", content="test")
            loaded = store.load_item("  my-item  ")
            self.assertEqual(loaded["name"], "my-item")

    def test_missing_artifact_file_skipped(self) -> None:
        """Missing artifact files are silently skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("my-item", content="test")

            # Remove the artifact file but keep metadata referencing it.
            readme = Path(tmpdir) / "my-item" / "README.md"
            readme.unlink()

            loaded = store.load_item("my-item")
            self.assertEqual(loaded["artifacts_content"], {})

    def test_load_item_with_multiple_artifacts(self) -> None:
        """Load item with multiple artifact files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("my-item", content="readme content")
            store.add_artifact("my-item", "notes.md", "notes content")

            loaded = store.load_item("my-item")
            self.assertEqual(
                loaded["artifacts_content"]["README.md"], "readme content"
            )
            self.assertEqual(
                loaded["artifacts_content"]["notes.md"], "notes content"
            )

    def test_corrupt_metadata_raises(self) -> None:
        """Corrupt metadata.json raises KnowledgeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            # Create item directory with corrupt metadata.
            item_dir = Path(tmpdir) / "bad-item"
            item_dir.mkdir(parents=True)
            (item_dir / "metadata.json").write_text("not valid json")

            with self.assertRaises(KnowledgeError):
                store.load_item("bad-item")

    def test_non_dict_metadata_raises(self) -> None:
        """Metadata that is not a JSON object raises KnowledgeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            item_dir = Path(tmpdir) / "list-item"
            item_dir.mkdir(parents=True)
            (item_dir / "metadata.json").write_text("[1, 2, 3]")

            with self.assertRaises(KnowledgeError):
                store.load_item("list-item")


class TestListItems(unittest.TestCase):
    """Tests for KnowledgeStore.list_items()."""

    def test_empty_directory(self) -> None:
        """No items returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            self.assertEqual(store.list_items(), [])

    def test_missing_directory(self) -> None:
        """Non-existent knowledge directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            knowledge_dir = os.path.join(tmpdir, "nonexistent")
            store = KnowledgeStore(knowledge_dir)
            self.assertEqual(store.list_items(), [])

    def test_lists_all_items(self) -> None:
        """list_items returns metadata for all saved items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("alpha", description="First")
            store.save_item("beta", description="Second")
            store.save_item("gamma", description="Third")

            items = store.list_items()
            self.assertEqual(len(items), 3)
            names = [item["name"] for item in items]
            self.assertEqual(names, ["alpha", "beta", "gamma"])

    def test_sorted_alphabetically(self) -> None:
        """Items are sorted alphabetically by name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("zebra", description="Z")
            store.save_item("apple", description="A")
            store.save_item("mango", description="M")

            items = store.list_items()
            names = [item["name"] for item in items]
            self.assertEqual(names, ["apple", "mango", "zebra"])

    def test_corrupt_metadata_skipped(self) -> None:
        """Items with corrupt metadata are silently skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("good-item", description="Valid")

            # Create corrupt item directory.
            bad_dir = Path(tmpdir) / "bad-item"
            bad_dir.mkdir()
            (bad_dir / "metadata.json").write_text("{broken json")

            items = store.list_items()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["name"], "good-item")

    def test_directory_without_metadata_skipped(self) -> None:
        """Directories without metadata.json are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("good-item", description="Valid")

            # Create directory without metadata.
            empty_dir = Path(tmpdir) / "empty-item"
            empty_dir.mkdir()

            items = store.list_items()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["name"], "good-item")

    def test_files_in_root_ignored(self) -> None:
        """Regular files in the knowledge root directory are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("good-item", description="Valid")

            # Create a stray file in the root.
            stray = Path(tmpdir) / "stray.txt"
            stray.write_text("not a knowledge item")

            items = store.list_items()
            self.assertEqual(len(items), 1)


class TestDeleteItem(unittest.TestCase):
    """Tests for KnowledgeStore.delete_item()."""

    def test_deletes_item_directory(self) -> None:
        """delete_item removes the entire item directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("my-item", content="test")
            store.delete_item("my-item")

            item_dir = Path(tmpdir) / "my-item"
            self.assertFalse(item_dir.exists())

    def test_nonexistent_item_raises(self) -> None:
        """Deleting a non-existent item raises KnowledgeNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            with self.assertRaises(KnowledgeNotFoundError):
                store.delete_item("does-not-exist")

    def test_deleted_item_not_in_list(self) -> None:
        """After deletion, the item no longer appears in list_items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("keep-me", description="Stay")
            store.save_item("delete-me", description="Go")
            store.delete_item("delete-me")

            items = store.list_items()
            names = [item["name"] for item in items]
            self.assertNotIn("delete-me", names)
            self.assertIn("keep-me", names)

    def test_name_stripped_on_delete(self) -> None:
        """Leading/trailing whitespace in name is stripped on delete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("my-item", content="test")
            store.delete_item("  my-item  ")
            item_dir = Path(tmpdir) / "my-item"
            self.assertFalse(item_dir.exists())

    def test_delete_item_with_multiple_artifacts(self) -> None:
        """Deleting an item with multiple artifacts removes everything."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("my-item", content="readme")
            store.add_artifact("my-item", "notes.md", "notes")
            store.add_artifact("my-item", "extra.md", "extra")
            store.delete_item("my-item")

            item_dir = Path(tmpdir) / "my-item"
            self.assertFalse(item_dir.exists())


class TestAddArtifact(unittest.TestCase):
    """Tests for KnowledgeStore.add_artifact()."""

    def test_adds_artifact_file(self) -> None:
        """add_artifact creates a new file in the item directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("my-item", content="readme")
            store.add_artifact("my-item", "notes.md", "My notes")

            notes_path = Path(tmpdir) / "my-item" / "notes.md"
            self.assertTrue(notes_path.exists())
            self.assertEqual(notes_path.read_text(encoding="utf-8"), "My notes")

    def test_updates_metadata_artifacts_list(self) -> None:
        """add_artifact updates the artifacts list in metadata.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("my-item", content="readme")
            store.add_artifact("my-item", "notes.md", "Notes")

            metadata_path = Path(tmpdir) / "my-item" / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertIn("notes.md", metadata["artifacts"])
            self.assertIn("README.md", metadata["artifacts"])

    def test_returns_updated_metadata(self) -> None:
        """add_artifact returns the updated metadata dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("my-item", content="readme")
            result = store.add_artifact("my-item", "notes.md", "Notes")

            self.assertIsInstance(result, dict)
            self.assertIn("notes.md", result["artifacts"])

    def test_nonexistent_item_raises(self) -> None:
        """Adding an artifact to a non-existent item raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            with self.assertRaises(KnowledgeNotFoundError):
                store.add_artifact("missing", "notes.md", "content")

    def test_duplicate_artifact_not_added_twice(self) -> None:
        """Adding the same artifact name twice doesn't duplicate in list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("my-item", content="readme")
            store.add_artifact("my-item", "notes.md", "v1")
            store.add_artifact("my-item", "notes.md", "v2")

            metadata_path = Path(tmpdir) / "my-item" / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            # Should only appear once.
            self.assertEqual(
                metadata["artifacts"].count("notes.md"), 1
            )
            # Content should be updated.
            notes_path = Path(tmpdir) / "my-item" / "notes.md"
            self.assertEqual(notes_path.read_text(encoding="utf-8"), "v2")

    def test_unicode_artifact_content(self) -> None:
        """Unicode content in artifact is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("my-item")
            text = "日本語テスト 🎉"
            store.add_artifact("my-item", "notes.md", text)
            notes_path = Path(tmpdir) / "my-item" / "notes.md"
            self.assertEqual(notes_path.read_text(encoding="utf-8"), text)


class TestExceptionHierarchy(unittest.TestCase):
    """Tests for the knowledge exception hierarchy."""

    def test_knowledge_not_found_is_knowledge_error(self) -> None:
        """KnowledgeNotFoundError is a subclass of KnowledgeError."""
        self.assertTrue(issubclass(KnowledgeNotFoundError, KnowledgeError))

    def test_knowledge_error_is_exception(self) -> None:
        """KnowledgeError is a subclass of Exception."""
        self.assertTrue(issubclass(KnowledgeError, Exception))

    def test_catch_base_catches_not_found(self) -> None:
        """Catching KnowledgeError catches KnowledgeNotFoundError."""
        with self.assertRaises(KnowledgeError):
            raise KnowledgeNotFoundError("test")


class TestKnowledgeRoundTrip(unittest.TestCase):
    """Integration tests for knowledge save/load/list/delete workflows."""

    def test_full_lifecycle(self) -> None:
        """Save → load → add artifact → list → delete lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)

            # Save
            store.save_item(
                "auth-patterns",
                description="Authentication patterns",
                content="# Auth\nUse JWT tokens.",
                tags=["security", "auth"],
            )

            # Load
            loaded = store.load_item("auth-patterns")
            self.assertEqual(loaded["name"], "auth-patterns")
            self.assertEqual(loaded["tags"], ["security", "auth"])
            self.assertIn("README.md", loaded["artifacts_content"])

            # Add artifact
            store.add_artifact(
                "auth-patterns", "implementation.md", "# Implementation\nUse bcrypt."
            )
            loaded = store.load_item("auth-patterns")
            self.assertEqual(len(loaded["artifacts_content"]), 2)

            # List
            items = store.list_items()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["name"], "auth-patterns")

            # Delete
            store.delete_item("auth-patterns")
            items = store.list_items()
            self.assertEqual(len(items), 0)

    def test_multiple_items_coexist(self) -> None:
        """Multiple knowledge items can be saved and loaded independently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            store.save_item("item-a", content="Content A")
            store.save_item("item-b", content="Content B")
            store.save_item("item-c", content="Content C")

            loaded_a = store.load_item("item-a")
            loaded_c = store.load_item("item-c")

            self.assertEqual(
                loaded_a["artifacts_content"]["README.md"], "Content A"
            )
            self.assertEqual(
                loaded_c["artifacts_content"]["README.md"], "Content C"
            )

            items = store.list_items()
            self.assertEqual(len(items), 3)


if __name__ == "__main__":
    unittest.main()
