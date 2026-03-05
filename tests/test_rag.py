"""Tests for local_cli.rag module."""

import math
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from local_cli.rag import (
    RAGEngine,
    _DEFAULT_CHUNK_OVERLAP,
    _DEFAULT_CHUNK_SIZE,
    _MAX_FILE_SIZE,
    _SKIP_DIRS,
    _chunk_text,
    _compute_file_hash,
    _cosine_similarity,
    _deserialize_embedding,
    _is_text_file,
    _serialize_embedding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client(embed_vector: list[float] | None = None) -> MagicMock:
    """Create a mock OllamaClient that returns deterministic embeddings.

    When ``embed_vector`` is provided, every call to ``client.embed()``
    returns that vector for each input string.  When ``None``, a default
    3-dimensional vector is used.
    """
    client = MagicMock()
    vec = embed_vector if embed_vector is not None else [0.1, 0.2, 0.3]

    def _embed(model: str, input_data):
        if isinstance(input_data, str):
            return [vec]
        return [vec] * len(input_data)

    client.embed.side_effect = _embed
    return client


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------


class TestComputeFileHash(unittest.TestCase):
    """Tests for _compute_file_hash()."""

    def test_consistent_hash(self) -> None:
        """Same content always produces the same hash."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("hello world\n")
            path = Path(f.name)

        try:
            h1 = _compute_file_hash(path)
            h2 = _compute_file_hash(path)
            self.assertEqual(h1, h2)
        finally:
            os.unlink(path)

    def test_different_content_different_hash(self) -> None:
        """Different content produces different hashes."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("content A")
            path_a = Path(f.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("content B")
            path_b = Path(f.name)

        try:
            self.assertNotEqual(
                _compute_file_hash(path_a), _compute_file_hash(path_b)
            )
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_returns_hex_string(self) -> None:
        """Hash is a 64-character hex string (SHA-256)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("test")
            path = Path(f.name)

        try:
            h = _compute_file_hash(path)
            self.assertEqual(len(h), 64)
            int(h, 16)  # Should not raise.
        finally:
            os.unlink(path)


class TestChunkText(unittest.TestCase):
    """Tests for _chunk_text()."""

    def test_empty_string(self) -> None:
        """Empty string returns empty list."""
        self.assertEqual(_chunk_text(""), [])

    def test_short_text_single_chunk(self) -> None:
        """Text shorter than chunk_size yields one chunk."""
        text = "short"
        chunks = _chunk_text(text, chunk_size=100, overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_chunks_have_overlap(self) -> None:
        """Adjacent chunks share overlapping characters."""
        text = "A" * 100
        chunks = _chunk_text(text, chunk_size=30, overlap=10)
        self.assertGreater(len(chunks), 1)
        # First chunk ends overlap characters before second starts.
        self.assertEqual(chunks[0][-10:], chunks[1][:10])

    def test_no_empty_chunks(self) -> None:
        """No chunk should be empty."""
        text = "x" * 500
        chunks = _chunk_text(text, chunk_size=100, overlap=20)
        for chunk in chunks:
            self.assertGreater(len(chunk), 0)


class TestIsTextFile(unittest.TestCase):
    """Tests for _is_text_file()."""

    def test_text_file(self) -> None:
        """Regular text file is detected as text."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Hello world\n")
            path = Path(f.name)

        try:
            self.assertTrue(_is_text_file(path))
        finally:
            os.unlink(path)

    def test_binary_file(self) -> None:
        """File containing null bytes is detected as binary."""
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".bin", delete=False
        ) as f:
            f.write(b"\x00\x01\x02\x03\x04")
            path = Path(f.name)

        try:
            self.assertFalse(_is_text_file(path))
        finally:
            os.unlink(path)

    def test_nonexistent_file(self) -> None:
        """Non-existent file returns False."""
        self.assertFalse(_is_text_file(Path("/tmp/does_not_exist_rag_test")))


class TestSerializeDeserializeEmbedding(unittest.TestCase):
    """Tests for _serialize_embedding() and _deserialize_embedding()."""

    def test_round_trip(self) -> None:
        """Serialize then deserialize preserves values."""
        vec = [0.1, 0.2, 0.3, -0.5, 1.0]
        data = _serialize_embedding(vec)
        result = _deserialize_embedding(data)
        self.assertEqual(len(result), len(vec))
        for a, b in zip(vec, result):
            self.assertAlmostEqual(a, b)

    def test_serialized_is_bytes(self) -> None:
        """Serialized form is bytes."""
        self.assertIsInstance(_serialize_embedding([1.0, 2.0]), bytes)


class TestCosineSimilarity(unittest.TestCase):
    """Tests for _cosine_similarity()."""

    def test_identical_vectors(self) -> None:
        """Identical vectors have similarity 1.0."""
        vec = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(_cosine_similarity(vec, vec), 1.0)

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors have similarity 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        self.assertAlmostEqual(_cosine_similarity(a, b), 0.0)

    def test_opposite_vectors(self) -> None:
        """Opposite vectors have similarity -1.0."""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        self.assertAlmostEqual(_cosine_similarity(a, b), -1.0)

    def test_zero_vector(self) -> None:
        """Zero vector returns 0.0 (no division error)."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(_cosine_similarity(a, b), 0.0)
        self.assertAlmostEqual(_cosine_similarity(b, a), 0.0)

    def test_known_value(self) -> None:
        """Test a known cosine similarity computation."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        # dot = 4+10+18 = 32
        # |a| = sqrt(14), |b| = sqrt(77)
        expected = 32.0 / (math.sqrt(14) * math.sqrt(77))
        self.assertAlmostEqual(_cosine_similarity(a, b), expected, places=6)


# ---------------------------------------------------------------------------
# Unit tests: RAGEngine schema creation
# ---------------------------------------------------------------------------


class TestRAGEngineSchemaCreation(unittest.TestCase):
    """Tests for RAGEngine SQLite schema setup."""

    def test_schema_created_on_init(self) -> None:
        """Schema is created when RAGEngine is instantiated."""
        client = _make_mock_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            engine = RAGEngine(client, db_path=db_path)

            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='chunks'"
                )
                tables = cursor.fetchall()
                conn.close()
                self.assertEqual(len(tables), 1)
                self.assertEqual(tables[0][0], "chunks")
            finally:
                engine.close()

    def test_schema_has_expected_columns(self) -> None:
        """The chunks table has the expected columns."""
        client = _make_mock_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            engine = RAGEngine(client, db_path=db_path)

            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.execute("PRAGMA table_info(chunks)")
                columns = {row[1] for row in cursor.fetchall()}
                conn.close()

                expected = {
                    "id", "file_path", "chunk_index",
                    "content", "file_hash", "embedding",
                }
                self.assertEqual(columns, expected)
            finally:
                engine.close()

    def test_indexes_created(self) -> None:
        """Expected indexes are created."""
        client = _make_mock_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            engine = RAGEngine(client, db_path=db_path)

            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                )
                index_names = {row[0] for row in cursor.fetchall()}
                conn.close()

                self.assertIn("idx_chunks_file_path", index_names)
                self.assertIn("idx_chunks_file_hash", index_names)
            finally:
                engine.close()

    def test_double_init_idempotent(self) -> None:
        """Creating RAGEngine twice on the same DB does not fail."""
        client = _make_mock_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            engine1 = RAGEngine(client, db_path=db_path)
            engine1.close()
            engine2 = RAGEngine(client, db_path=db_path)
            engine2.close()


# ---------------------------------------------------------------------------
# Unit tests: RAGEngine indexing
# ---------------------------------------------------------------------------


class TestRAGEngineIndexing(unittest.TestCase):
    """Tests for RAGEngine.index_directory() with mock embeddings."""

    def test_index_text_file(self) -> None:
        """A small text file is indexed and chunks are stored."""
        client = _make_mock_client([0.5, 0.6, 0.7])
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "rag.db")
            engine = RAGEngine(client, db_path=db_path)

            try:
                # Create a text file in a source directory.
                src_dir = os.path.join(tmpdir, "src")
                os.makedirs(src_dir)
                test_file = os.path.join(src_dir, "hello.py")
                with open(test_file, "w") as f:
                    f.write("print('hello world')\n")

                stats = engine.index_directory(src_dir)
                self.assertEqual(stats["files_indexed"], 1)
                self.assertGreater(stats["chunks_indexed"], 0)

                # Verify embed was called.
                client.embed.assert_called()
            finally:
                engine.close()

    def test_skip_binary_files(self) -> None:
        """Binary files are skipped during indexing."""
        client = _make_mock_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "rag.db")
            engine = RAGEngine(client, db_path=db_path)

            try:
                src_dir = os.path.join(tmpdir, "src")
                os.makedirs(src_dir)

                # Create a binary file.
                bin_file = os.path.join(src_dir, "image.png")
                with open(bin_file, "wb") as f:
                    f.write(b"\x89PNG\x00\x00\x00" + b"\x00" * 100)

                stats = engine.index_directory(src_dir)
                self.assertEqual(stats["files_indexed"], 0)
                self.assertEqual(stats["files_skipped"], 1)
            finally:
                engine.close()

    def test_skip_large_files(self) -> None:
        """Files exceeding _MAX_FILE_SIZE are skipped."""
        client = _make_mock_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "rag.db")
            engine = RAGEngine(client, db_path=db_path)

            try:
                src_dir = os.path.join(tmpdir, "src")
                os.makedirs(src_dir)

                # Create a file larger than 256KB.
                large_file = os.path.join(src_dir, "large.txt")
                with open(large_file, "w") as f:
                    f.write("x" * (_MAX_FILE_SIZE + 1))

                stats = engine.index_directory(src_dir)
                self.assertEqual(stats["files_indexed"], 0)
                self.assertGreater(stats["files_skipped"], 0)
            finally:
                engine.close()

    def test_skip_ignored_directories(self) -> None:
        """Directories in _SKIP_DIRS are pruned during walk."""
        client = _make_mock_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "rag.db")
            engine = RAGEngine(client, db_path=db_path)

            try:
                src_dir = os.path.join(tmpdir, "project")
                os.makedirs(src_dir)

                # Create a file in the root (should be indexed).
                root_file = os.path.join(src_dir, "main.py")
                with open(root_file, "w") as f:
                    f.write("print('main')\n")

                # Create files inside ignored directories (should NOT be indexed).
                for skip_dir in ["node_modules", "__pycache__", ".git"]:
                    ignored = os.path.join(src_dir, skip_dir)
                    os.makedirs(ignored)
                    skip_file = os.path.join(ignored, "file.py")
                    with open(skip_file, "w") as f:
                        f.write("print('ignored')\n")

                stats = engine.index_directory(src_dir)
                # Only the root file should be indexed.
                self.assertEqual(stats["files_indexed"], 1)
            finally:
                engine.close()

    def test_file_hash_change_detection(self) -> None:
        """Changed files are re-indexed; unchanged files are skipped."""
        client = _make_mock_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "rag.db")
            engine = RAGEngine(client, db_path=db_path)

            try:
                src_dir = os.path.join(tmpdir, "src")
                os.makedirs(src_dir)
                test_file = os.path.join(src_dir, "code.py")

                # First index.
                with open(test_file, "w") as f:
                    f.write("version = 1\n")
                stats1 = engine.index_directory(src_dir)
                self.assertEqual(stats1["files_indexed"], 1)

                # Second index with same content — should be unchanged.
                stats2 = engine.index_directory(src_dir)
                self.assertEqual(stats2["files_indexed"], 0)
                self.assertEqual(stats2["files_unchanged"], 1)

                # Modify the file — should be re-indexed.
                with open(test_file, "w") as f:
                    f.write("version = 2\n")
                stats3 = engine.index_directory(src_dir)
                self.assertEqual(stats3["files_indexed"], 1)
                self.assertEqual(stats3["files_unchanged"], 0)
            finally:
                engine.close()

    def test_nonexistent_directory_returns_empty_stats(self) -> None:
        """Indexing a non-existent directory returns zero counts."""
        client = _make_mock_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "rag.db")
            engine = RAGEngine(client, db_path=db_path)

            try:
                stats = engine.index_directory("/tmp/nonexistent_rag_dir_test")
                self.assertEqual(stats["files_indexed"], 0)
                self.assertEqual(stats["chunks_indexed"], 0)
            finally:
                engine.close()

    def test_embed_failure_skips_file(self) -> None:
        """When embed() raises an exception, the file is skipped."""
        client = MagicMock()
        client.embed.side_effect = RuntimeError("Ollama unavailable")

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "rag.db")
            engine = RAGEngine(client, db_path=db_path)

            try:
                src_dir = os.path.join(tmpdir, "src")
                os.makedirs(src_dir)
                test_file = os.path.join(src_dir, "code.py")
                with open(test_file, "w") as f:
                    f.write("print('hello')\n")

                stats = engine.index_directory(src_dir)
                self.assertEqual(stats["files_indexed"], 0)
                self.assertEqual(stats["files_skipped"], 1)
            finally:
                engine.close()

    def test_multiple_files_indexed(self) -> None:
        """Multiple text files in a directory tree are indexed."""
        client = _make_mock_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "rag.db")
            engine = RAGEngine(client, db_path=db_path)

            try:
                src_dir = os.path.join(tmpdir, "project")
                sub_dir = os.path.join(src_dir, "lib")
                os.makedirs(sub_dir)

                for name, directory in [
                    ("main.py", src_dir),
                    ("utils.py", sub_dir),
                    ("helpers.py", sub_dir),
                ]:
                    with open(os.path.join(directory, name), "w") as f:
                        f.write(f"# {name}\n")

                stats = engine.index_directory(src_dir)
                self.assertEqual(stats["files_indexed"], 3)
            finally:
                engine.close()


# ---------------------------------------------------------------------------
# Unit tests: RAGEngine querying
# ---------------------------------------------------------------------------


class TestRAGEngineQuery(unittest.TestCase):
    """Tests for RAGEngine.query() with mock embeddings."""

    def _create_engine_with_data(
        self, tmpdir: str
    ) -> tuple["RAGEngine", MagicMock]:
        """Create a RAGEngine and index some test files.

        Stores vectors with known values so we can predict cosine
        similarity rankings.
        """
        # We'll manually insert chunks with known embeddings to
        # control the similarity results precisely.
        db_path = os.path.join(tmpdir, "rag.db")
        client = MagicMock()
        engine = RAGEngine(client, db_path=db_path)

        # Insert test chunks directly into the database.
        test_data = [
            ("file_a.py", 0, "def hello():", [1.0, 0.0, 0.0]),
            ("file_a.py", 1, "    return 'hi'", [0.9, 0.1, 0.0]),
            ("file_b.py", 0, "import math", [0.0, 1.0, 0.0]),
            ("file_c.py", 0, "class Foo:", [0.0, 0.0, 1.0]),
            ("file_d.py", 0, "x = 42", [0.5, 0.5, 0.0]),
        ]

        for fpath, idx, content, vec in test_data:
            engine._conn.execute(
                "INSERT INTO chunks "
                "(file_path, chunk_index, content, file_hash, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                (fpath, idx, content, "testhash", _serialize_embedding(vec)),
            )
        engine._conn.commit()

        return engine, client

    def test_query_returns_top_k_results(self) -> None:
        """Query returns at most top_k results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine, client = self._create_engine_with_data(tmpdir)
            # Make embed return a vector similar to file_a's chunks.
            client.embed.return_value = [[1.0, 0.0, 0.0]]

            try:
                results = engine.query("hello function", top_k=2)
                self.assertEqual(len(results), 2)
            finally:
                engine.close()

    def test_query_ranking_by_cosine_similarity(self) -> None:
        """Results are ranked by cosine similarity (highest first)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine, client = self._create_engine_with_data(tmpdir)
            # Query vector [1, 0, 0] — most similar to file_a chunks.
            client.embed.return_value = [[1.0, 0.0, 0.0]]

            try:
                results = engine.query("hello", top_k=5)
                # First result should be file_a.py chunk 0 (exact match).
                self.assertEqual(results[0]["file_path"], "file_a.py")
                self.assertEqual(results[0]["chunk_index"], 0)
                self.assertAlmostEqual(results[0]["score"], 1.0)

                # Scores should be in descending order.
                for i in range(len(results) - 1):
                    self.assertGreaterEqual(
                        results[i]["score"], results[i + 1]["score"]
                    )
            finally:
                engine.close()

    def test_query_result_structure(self) -> None:
        """Each result contains expected keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine, client = self._create_engine_with_data(tmpdir)
            client.embed.return_value = [[0.5, 0.5, 0.0]]

            try:
                results = engine.query("test", top_k=1)
                self.assertEqual(len(results), 1)
                result = results[0]
                self.assertIn("file_path", result)
                self.assertIn("chunk_index", result)
                self.assertIn("content", result)
                self.assertIn("score", result)
            finally:
                engine.close()

    def test_query_empty_index(self) -> None:
        """Querying an empty index returns no results."""
        client = _make_mock_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "rag.db")
            engine = RAGEngine(client, db_path=db_path)

            try:
                results = engine.query("anything", top_k=5)
                self.assertEqual(results, [])
            finally:
                engine.close()

    def test_query_embed_failure_returns_empty(self) -> None:
        """When embed() fails during query, returns empty list."""
        client = MagicMock()
        client.embed.side_effect = RuntimeError("Ollama error")

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "rag.db")
            engine = RAGEngine(client, db_path=db_path)

            try:
                results = engine.query("test", top_k=5)
                self.assertEqual(results, [])
            finally:
                engine.close()


# ---------------------------------------------------------------------------
# Unit tests: RAGEngine augment_prompt
# ---------------------------------------------------------------------------


class TestRAGEngineAugmentPrompt(unittest.TestCase):
    """Tests for RAGEngine.augment_prompt()."""

    def test_returns_original_when_no_results(self) -> None:
        """When index is empty, returns the original prompt unchanged."""
        client = _make_mock_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "rag.db")
            engine = RAGEngine(client, db_path=db_path)

            try:
                result = engine.augment_prompt("What does this do?")
                self.assertEqual(result, "What does this do?")
            finally:
                engine.close()

    def test_includes_context_when_results_exist(self) -> None:
        """When results exist, the augmented prompt includes context."""
        client = MagicMock()
        client.embed.return_value = [[1.0, 0.0, 0.0]]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "rag.db")
            engine = RAGEngine(client, db_path=db_path)

            try:
                # Insert a test chunk.
                engine._conn.execute(
                    "INSERT INTO chunks "
                    "(file_path, chunk_index, content, file_hash, embedding) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        "test.py",
                        0,
                        "def foo(): pass",
                        "hash",
                        _serialize_embedding([1.0, 0.0, 0.0]),
                    ),
                )
                engine._conn.commit()

                result = engine.augment_prompt("What is foo?")
                self.assertIn("relevant context", result)
                self.assertIn("def foo(): pass", result)
                self.assertIn("User question: What is foo?", result)
            finally:
                engine.close()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRAGEngineIntegration(unittest.TestCase):
    """Integration test: index and query with mock client."""

    def test_index_then_query(self) -> None:
        """Index files, then query and get relevant results."""
        # Use different vectors for different files to test ranking.
        call_count = 0

        def _embed(model, input_data):
            nonlocal call_count
            call_count += 1
            if isinstance(input_data, str):
                # Query embedding — similar to "python" files.
                return [[0.9, 0.1, 0.0]]
            # Return distinct vectors for each chunk.
            vectors = []
            for _ in input_data:
                vectors.append([0.8, 0.2, 0.0])
            return vectors

        client = MagicMock()
        client.embed.side_effect = _embed

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "rag.db")
            engine = RAGEngine(client, db_path=db_path)

            try:
                # Create source files.
                src_dir = os.path.join(tmpdir, "src")
                os.makedirs(src_dir)
                with open(os.path.join(src_dir, "app.py"), "w") as f:
                    f.write("def main():\n    print('hello')\n")

                # Index.
                stats = engine.index_directory(src_dir)
                self.assertGreater(stats["files_indexed"], 0)

                # Query.
                results = engine.query("python main function", top_k=3)
                self.assertGreater(len(results), 0)
                self.assertIn("content", results[0])
            finally:
                engine.close()


if __name__ == "__main__":
    unittest.main()
