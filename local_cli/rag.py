"""RAG (Retrieval-Augmented Generation) engine for local-cli.

Uses SQLite for vector storage and Ollama embeddings for semantic search.
Indexes local codebases and retrieves relevant chunks to augment prompts.
"""

import hashlib
import math
import os
import sqlite3
from pathlib import Path
from typing import Any

from local_cli.ollama_client import OllamaClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum file size to index (256KB).
_MAX_FILE_SIZE = 256 * 1024

# Default chunk size in characters.
_DEFAULT_CHUNK_SIZE = 1000

# Default overlap between chunks in characters.
_DEFAULT_CHUNK_OVERLAP = 200

# Directories to skip during indexing.
_SKIP_DIRS = frozenset({
    ".git",
    "node_modules",
    "__pycache__",
    "venv",
    ".venv",
    "env",
    ".env",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
})

# Default embedding model.
_DEFAULT_EMBED_MODEL = "all-minilm"

# Default number of results to return.
_DEFAULT_TOP_K = 5

# SQLite database filename.
_DB_FILENAME = "rag_index.db"


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    embedding BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_file_path
    ON chunks (file_path);

CREATE INDEX IF NOT EXISTS idx_chunks_file_hash
    ON chunks (file_path, file_hash);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file's contents.

    Args:
        file_path: Path to the file.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _chunk_text(
    text: str,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
        # Avoid tiny trailing chunks.
        if start + overlap >= text_len:
            break

    return chunks


def _is_text_file(file_path: Path) -> bool:
    """Heuristic check whether a file is a text file.

    Reads the first 8KB and checks for null bytes (a strong indicator
    of binary content).

    Args:
        file_path: Path to the file.

    Returns:
        True if the file appears to be text, False otherwise.
    """
    try:
        with open(file_path, "rb") as f:
            sample = f.read(8192)
        return b"\x00" not in sample
    except OSError:
        return False


def _serialize_embedding(vector: list[float]) -> bytes:
    """Serialize an embedding vector to bytes for SQLite storage.

    Uses a simple format: comma-separated float strings encoded as UTF-8.
    This avoids importing ``struct`` for packing and keeps the data
    human-readable in the database.

    Args:
        vector: List of float values.

    Returns:
        UTF-8 encoded bytes of comma-separated floats.
    """
    return ",".join(str(v) for v in vector).encode("utf-8")


def _deserialize_embedding(data: bytes) -> list[float]:
    """Deserialize an embedding vector from bytes.

    Args:
        data: UTF-8 encoded bytes of comma-separated floats.

    Returns:
        List of float values.
    """
    return [float(v) for v in data.decode("utf-8").split(",")]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity score in [-1, 1].
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------


class RAGEngine:
    """Retrieval-Augmented Generation engine using SQLite and Ollama embeddings.

    Indexes a local directory by chunking text files, computing embeddings
    via Ollama, and storing them in a SQLite database.  Supports semantic
    search over the indexed content using cosine similarity.

    Args:
        client: An :class:`OllamaClient` instance for computing embeddings.
        db_path: Path to the SQLite database file.  If ``None``, defaults
            to ``rag_index.db`` in the current directory.
        embedding_model: Name of the Ollama embedding model to use.
    """

    def __init__(
        self,
        client: OllamaClient,
        db_path: str | None = None,
        embedding_model: str = _DEFAULT_EMBED_MODEL,
    ) -> None:
        self.client = client
        self.embedding_model = embedding_model

        if db_path is None:
            db_path = _DB_FILENAME

        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._migrate()

    def _migrate(self) -> None:
        """Create or migrate the SQLite schema."""
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        """Close the SQLite database connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_directory(
        self,
        path: str,
        embedding_model: str | None = None,
    ) -> dict[str, Any]:
        """Walk a directory and index text files into the database.

        Skips directories listed in ``_SKIP_DIRS``, files larger than
        ``_MAX_FILE_SIZE``, and binary files.  Uses SHA-256 hashing for
        change detection — files that haven't changed since last indexing
        are skipped.

        Args:
            path: Root directory to index.
            embedding_model: Override the embedding model for this operation.
                If ``None``, uses the engine's default model.

        Returns:
            A summary dictionary with counts of files processed, skipped,
            and chunks indexed.
        """
        model = embedding_model or self.embedding_model
        root = Path(path).resolve()

        stats: dict[str, int] = {
            "files_indexed": 0,
            "files_skipped": 0,
            "files_unchanged": 0,
            "chunks_indexed": 0,
        }

        if not root.is_dir():
            return stats

        for dirpath, dirnames, filenames in os.walk(root):
            # Prune skipped directories (modifying dirnames in-place).
            dirnames[:] = [
                d for d in dirnames if d not in _SKIP_DIRS
            ]

            for filename in filenames:
                file_path = Path(dirpath) / filename

                # Skip files that are too large.
                try:
                    file_size = file_path.stat().st_size
                except OSError:
                    stats["files_skipped"] += 1
                    continue

                if file_size > _MAX_FILE_SIZE:
                    stats["files_skipped"] += 1
                    continue

                # Skip non-text files.
                if not _is_text_file(file_path):
                    stats["files_skipped"] += 1
                    continue

                # Compute file hash for change detection.
                try:
                    file_hash = _compute_file_hash(file_path)
                except OSError:
                    stats["files_skipped"] += 1
                    continue

                # Check if this file version is already indexed.
                rel_path = str(file_path)
                cursor = self._conn.execute(
                    "SELECT COUNT(*) FROM chunks "
                    "WHERE file_path = ? AND file_hash = ?",
                    (rel_path, file_hash),
                )
                count = cursor.fetchone()[0]
                if count > 0:
                    stats["files_unchanged"] += 1
                    continue

                # Read file content.
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    stats["files_skipped"] += 1
                    continue

                # Chunk the content.
                chunks = _chunk_text(content)
                if not chunks:
                    stats["files_skipped"] += 1
                    continue

                # Compute embeddings for all chunks.
                try:
                    embeddings = self.client.embed(model, chunks)
                except Exception:
                    stats["files_skipped"] += 1
                    continue

                if len(embeddings) != len(chunks):
                    stats["files_skipped"] += 1
                    continue

                # Remove old entries for this file path (content has changed).
                self._conn.execute(
                    "DELETE FROM chunks WHERE file_path = ?",
                    (rel_path,),
                )

                # Insert new chunks.
                for i, (chunk_text, embedding) in enumerate(
                    zip(chunks, embeddings)
                ):
                    self._conn.execute(
                        "INSERT INTO chunks "
                        "(file_path, chunk_index, content, file_hash, embedding) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            rel_path,
                            i,
                            chunk_text,
                            file_hash,
                            _serialize_embedding(embedding),
                        ),
                    )

                self._conn.commit()
                stats["files_indexed"] += 1
                stats["chunks_indexed"] += len(chunks)

        return stats

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        top_k: int = _DEFAULT_TOP_K,
    ) -> list[dict[str, Any]]:
        """Query the index for chunks most relevant to the given text.

        Computes an embedding for the query text, then performs cosine
        similarity search across all stored vectors to find the most
        relevant chunks.

        Args:
            text: The query text.
            top_k: Maximum number of results to return.

        Returns:
            A list of result dictionaries sorted by relevance (highest
            similarity first), each containing:

            - ``file_path``: Path to the source file.
            - ``chunk_index``: Index of the chunk within the file.
            - ``content``: The chunk text.
            - ``score``: Cosine similarity score.
        """
        # Compute query embedding.
        try:
            embeddings = self.client.embed(self.embedding_model, text)
        except Exception:
            return []

        if not embeddings:
            return []

        query_vec = embeddings[0]

        # Load all stored vectors and compute similarities.
        cursor = self._conn.execute(
            "SELECT id, file_path, chunk_index, content, embedding FROM chunks"
        )

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in cursor:
            row_id, file_path, chunk_index, content, emb_data = row
            stored_vec = _deserialize_embedding(emb_data)
            score = _cosine_similarity(query_vec, stored_vec)
            scored.append((
                score,
                {
                    "file_path": file_path,
                    "chunk_index": chunk_index,
                    "content": content,
                    "score": score,
                },
            ))

        # Sort by score descending, take top-k.
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    # ------------------------------------------------------------------
    # Prompt augmentation
    # ------------------------------------------------------------------

    def augment_prompt(
        self,
        user_text: str,
        top_k: int = _DEFAULT_TOP_K,
    ) -> str:
        """Augment a user prompt with relevant context from the index.

        Queries the index for relevant chunks and prepends them to the
        user's prompt as context.

        Args:
            user_text: The original user prompt.
            top_k: Number of context chunks to include.

        Returns:
            The augmented prompt string.  If no relevant context is found,
            returns the original prompt unchanged.
        """
        results = self.query(user_text, top_k=top_k)
        if not results:
            return user_text

        context_parts: list[str] = []
        for r in results:
            header = f"[{r['file_path']} (chunk {r['chunk_index']}, score: {r['score']:.3f})]"
            context_parts.append(f"{header}\n{r['content']}")

        context_block = "\n\n---\n\n".join(context_parts)

        return (
            "Here is relevant context from the codebase:\n\n"
            f"{context_block}\n\n"
            "---\n\n"
            f"User question: {user_text}"
        )
