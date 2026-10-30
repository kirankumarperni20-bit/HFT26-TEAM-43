"""Repository layer for document chunk persistence and retrieval."""

from collections.abc import Mapping, Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.config import EMBEDDING_DIMENSION
from backend.database.models import ChunkRecord
from backend.schemas.chunk import DocumentChunk


class ChunkRepositoryError(Exception):
    """Base exception for chunk repository operations."""

    pass


class DuplicateChunkError(ChunkRepositoryError):
    """Raised when an attempt is made to insert a chunk with a duplicate chunk_id."""

    pass


class DatabaseIntegrityError(ChunkRepositoryError):
    """Raised when a non-duplicate database integrity constraint is violated."""

    pass


def is_duplicate_chunk_id_violation(exc: IntegrityError) -> bool:
    """Determine whether an IntegrityError specifically represents a duplicate chunk_id violation.

    Uses structured database diagnostics rather than brittle error-message substrings:
    1. For PostgreSQL/psycopg:
       - Requires SQLSTATE '23505' (UNIQUE_VIOLATION).
       - Recognizes a duplicate chunk ID only when structured diagnostics identify
         the 'chunks_pkey' constraint, or reliably identify table 'chunks' and column 'chunk_id'.
       - Does NOT classify based on str(orig) or message_primary substring matching.
       - If diagnostic information is insufficient or missing, returns False.
    2. For SQLite:
       - Inspects sqlite_errorname and sqlite_errorcode from Python's sqlite3 driver.
       - Under the chunks table schema (where chunk_id is the primary key),
         a SQLITE_CONSTRAINT_PRIMARYKEY violation is classified as a duplicate chunk ID.
       - Does NOT classify SQLITE_CONSTRAINT_UNIQUE violations as duplicate chunk IDs
         using error-message substring matching. Unknown unique violations return False.
       - Does NOT classify SQLITE_CONSTRAINT_NOTNULL, SQLITE_CONSTRAINT_CHECK, or
         SQLITE_CONSTRAINT_FOREIGNKEY as duplicate chunk IDs.
    3. Fallback:
       - Returns False if the error cannot confidently be verified as a duplicate chunk ID.
    """
    orig = getattr(exc, "orig", None)
    if orig is None:
        return False

    # 1. PostgreSQL / psycopg structured diagnostics
    sqlstate = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
    diag = getattr(orig, "diag", None)
    if diag is not None:
        sqlstate = getattr(diag, "sqlstate", None) or sqlstate

    if sqlstate is not None:
        if str(sqlstate) == "23505":
            if diag is None:
                return False
            constraint = getattr(diag, "constraint_name", None) or ""
            table = getattr(diag, "table_name", None) or ""
            column = getattr(diag, "column_name", None) or ""

            c_lower = constraint.lower()
            t_lower = table.lower()
            col_lower = column.lower()

            if c_lower in ("chunks_pkey", "pk_chunks", "chunks_pk"):
                return True
            if t_lower == "chunks" and col_lower == "chunk_id":
                return True
            # Insufficient or non-matching structured constraint diagnostics
            return False
        # Any other SQLSTATE (e.g. 23502 NOT NULL, 23503 FOREIGN KEY, 23514 CHECK) is NOT a duplicate
        return False

    # 2. SQLite structured error codes
    sqlite_errorname = getattr(orig, "sqlite_errorname", None)
    sqlite_errorcode = getattr(orig, "sqlite_errorcode", None)

    try:
        import sqlite3

        sqlite_pk_code = getattr(sqlite3, "SQLITE_CONSTRAINT_PRIMARYKEY", 1555)
        sqlite_unique_code = getattr(sqlite3, "SQLITE_CONSTRAINT_UNIQUE", 2067)
    except ImportError:
        sqlite_pk_code = 1555
        sqlite_unique_code = 2067

    is_sqlite_pk = (
        sqlite_errorname == "SQLITE_CONSTRAINT_PRIMARYKEY"
        or sqlite_errorcode == sqlite_pk_code
    )
    is_sqlite_unique = (
        sqlite_errorname == "SQLITE_CONSTRAINT_UNIQUE"
        or sqlite_errorcode == sqlite_unique_code
    )

    if is_sqlite_pk:
        return True

    # Under SQLite, SQLITE_CONSTRAINT_UNIQUE does not reliably provide structured
    # column diagnostics in driver exceptions. Do not classify based on error message substrings;
    # treat unknown unique violations as non-duplicate integrity errors.
    if is_sqlite_unique:
        constraint = (getattr(orig, "constraint_name", None) or "").lower()
        column = (getattr(orig, "column_name", None) or "").lower()
        if constraint in ("chunks_pkey", "pk_chunks", "chunks_pk") or column == "chunk_id":
            return True
        return False

    return False


class ChunkRepository:
    """Repository handling all chunk persistence and query operations.

    Accepts a SQLAlchemy session and operates strictly on DocumentChunk Pydantic objects.
    Independent of FastAPI and LLM components.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def insert_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        """Insert a collection of DocumentChunk objects into the active session.

        Validates against intra-batch and pre-existing duplicate chunk IDs, adds records
        to the session, and flushes them to detect any database-level constraints.
        Does NOT commit the transaction; transaction lifecycle (commit/rollback) is owned
        by the caller's session context (e.g. get_db_session()).

        Args:
            chunks: Sequence of DocumentChunk Pydantic objects to persist.

        Returns:
            The number of successfully added and flushed chunks (0 if chunks is empty).

        Raises:
            DuplicateChunkError: If any chunk_id already exists in the database or
                appears more than once in the input sequence.
        """
        if not chunks:
            return 0

        # Check for intra-batch duplicate chunk IDs
        incoming_ids = [c.chunk_id for c in chunks]
        if len(incoming_ids) != len(set(incoming_ids)):
            seen = set()
            duplicates = {x for x in incoming_ids if x in seen or seen.add(x)}
            raise DuplicateChunkError(
                f"Duplicate chunk IDs found within the input batch: {list(duplicates)}"
            )

        # Check for pre-existing chunk IDs in the database
        stmt = select(ChunkRecord.chunk_id).where(ChunkRecord.chunk_id.in_(incoming_ids))
        existing_ids = self.session.scalars(stmt).all()
        if existing_ids:
            raise DuplicateChunkError(
                f"Cannot insert duplicate chunks. Existing chunk IDs: {list(existing_ids)}"
            )

        records = [
            ChunkRecord(
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                page_number=chunk.page_number,
                paragraph_number=chunk.paragraph_number,
                text=chunk.text,
                section=chunk.section,
            )
            for chunk in chunks
        ]

        try:
            self.session.add_all(records)
            self.session.flush()
        except IntegrityError as exc:
            if is_duplicate_chunk_id_violation(exc):
                raise DuplicateChunkError(
                    f"Duplicate chunk ID constraint violated during insertion: {exc.orig}"
                ) from exc
            raise DatabaseIntegrityError(
                f"Database integrity constraint violated during chunk insertion: {exc.orig}"
            ) from exc

        return len(records)

    def get_chunks_by_paper_ids(self, paper_ids: Sequence[str]) -> list[DocumentChunk]:
        """Retrieve chunks belonging to the specified paper IDs.

        Args:
            paper_ids: Sequence of paper identifiers to filter chunks by.

        Returns:
            A list of DocumentChunk Pydantic models with stable ordering by
            paper_id, page_number, paragraph_number, and chunk_id.
            Returns an empty list immediately if paper_ids is empty or no matches exist.
        """
        if not paper_ids:
            return []

        # Deduplicate paper_ids preserving list structure
        unique_paper_ids = list(dict.fromkeys(paper_ids))

        stmt = (
            select(ChunkRecord)
            .where(ChunkRecord.paper_id.in_(unique_paper_ids))
            .order_by(
                ChunkRecord.paper_id.asc(),
                ChunkRecord.page_number.asc(),
                ChunkRecord.paragraph_number.asc(),
                ChunkRecord.chunk_id.asc(),
            )
        )

        records = self.session.scalars(stmt).all()

        return [record.to_pydantic() for record in records]

    def update_embeddings(
        self,
        embeddings_by_chunk_id: Mapping[str, Sequence[float] | np.ndarray],
    ) -> int:
        """Update dense embedding vectors for existing document chunks.

        Validates vector dimension, non-emptiness, finiteness, and chunk existence.
        Flushes changes to the database without committing, leaving transaction ownership
        with the caller context.

        Args:
            embeddings_by_chunk_id: Mapping from chunk_id to 384-dimensional vector.

        Returns:
            The number of records updated (0 if empty mapping).

        Raises:
            ValueError: If vector has incorrect dimension, contains NaN/Inf values,
                or chunk_id is invalid.
            KeyError: If any chunk_id in the mapping does not exist in the database.
        """
        if not embeddings_by_chunk_id:
            return 0

        # Validate vectors: dimension, finite numbers, format
        validated_vectors: dict[str, list[float]] = {}
        for chunk_id, vec in embeddings_by_chunk_id.items():
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("chunk_id keys must be non-empty strings.")

            arr = np.asarray(vec, dtype=np.float32)
            if arr.shape != (EMBEDDING_DIMENSION,):
                raise ValueError(
                    f"Vector for chunk '{chunk_id}' has invalid dimension {arr.shape}; "
                    f"expected ({EMBEDDING_DIMENSION},)."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"Vector for chunk '{chunk_id}' contains NaN or infinite values."
                )
            validated_vectors[chunk_id] = arr.tolist()

        # Check that all chunk IDs exist in the database
        target_ids = list(validated_vectors.keys())
        stmt = select(ChunkRecord).where(ChunkRecord.chunk_id.in_(target_ids))
        existing_records = {record.chunk_id: record for record in self.session.scalars(stmt).all()}

        missing_ids = set(target_ids) - set(existing_records.keys())
        if missing_ids:
            raise KeyError(
                f"Cannot update embeddings for non-existent chunk IDs: {list(missing_ids)}"
            )

        # Update embedding column for each record
        for chunk_id, vec_list in validated_vectors.items():
            record = existing_records[chunk_id]
            record.embedding = vec_list

        self.session.flush()
        return len(validated_vectors)
