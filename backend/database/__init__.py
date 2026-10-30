"""Database module providing ORM models, connection management, and repository layers."""

from backend.database.connection import (
    DatabaseConfigurationError,
    create_db_engine,
    get_database_url,
    get_db_session,
    get_session_factory,
)
from backend.database.models import Base, ChunkRecord
from backend.database.repository import (
    ChunkRepository,
    ChunkRepositoryError,
    DatabaseIntegrityError,
    DuplicateChunkError,
    is_duplicate_chunk_id_violation,
)

__all__ = [
    "Base",
    "ChunkRecord",
    "ChunkRepository",
    "ChunkRepositoryError",
    "DuplicateChunkError",
    "DatabaseIntegrityError",
    "is_duplicate_chunk_id_violation",
    "DatabaseConfigurationError",
    "create_db_engine",
    "get_database_url",
    "get_session_factory",
    "get_db_session",
]
