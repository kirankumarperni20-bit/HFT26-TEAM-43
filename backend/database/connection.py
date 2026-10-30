"""Database connection and session-factory utilities."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


class DatabaseConfigurationError(Exception):
    """Raised when database configuration is missing, invalid, or improperly specified."""

    pass


def get_database_url(explicit_url: str | None = None) -> str:
    """Retrieve the database URL from an explicit argument or the DATABASE_URL environment variable.

    Raises:
        DatabaseConfigurationError: If no database URL is supplied or set in the environment.
    """
    url = explicit_url or os.environ.get("DATABASE_URL")
    if not url or not url.strip():
        raise DatabaseConfigurationError(
            "Production database URL is not configured. "
            "Please supply an explicit database_url or set the DATABASE_URL environment variable."
        )
    return url.strip()


def create_db_engine(database_url: str | None = None, **engine_kwargs: Any) -> Engine:
    """Create a SQLAlchemy Engine for the application.

    Supports:
        - PostgreSQL using the psycopg driver (e.g., 'postgresql+psycopg://...')
        - SQLite only when explicitly provided (e.g., 'sqlite:///:memory:' or 'sqlite:///test.db')

    Critical design rule:
        Never silently fall back to SQLite when a PostgreSQL URL is configured.
        Importing this module does not trigger connection initialization.
    """
    resolved_url = get_database_url(database_url)

    # SQLite-specific default kwargs (disable same-thread check for in-memory / testing)
    if resolved_url.startswith("sqlite"):
        if "connect_args" not in engine_kwargs:
            engine_kwargs["connect_args"] = {"check_same_thread": False}
    else:
        if "pool_pre_ping" not in engine_kwargs:
            engine_kwargs["pool_pre_ping"] = True

    return create_engine(resolved_url, **engine_kwargs)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a SQLAlchemy session factory bound to the provided engine."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def get_db_session(session_factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    """Context manager yielding a transactional database session.

    Commits on successful completion, rolls back on exception, and always closes the session.
    """
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
