"""Database connection and session-factory utilities."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


class DatabaseConfigurationError(Exception):
    """Raised when database configuration is missing, invalid, or improperly specified."""

    pass


_ENV_LOADED = False


def load_environment(force_reload: bool = False, env_path: Path | str | None = None) -> bool:
    """Reliably load environment variables from backend/.env or root .env into os.environ.

    Checks:
    1. Explicit env_path if provided.
    2. backend/.env (directory of backend package).
    3. Root .env (repository root).

    Does not overwrite existing environment variables (override=False) to preserve
    process-level environment configuration.

    Args:
        force_reload: If True, reload the .env file even if already loaded once.
        env_path: Optional explicit path to a .env file.

    Returns:
        True if any .env file was successfully loaded, False otherwise.
    """
    global _ENV_LOADED
    if _ENV_LOADED and not force_reload and env_path is None:
        return False

    if env_path is not None:
        target = Path(env_path)
        if target.is_file():
            loaded = load_dotenv(dotenv_path=target, override=False)
            _ENV_LOADED = True
            return loaded
        return False

    backend_dir = Path(__file__).resolve().parent.parent
    backend_env = backend_dir / ".env"
    root_env = backend_dir.parent / ".env"

    loaded = False
    if backend_env.is_file():
        loaded = load_dotenv(dotenv_path=backend_env, override=False) or loaded
    if root_env.is_file():
        loaded = load_dotenv(dotenv_path=root_env, override=False) or loaded

    _ENV_LOADED = True
    return loaded


# Automatically load environment upon module import
load_environment()


def is_database_configured() -> bool:
    """Check if DATABASE_URL is set in the environment or .env without raising an exception."""
    url = os.environ.get("DATABASE_URL")
    if not url or not url.strip():
        load_environment()
        url = os.environ.get("DATABASE_URL")
    return bool(url and url.strip())


def get_database_url(explicit_url: str | None = None) -> str:
    """Retrieve the database URL from an explicit argument or the DATABASE_URL environment variable.

    Raises:
        DatabaseConfigurationError: If no database URL is supplied or set in the environment.
    """
    if explicit_url and explicit_url.strip():
        return explicit_url.strip()

    url = os.environ.get("DATABASE_URL")
    if not url or not url.strip():
        load_environment()
        url = os.environ.get("DATABASE_URL")

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
        if "connect_args" not in engine_kwargs:
            engine_kwargs["connect_args"] = {"connect_timeout": 2}

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


_CACHED_ENGINE: Engine | None = None


def reset_cached_engine() -> None:
    """Reset the cached engine (used in tests or during reconfiguration)."""
    global _CACHED_ENGINE
    if _CACHED_ENGINE is not None:
        try:
            _CACHED_ENGINE.dispose()
        except Exception:
            pass
        _CACHED_ENGINE = None


def get_db_session_dep() -> Generator[Session | None, None, None]:
    """FastAPI dependency yielding a managed database session per request if DATABASE_URL is configured.

    Yields None if DATABASE_URL is absent or unconfigured, permitting graceful degradation
    to the local test corpus without crashing server startup.
    Ensures that a single session is reused across all claim evaluations within a request,
    and cleanly closes connections in the finally block.
    """
    global _CACHED_ENGINE
    load_environment()
    db_url = os.environ.get("DATABASE_URL")
    if not db_url or not db_url.strip():
        logger.warning(
            "DATABASE_URL is not configured in environment or .env. "
            "Database retrieval is inactive; operating in local PDF fallback mode."
        )
        yield None
        return

    session: Session | None = None
    try:
        if _CACHED_ENGINE is None:
            _CACHED_ENGINE = create_db_engine(db_url)
        # Probe connection to verify the database is actually reachable
        with _CACHED_ENGINE.connect() as probe_conn:
            probe_conn.execute(text("SELECT 1"))
        session_factory = get_session_factory(_CACHED_ENGINE)
        session = session_factory()
    except Exception as exc:
        logger.error(
            "Failed to initialize database session from configured DATABASE_URL: %s. "
            "Operating in local PDF fallback mode.",
            type(exc).__name__,
        )
        yield None
        return

    try:
        yield session
    finally:
        if session is not None:
            session.close()
