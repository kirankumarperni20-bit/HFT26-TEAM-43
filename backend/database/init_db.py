"""
init_db.py — Safe database and schema initialization for the AI Research Assistant.

Responsibilities:
1. Verify database connectivity before applying schema changes.
2. Enable the PostgreSQL 'vector' extension (pgvector) when using PostgreSQL.
3. Create all tables and indexes defined in SQLAlchemy declarative models (ChunkRecord).
4. Strictly safe: Never drops tables or deletes existing data (checkfirst=True).
5. Verify schema readiness and report status clearly without exposing credentials.

Usage:
    Command line:
        PYTHONPATH=. python3 -m backend.database.init_db
        # Or directly:
        python3 backend/database/init_db.py

    Programmatic:
        from backend.database.init_db import init_database
        success = init_database()
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path if invoked directly
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import OperationalError, ProgrammingError

try:
    from backend.database.connection import (
        DatabaseConfigurationError,
        create_db_engine,
        get_database_url,
        is_database_configured,
        load_environment,
    )
    from backend.database.models import Base
except ModuleNotFoundError:
    from database.connection import (
        DatabaseConfigurationError,
        create_db_engine,
        get_database_url,
        is_database_configured,
        load_environment,
    )
    from database.models import Base

logger = logging.getLogger("backend.database.init_db")


def sanitize_url(engine: Engine) -> str:
    """Return a sanitized connection string with password hidden."""
    try:
        return engine.url.render_as_string(hide_password=True)
    except Exception:
        return "<sanitized database connection>"


def sanitize_error_message(message: str) -> str:
    """Mask potential passwords or sensitive tokens from raw exception text."""
    # Redact password patterns like :password@ in URLs
    sanitized = re.sub(r":([^:@\s]+)@", ":***@", message)
    # Redact password=value parameters
    sanitized = re.sub(r"password=['\"][^'\"]*['\"]", "password='***'", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"password=\S+", "password=***", sanitized, flags=re.IGNORECASE)
    return sanitized


def init_database(
    database_url: str | None = None,
    engine: Engine | None = None,
) -> bool:
    """Initialize the database schema safely.

    Steps:
    1. Resolve engine using provided engine, database_url, or DATABASE_URL from .env/environment.
    2. Confirm database connection (SELECT 1).
    3. If running on PostgreSQL, enable the pgvector extension (CREATE EXTENSION IF NOT EXISTS vector).
    4. Create all model tables and indexes safely (never drops existing data).
    5. Inspect and verify that expected tables exist.

    Args:
        database_url: Optional explicit database URL (e.g. 'sqlite:///:memory:' or PostgreSQL URL).
        engine: Optional pre-constructed SQLAlchemy Engine (useful for test fixtures).

    Returns:
        True if initialization and verification succeeded, False otherwise.
    """
    load_environment()

    # Step 1: Resolve engine
    active_engine = engine
    if active_engine is None:
        try:
            active_engine = create_db_engine(database_url=database_url)
        except DatabaseConfigurationError as exc:
            logger.error(
                "Database initialization aborted: %s. "
                "Please configure DATABASE_URL in backend/.env or your environment.",
                exc,
            )
            return False
        except Exception as exc:
            logger.error(
                "Failed to construct database engine: %s",
                sanitize_error_message(str(exc)),
            )
            return False

    should_dispose = engine is None
    try:
        safe_target = sanitize_url(active_engine)
        dialect_name = active_engine.dialect.name

        # Step 2: Confirm connectivity
        logger.info("Connecting to database at '%s' (dialect: %s)...", safe_target, dialect_name)
        try:
            with active_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as exc:
            logger.error(
                "Could not establish database connection to '%s': %s (%s). "
                "Please ensure the database service is running, accessible, and credentials are correct.",
                safe_target,
                type(exc).__name__,
                sanitize_error_message(str(exc)),
            )
            return False

        # Step 3: Enable pgvector extension on PostgreSQL
        if dialect_name == "postgresql":
            logger.info("PostgreSQL dialect detected. Ensuring 'vector' extension is active...")
            try:
                with active_engine.begin() as conn:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                logger.info("PostgreSQL pgvector extension verified/enabled.")
            except Exception as exc:
                logger.error(
                    "Failed to enable 'vector' extension on PostgreSQL: %s (%s). "
                    "Please verify pgvector is installed on the PostgreSQL instance and that "
                    "the database user has CREATE EXTENSION privileges.",
                    type(exc).__name__,
                    sanitize_error_message(str(exc)),
                )
                return False
        else:
            logger.info("Dialect is '%s'; skipping pgvector extension registration.", dialect_name)

        # Step 4: Create model tables safely (checkfirst=True guarantees no data loss)
        logger.info("Creating tables and indexes from declarative models (safe checkfirst=True)...")
        try:
            Base.metadata.create_all(bind=active_engine, checkfirst=True)
        except Exception as exc:
            logger.error(
                "Failed to create schema tables: %s (%s)",
                type(exc).__name__,
                sanitize_error_message(str(exc)),
            )
            return False

        # Step 5: Verification
        try:
            inspector = inspect(active_engine)
            actual_tables = set(inspector.get_table_names())
            expected_tables = set(Base.metadata.tables.keys())

            missing = expected_tables - actual_tables
            if missing:
                logger.error("Database initialization verification failed: missing tables %s", missing)
                return False

            logger.info(
                "Database initialization completed successfully! Active tables: %s",
                sorted(list(actual_tables)),
            )
            return True
        except Exception as exc:
            logger.error(
                "Schema verification inspection failed: %s (%s)",
                type(exc).__name__,
                sanitize_error_message(str(exc)),
            )
            return False
    finally:
        if should_dispose and active_engine is not None:
            active_engine.dispose()



def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for database initialization."""
    parser = argparse.ArgumentParser(
        description="Initialize the AI Research Assistant database and pgvector schema safely."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional explicit database URL (overrides DATABASE_URL in .env).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging output.",
    )
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("===================================================================")
    print("AI Research Assistant — Database & Schema Initialization")
    print("===================================================================")

    success = init_database(database_url=args.url)
    if success:
        print("[SUCCESS] Database initialization completed successfully.")
        return 0
    else:
        print(
            "[ERROR] Database initialization failed. Check log messages above.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
