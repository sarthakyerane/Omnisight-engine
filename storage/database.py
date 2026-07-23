"""
Database engine and session factory for RIOM.

Creates the SQLAlchemy engine from DATABASE_URL in config.
SQLite databases automatically have WAL mode enabled for safe multi-thread writes.
"""
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

from config import settings


def _apply_sqlite_pragmas(dbapi_conn, _connection_record):
    """Enable WAL mode and foreign key enforcement on every new SQLite connection."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    # For SQLite in multi-thread mode, StaticPool prevents "database is locked" errors.
    # For Postgres/MySQL, remove pool_pre_ping and use the default QueuePool.
    pool_pre_ping=True,
)

if "sqlite" in settings.DATABASE_URL:
    event.listen(engine, "connect", _apply_sqlite_pragmas)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
