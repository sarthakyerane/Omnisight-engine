"""
Shared pytest fixtures for RIOM tests.

Provides:
  - test_db: an in-memory SQLite Session with all tables created
  - override_db: FastAPI dependency override for get_db
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from storage.models import Base


@pytest.fixture()
def test_db():
    """
    Yield a SQLAlchemy Session backed by a fresh in-memory SQLite DB.
    Tables are created before the test and the DB is discarded afterwards.
    """
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()
