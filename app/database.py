from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import inspect
from sqlmodel import Session, SQLModel, create_engine

from .config import get_settings
from .models import ChatMessage, ChatSession

settings = get_settings()
engine = create_engine(settings.postgres_url, echo=False, pool_pre_ping=True)


def init_db() -> None:
  """Create database tables if they do not exist."""
  inspector = inspect(engine)
  tables = inspector.get_table_names()
  if 'chatsession' in tables:
    columns = {col['name'] for col in inspector.get_columns('chatsession')}
    if 'updated_at' not in columns:
      ChatMessage.__table__.drop(engine, checkfirst=True)  # type: ignore[attr-defined]
      ChatSession.__table__.drop(engine, checkfirst=True)  # type: ignore[attr-defined]
  SQLModel.metadata.create_all(engine)


@contextmanager
def session_scope() -> Iterator[Session]:
  """Provide a transactional scope around a series of operations."""
  with Session(engine) as session:
    yield session


def get_session() -> Iterator[Session]:
  with Session(engine) as session:
    yield session
