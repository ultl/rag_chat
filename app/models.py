import typing
from datetime import datetime
from uuid import uuid4

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def _uuid() -> str:
  return str(uuid4())


class ChatSession(SQLModel, table=True):  # type: ignore[misc]
  """A chat session grouping multiple messages."""

  id: str = Field(default_factory=_uuid, primary_key=True)
  title: str | None = None
  created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
  updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class ChatMessage(SQLModel, table=True):  # type: ignore[misc]
  """Messages exchanged in a session."""

  id: str = Field(default_factory=_uuid, primary_key=True)
  session_id: str = Field(index=True, foreign_key='chatsession.id')
  role: str
  content: str
  created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
  extras: dict[str, typing.Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
