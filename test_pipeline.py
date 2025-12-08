from __future__ import annotations

import asyncio
import mimetypes
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from environs import Env

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from app.agent import AgentDeps, agent, build_history
from app.config import get_settings
from app.database import init_db, session_scope
from app.deps import get_embedder, get_minio_client, get_redis_client, get_vector_store
from app.ingest import IngestionPipeline
from app.models import ChatMessage, ChatSession
from app.retrieval import Retriever

if TYPE_CHECKING:
  from pydantic_ai.agent import AgentRunResult
  from sqlmodel import Session

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

env = Env()
env.read_env(ROOT / '.env')
settings = get_settings()


def ensure_documents(files: list[Path]) -> None:
  pipeline = IngestionPipeline(
    minio=get_minio_client(),
    vector_store=get_vector_store(),
    embedder=get_embedder(),
  )
  existing = {doc['filename'] for doc in get_vector_store().list_documents()}
  for path in files:
    if path.name in existing:
      print(f'Skipping {path.name}, already ingested')
      continue
    with path.open('rb') as f:
      content = f.read()
    ctype = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
    doc = pipeline.ingest_upload(content, filename=path.name, content_type=ctype)
    print(f'Ingested {path.name} -> {doc.id}')


def run_retrieval(session: Session, queries: list[str]) -> None:
  retriever = Retriever(
    redis_client=get_redis_client(),
    session=session,
    vector_store=get_vector_store(),
    embedder=get_embedder(),
  )
  for query in queries:
    result = retriever.retrieve(query)
    print(f'\nQuery: {query}')
    print(f'Doc IDs: {result.document_ids}')
    for chunk in result.chunks[:2]:
      print(f'- [{chunk.document_id}] {chunk.text[:120].replace("\\n", " ")} (score {chunk.score:.4f})')


def run_agent_sample(session: Session, question: str) -> None:
  chat_session = ChatSession()
  session.add(chat_session)
  session.commit()
  session.refresh(chat_session)

  history = build_history([])
  retriever = Retriever(
    redis_client=get_redis_client(),
    session=session,
    vector_store=get_vector_store(),
    embedder=get_embedder(),
  )

  async def _run() -> AgentRunResult[object]:
    return await agent.run(question, deps=AgentDeps(retriever=retriever), message_history=history)

  result: AgentRunResult[object] = asyncio.run(_run())
  session.add(ChatMessage(session_id=chat_session.id, role='user', content=question))
  session.add(ChatMessage(session_id=chat_session.id, role='assistant', content=str(result.output)))
  session.commit()
  print(f'\nAgent reply: {result.output}')


def main() -> None:
  init_db()
  files = [ROOT / 'data' / name for name in ('1.pdf', '2.pdf', '1.docx')]
  queries_file = ROOT / 'data' / 'queries.md'
  sample_queries = []
  for line in queries_file.read_text().splitlines():
    if not line or line.startswith('#'):
      continue
    sample_queries.append(line.lstrip('0123456789. ').strip())

  with session_scope() as session:
    ensure_documents(files=files)

  with session_scope() as session:
    run_retrieval(session, sample_queries[:4])
    run_agent_sample(session, sample_queries[0])


if __name__ == '__main__':
  main()
