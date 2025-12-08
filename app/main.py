import asyncio
import contextlib
import json
import logging
import typing
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_ai import _agent_graph, messages
from pydantic_graph.nodes import End
from sqlalchemy import asc, desc
from sqlmodel import Session, delete, select

from app.agent import AgentDeps, agent, build_history
from app.config import get_settings
from app.database import get_session, init_db
from app.deps import get_embedder, get_minio_client, get_redis_client, get_vector_store
from app.ingest import IngestionPipeline
from app.models import ChatMessage, ChatSession
from app.retrieval import Retriever

settings = get_settings()
app = FastAPI(title='RAG Chatbot')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')

app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)


@app.on_event('startup')
def on_startup() -> None:
  init_db()


class UploadResponse(BaseModel):
  id: str
  filename: str
  object_name: str
  size: int


class DocumentOut(BaseModel):
  id: str
  filename: str
  size: int
  created_at: datetime
  url: str


@app.post('/api/upload')
async def upload_document(
  file: Annotated[UploadFile, File()],
) -> UploadResponse:
  content = await file.read()
  filename = file.filename or 'uploaded_file'
  try:
    pipeline = IngestionPipeline(
      minio=get_minio_client(),
      vector_store=get_vector_store(),
      embedder=get_embedder(),
    )
    document = pipeline.ingest_upload(
      content, filename=filename, content_type=file.content_type or 'application/octet-stream'
    )
    return UploadResponse(
      id=document.id,
      filename=document.filename,
      object_name=document.object_name,
      size=document.size,
    )
  except Exception as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get('/api/documents')
def list_documents() -> list[DocumentOut]:
  docs = get_vector_store().list_documents()
  minio = get_minio_client()
  results: list[DocumentOut] = []
  for doc in docs:
    try:
      url = minio.presigned_url(doc['object_name'], expires=3600)
    except Exception:
      url = ''
    results.append(
      DocumentOut(
        id=doc['id'],
        filename=doc['filename'],
        size=doc['size'],
        created_at=doc['created_at'],
        url=url,
      )
    )
  return results


@app.delete('/api/documents/{document_id}')
def delete_document(document_id: str) -> dict[str, str]:
  docs = get_vector_store().list_documents()
  match = next((d for d in docs if d['id'] == document_id), None)
  if not match:
    raise HTTPException(status_code=404, detail='Document not found')

  get_vector_store().delete_document(document_id)

  with contextlib.suppress(Exception):
    get_minio_client().remove(match['object_name'])

  return {'status': 'deleted'}


class SessionOut(BaseModel):
  id: str
  title: str | None
  created_at: datetime
  updated_at: datetime


class MessageOut(BaseModel):
  id: str
  role: str
  content: str
  created_at: datetime
  documents: list[dict[str, str]] = []
  tools: list[str] = []
  support: bool | None = None
  # Retrieved chunk debugging info
  chunks: list[dict[str, str]] = []
  tool_logs: list[dict[str, typing.Any]] = []


@app.get('/api/sessions', response_model=list[SessionOut])
def list_sessions(session: Annotated[Session, Depends(get_session)]) -> list[ChatSession]:
  statement = select(ChatSession).order_by(desc(ChatSession.updated_at))  # type: ignore[arg-type]
  return list(session.exec(statement))


class CreateSessionRequest(BaseModel):
  title: str | None = None


@app.post('/api/sessions', response_model=SessionOut)
def create_session(payload: CreateSessionRequest, session: Annotated[Session, Depends(get_session)]) -> ChatSession:
  chat_session = ChatSession(title=payload.title)
  session.add(chat_session)
  session.commit()
  session.refresh(chat_session)
  return chat_session


class RenameSessionRequest(BaseModel):
  title: str


@app.post('/api/sessions/{session_id}/rename', response_model=SessionOut)
def rename_session(
  session_id: str, payload: RenameSessionRequest, session: Annotated[Session, Depends(get_session)]
) -> ChatSession:
  chat_session = session.get(ChatSession, session_id)
  if not chat_session:
    raise HTTPException(status_code=404, detail='Session not found')
  chat_session.title = payload.title.strip() or chat_session.title
  chat_session.updated_at = datetime.utcnow()
  session.add(chat_session)
  session.commit()
  session.refresh(chat_session)
  return chat_session


@app.delete('/api/sessions/{session_id}')
def delete_session(session_id: str, session: Annotated[Session, Depends(get_session)]) -> dict[str, str]:
  chat_session = session.get(ChatSession, session_id)
  if not chat_session:
    raise HTTPException(status_code=404, detail='Session not found')
  session.exec(delete(ChatMessage).where(ChatMessage.session_id == session_id))  # type: ignore[arg-type]
  session.delete(chat_session)
  session.commit()
  return {'status': 'deleted'}


@app.get('/api/sessions/{session_id}/messages')
def get_session_messages(session_id: str, session: Annotated[Session, Depends(get_session)]) -> list[MessageOut]:
  _ensure_session(session, session_id, create_if_missing=False)
  messages_out: list[MessageOut] = []
  for msg in _get_messages(session, session_id):
    meta = msg.extras or {}
    documents = meta.get('documents', []) if isinstance(meta, dict) else []
    tools = meta.get('tools', []) if isinstance(meta, dict) else []
    support = meta.get('support') if isinstance(meta, dict) else None
    chunks = meta.get('chunks', []) if isinstance(meta, dict) else []
    tool_logs = meta.get('tool_logs', []) if isinstance(meta, dict) else []
    messages_out.append(
      MessageOut(
        id=msg.id,
        role=msg.role,
        content=msg.content,
        created_at=msg.created_at,
        documents=documents,
        tools=tools,
        support=support,
        chunks=chunks,
        tool_logs=tool_logs,
      )
    )
  return messages_out


class ChatRequest(BaseModel):
  session_id: str | None = None
  message: str


def _ensure_session(
  session: Session, session_id: str | None, title_hint: str | None = None, create_if_missing: bool = True
) -> ChatSession:
  if session_id:
    existing = session.get(ChatSession, session_id)
    if existing:
      return existing
    if not create_if_missing:
      raise HTTPException(status_code=404, detail='Session not found')
  chat_session = ChatSession(title=(title_hint or '').strip() or None)
  session.add(chat_session)
  session.commit()
  session.refresh(chat_session)
  return chat_session


def _get_messages(session: Session, session_id: str) -> list[ChatMessage]:
  statement = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(asc(ChatMessage.created_at))  # type: ignore[arg-type]
  return list(session.exec(statement))


def _extract_doc_ids(content: typing.Any) -> list[str]:
  if isinstance(content, dict):
    ids = list(content.get('document_ids') or [])
    for item in content.get('chunks') or []:
      doc_id = item.get('document_id')
      if doc_id:
        ids.append(doc_id)
    return ids
  if hasattr(content, 'document_ids'):
    try:
      return list(content.document_ids)
    except Exception:
      return []
  return []


def _extract_chunk_texts(content: typing.Any) -> list[str]:
  texts: list[str] = []
  if isinstance(content, dict):
    for item in content.get('chunks') or []:
      text = item.get('text')
      if text:
        texts.append(str(text))
  elif hasattr(content, 'chunks'):
    try:
      for item in content.chunks:
        text = getattr(item, 'text', None)
        if text:
          texts.append(str(text))
    except Exception:
      return texts
  return texts


def _extract_chunk_infos(content: typing.Any) -> list[dict[str, str]]:
  chunks: list[dict[str, str]] = []
  if isinstance(content, dict):
    for item in content.get('chunks') or []:
      chunk_id = item.get('chunk_id')
      text = item.get('text')
      doc_id = item.get('document_id')
      if chunk_id or text:
        chunks.append({'chunk_id': str(chunk_id or ''), 'text': str(text or ''), 'document_id': str(doc_id or '')})
    return chunks
  if hasattr(content, 'chunks'):
    try:
      for item in content.chunks:  # type: ignore[attr-defined]
        chunk_id = getattr(item, 'chunk_id', '') or ''
        text = getattr(item, 'text', '') or ''
        doc_id = getattr(item, 'document_id', '') or ''
        if chunk_id or text:
          chunks.append({'chunk_id': str(chunk_id), 'text': str(text), 'document_id': str(doc_id)})
    except Exception:
      return chunks
  return chunks


def _dedupe_chunk_infos(chunks: list[dict[str, str]]) -> list[dict[str, str]]:
  seen: set[tuple[str, str]] = set()
  unique: list[dict[str, str]] = []
  for item in chunks:
    cid = item.get('chunk_id', '')
    text = item.get('text', '')
    key = (cid, text)
    if key in seen:
      continue
    seen.add(key)
    unique.append(item)
  return unique


def _jsonable(obj: typing.Any) -> typing.Any:
  if obj is None:
    return None
  if isinstance(obj, (str, int, float, bool)):
    return obj
  if isinstance(obj, (list, tuple)):
    return [_jsonable(x) for x in obj]
  if isinstance(obj, dict):
    return {str(k): _jsonable(v) for k, v in obj.items()}
  if hasattr(obj, 'model_dump'):
    try:
      return _jsonable(obj.model_dump())
    except Exception:
      return str(obj)
  try:
    return obj.__dict__
  except Exception:
    return str(obj)


@app.post('/api/chat/stream')
async def chat_stream(
  request: ChatRequest,
  session: Annotated[Session, Depends(get_session)],
) -> StreamingResponse:
  logger.info('chat_stream start session_id=%s message=%s', getattr(request, 'session_id', None), request.message)
  chat_session = _ensure_session(session, request.session_id, title_hint=request.message[:80])
  previous_messages = _get_messages(session, chat_session.id)
  history = build_history(previous_messages)

  now = datetime.utcnow()
  user_message = ChatMessage(session_id=chat_session.id, role='user', content=request.message, created_at=now)
  session.add(user_message)
  if not chat_session.title:
    chat_session.title = request.message[:80] or chat_session.title
  chat_session.updated_at = now
  session.add(chat_session)
  session.commit()

  retriever = Retriever(
    redis_client=get_redis_client(),
    session=session,
    vector_store=get_vector_store(),
    embedder=get_embedder(),
  )

  async def event_generator() -> AsyncGenerator[bytes]:
    async def _stream_tokens(text: str, size: int = 60) -> AsyncGenerator[str]:
      for idx in range(0, len(text), size):
        yield text[idx : idx + size]
        # Give the event loop a chance to flush each chunk to the client.
        await asyncio.sleep(0.02)

    assistant_text = ''
    doc_ids: set[str] = set()
    support_called = False
    tools_used: set[str] = set()
    chunk_infos: list[dict[str, str]] = []
    doc_meta: list[dict[str, str]] = []
    tool_logs: list[dict[str, typing.Any]] = []
    seen_call_ids: set[str] = set()
    seen_return_ids: set[str] = set()
    last_text = ''

    # Kick off the stream so the client renders immediately.
    yield b': ping\n\n'

    try:
      async with agent.iter(
        request.message,
        deps=AgentDeps(retriever=retriever),
        message_history=history,
      ) as run:
        node = run.next_node
        while True:
          if isinstance(node, _agent_graph.ModelRequestNode):
            graph_ctx = run.ctx
            async with node._stream(graph_ctx) as stream:
              async for event in stream:
                if isinstance(event, messages.PartStartEvent):
                  if isinstance(event.part, messages.TextPart):
                    new_text = event.part.content or ''
                    delta_text = new_text.removeprefix(last_text)
                    if delta_text:
                      async for token in _stream_tokens(delta_text):
                        yield f'data: {json.dumps({"type": "delta", "token": token})}\n\n'.encode()
                    assistant_text = new_text
                    last_text = new_text
                  elif isinstance(event.part, messages.ToolCallPart):
                    call_id = getattr(event.part, 'tool_call_id', '') or f'call-{event.part.tool_name}-{len(tool_logs)}'
                    if call_id not in seen_call_ids:
                      seen_call_ids.add(call_id)
                      tools_used.add(event.part.tool_name)
                      tool_logs.append({
                        'kind': 'call',
                        'tool': event.part.tool_name,
                        'args': _jsonable(event.part.args),
                      })
                      yield f'data: {json.dumps({"type": "tool", "logs": tool_logs, "doc_ids": list(doc_ids), "chunks": _dedupe_chunk_infos(chunk_infos)})}\n\n'.encode()
                elif isinstance(event, messages.PartDeltaEvent):
                  if isinstance(event.delta, messages.TextPartDelta) and event.delta.content_delta:
                    delta_text = event.delta.content_delta
                    assistant_text += delta_text
                    last_text = assistant_text
                    async for token in _stream_tokens(delta_text):
                      yield f'data: {json.dumps({"type": "delta", "token": token})}\n\n'.encode()
            node = await run.next(node)
            continue

          if isinstance(node, _agent_graph.CallToolsNode):
            async with node.stream(run.ctx) as tool_events:
              async for event in tool_events:
                if isinstance(event, messages.FunctionToolCallEvent):
                  part = event.part
                  call_id = getattr(part, 'tool_call_id', '') or f'call-{part.tool_name}-{len(tool_logs)}'
                  if call_id not in seen_call_ids:
                    seen_call_ids.add(call_id)
                    tools_used.add(part.tool_name)
                    logger.info('tool call %s args=%s', part.tool_name, _jsonable(part.args))
                    tool_logs.append({'kind': 'call', 'tool': part.tool_name, 'args': _jsonable(part.args)})
                    payload_chunks = [
                      {'chunk_id': c.get('chunk_id'), 'document_id': c.get('document_id')}
                      for c in _dedupe_chunk_infos(chunk_infos)
                    ]
                    yield f'data: {json.dumps({"type": "tool", "logs": tool_logs, "doc_ids": list(doc_ids), "chunks": payload_chunks})}\n\n'.encode()
                elif isinstance(event, messages.FunctionToolResultEvent):
                  result = event.result
                  ret_id = (
                    getattr(result, 'tool_call_id', '')
                    or f'return-{getattr(result, "tool_name", "tool")}-{len(tool_logs)}'
                  )
                  if ret_id in seen_return_ids:
                    continue
                  seen_return_ids.add(ret_id)
                  tool_name = getattr(result, 'tool_name', '')
                  if tool_name:
                    tools_used.add(tool_name)
                  logger.info('tool return %s content=%s', tool_name, _jsonable(getattr(result, 'content', None)))
                  if isinstance(result, messages.ToolReturnPart):
                    if result.tool_name == 'retrieveDocument':
                      doc_ids.update(_extract_doc_ids(result.content))
                      chunk_infos.extend(_extract_chunk_infos(result.content))
                      summary_content: typing.Any = {'document_ids': list(doc_ids), 'chunks_count': len(chunk_infos)}
                    else:
                      summary_content = _jsonable(result.content)
                      if result.tool_name == 'transferToSupport':
                        support_called = True
                  else:
                    summary_content = _jsonable(getattr(result, 'content', None))
                  tool_logs.append({'kind': 'return', 'tool': tool_name or 'tool', 'content': summary_content})
                  payload_chunks = [
                    {'chunk_id': c.get('chunk_id'), 'document_id': c.get('document_id')}
                    for c in _dedupe_chunk_infos(chunk_infos)
                  ]
                  yield f'data: {json.dumps({"type": "tool", "logs": tool_logs, "doc_ids": list(doc_ids), "chunks": payload_chunks})}\n\n'.encode()
            node = await run.next(node)
            continue

          if isinstance(node, End):
            break

          node = await run.next(node)

        if not assistant_text and run.result and run.result.output:
          fallback_text = str(run.result.output)
          async for token in _stream_tokens(fallback_text):
            assistant_text += token
            yield f'data: {json.dumps({"type": "delta", "token": token})}\n\n'.encode()
          last_text = assistant_text

      chunk_infos_deduped = _dedupe_chunk_infos(chunk_infos)

      if doc_ids:
        minio = get_minio_client()
        docs_lookup = {d['id']: d for d in get_vector_store().list_documents()}
        for doc_id in list(doc_ids):
          doc = docs_lookup.get(doc_id)
          if not doc:
            continue
          try:
            url = minio.presigned_url(doc['object_name'], expires=3600)
          except Exception:
            url = ''
          doc_meta.append({'id': doc['id'], 'filename': doc['filename'], 'url': url})

      assistant_msg = ChatMessage(
        session_id=chat_session.id,
        role='assistant',
        content=assistant_text,
        extras={
          'documents': doc_meta,
          'tools': sorted(tools_used),
          'support': support_called,
          'chunks': chunk_infos_deduped,
          'tool_logs': tool_logs,
        },
      )
      chat_session.updated_at = datetime.utcnow()
      session.add_all([assistant_msg, chat_session])
      session.commit()

      payload = {
        'type': 'final',
        'session_id': chat_session.id,
        'text': assistant_text,
        'documents': doc_meta,
        'support': support_called,
        'tools': sorted(tools_used),
        'chunks': chunk_infos_deduped,
        'tool_logs': tool_logs,
      }
      yield f'data: {json.dumps(payload)}\n\n'.encode()
    except Exception as exc:  # pragma: no cover - safety net for streaming
      error_payload = {
        'type': 'final',
        'session_id': chat_session.id,
        'text': f'Error: {exc}',
        'documents': [],
        'support': False,
        'tools': [],
        'chunks': [],
        'tool_logs': [],
      }
      yield f'data: {json.dumps(error_payload)}\n\n'.encode()

  headers = {
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no',
  }
  return StreamingResponse(event_generator(), media_type='text/event-stream', headers=headers)


@app.get('/')
async def health() -> dict[str, str]:
  return {'status': 'ok'}
