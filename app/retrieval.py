import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

from openai import OpenAI
from redis import Redis
from sqlmodel import Session

from .config import get_settings
from .embeddings import EmbeddingClient, get_embedding_client
from .vectorstore import MilvusVectorStore, VectorHit


@dataclass
class ChunkContext:
  chunk_id: str
  document_id: str
  text: str
  score: float


@dataclass
class RetrievalResult:
  document_ids: list[str]
  chunks: list[ChunkContext]


class QueryRewriter:
  """Rewrite incoming queries into English and Japanese variants."""

  def __init__(self) -> None:
    settings = get_settings()
    self.client = OpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key)
    self.model = settings.ollama_chat_model

  def rewrite(self, query: str) -> dict[str, str]:
    prompt = (
      'Rewrite the user query into concise English and Japanese equivalents for search. '
      "Return JSON with keys 'english' and 'japanese'. Avoid extra text."
    )
    response = self.client.chat.completions.create(
      model=self.model,
      messages=[
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': query},
      ],
      max_tokens=200,
    )
    content = (response.choices[0].message.content or '').strip()
    try:
      payload = json.loads(content)
      english = payload.get('english') or query
      japanese = payload.get('japanese') or query
    except Exception:
      english, japanese = self._fallback_parse(content, query)
    return {'english': english.strip(), 'japanese': japanese.strip()}

  @staticmethod
  def _fallback_parse(content: str, query: str) -> tuple[str, str]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) >= 2:
      return lines[0], lines[1]
    return query, query


class Retriever:
  """Perform cached retrieval over Milvus and hydrate chunk text from PostgreSQL."""

  def __init__(
    self,
    redis_client: Redis,
    session: Session,
    vector_store: MilvusVectorStore,
    embedder: EmbeddingClient | None = None,
  ) -> None:
    self.settings = get_settings()
    self.redis = redis_client
    self.session = session
    self.vector_store = vector_store
    self.embedder = embedder or get_embedding_client()
    self.rewriter = QueryRewriter()

  def retrieve(self, query: str) -> RetrievalResult:
    rewrites = self.rewriter.rewrite(query)
    doc_ids: set[str] = set()
    chunk_contexts: dict[str, ChunkContext] = {}

    for lang, text in rewrites.items():
      cached = self._get_cached(lang, text)
      if cached:
        doc_ids.update(cached['doc_ids'])
        chunk_contexts.update(cached['chunks'])
        continue

      embedding = self.embedder.embed(text)
      hits = self.vector_store.search(embedding, limit=self.settings.retrieval_top_k)
      hydrated = self._hydrate_hits(hits)
      for ctx in hydrated:
        doc_ids.add(ctx.document_id)
        chunk_contexts[ctx.chunk_id] = ctx
      self._set_cached(lang, text, hydrated, list(doc_ids))

    ordered_chunks = sorted(chunk_contexts.values(), key=lambda c: c.score)
    ordered_chunks.reverse()
    return RetrievalResult(document_ids=list(doc_ids), chunks=ordered_chunks)

  def _hydrate_hits(self, hits: Sequence[VectorHit]) -> list[ChunkContext]:
    contexts: list[ChunkContext] = []
    for hit in hits:
      text = getattr(hit, 'text', None)
      if text:
        contexts.append(
          ChunkContext(
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            text=text,
            score=hit.score,
          )
        )
    return contexts

  def _cache_key(self, lang: str, query: str) -> str:
    digest = sha256(query.lower().encode()).hexdigest()
    return f'rag:retrieval:{lang}:{digest}'

  def _get_cached(self, lang: str, query: str) -> dict | None:
    key = self._cache_key(lang, query)
    raw = self.redis.get(key)
    if not raw:
      return None
    try:
      data = json.loads(str(raw))
      chunks = {
        item['chunk_id']: ChunkContext(
          chunk_id=item['chunk_id'],
          document_id=item['document_id'],
          text=item['text'],
          score=item['score'],
        )
        for item in data.get('chunks', [])
      }
      doc_ids = data.get('doc_ids', [])
      if not doc_ids:
        return None
      return {'doc_ids': doc_ids, 'chunks': chunks}
    except Exception:
      return None

  def _set_cached(self, lang: str, query: str, chunks: list[ChunkContext], doc_ids: list[str]) -> None:
    if not doc_ids:
      return
    key = self._cache_key(lang, query)
    payload = {
      'doc_ids': doc_ids,
      'chunks': [
        {
          'chunk_id': chunk.chunk_id,
          'document_id': chunk.document_id,
          'text': chunk.text,
          'score': chunk.score,
        }
        for chunk in chunks
      ],
    }
    self.redis.set(key, json.dumps(payload), ex=self.settings.cache_ttl_seconds)
