import asyncio
import contextlib
import operator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pymilvus import (
  Collection,
  CollectionSchema,
  DataType,
  FieldSchema,
  MilvusException,
  connections,
)

from .config import get_settings
from .embeddings import EmbeddingClient, get_embedding_client


@dataclass
class VectorHit:
  chunk_id: str
  document_id: str
  score: float
  text: str


class MilvusVectorStore:
  """Abstraction over Milvus for storing and searching embeddings."""

  def __init__(self, embedding_client: EmbeddingClient | None = None) -> None:
    self.settings = get_settings()
    self.embedding_client = embedding_client or get_embedding_client()
    self._connect()
    self.collection = self._ensure_chunk_collection()
    self.doc_collection = self._ensure_doc_collection()

  def _connect(self) -> None:
    alias = 'default'
    try:
      connections.connect(
        alias=alias,
        host=self.settings.milvus_host,
        port=str(self.settings.milvus_grpc_port),
      )
    except MilvusException:
      connections.disconnect(alias)
      connections.connect(
        alias=alias,
        host=self.settings.milvus_host,
        port=str(self.settings.milvus_grpc_port),
      )

  def _ensure_chunk_collection(self) -> Collection:
    dim = self.embedding_client.dimension
    fields = [
      FieldSchema(name='vector_id', dtype=DataType.INT64, is_primary=True, auto_id=True),
      FieldSchema(name='chunk_id', dtype=DataType.VARCHAR, max_length=128),
      FieldSchema(name='document_id', dtype=DataType.VARCHAR, max_length=128),
      FieldSchema(name='chunk_text', dtype=DataType.VARCHAR, max_length=8192),
      FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description='RAG document chunks')

    collection_name = self.settings.milvus_collection
    if collection_name not in list_collections():
      collection = Collection(name=collection_name, schema=schema)
      _maybe_await(
        collection.create_index(
          field_name='embedding',
          index_params={
            'index_type': 'IVF_FLAT',
            'metric_type': 'COSINE',
            'params': {'nlist': 1024},
          },
        )
      )
    else:
      collection = Collection(name=collection_name)
      existing_fields = {field.name for field in collection.schema.fields}
      if 'chunk_text' not in existing_fields:
        collection.drop()
        collection = Collection(name=collection_name, schema=schema)
        _maybe_await(
          collection.create_index(
            field_name='embedding',
            index_params={
              'index_type': 'IVF_FLAT',
              'metric_type': 'COSINE',
              'params': {'nlist': 1024},
            },
          )
        )
    collection.load()
    return collection

  def _ensure_doc_collection(self) -> Collection:
    # Milvus requires at least one vector field; include a small 2-d vector for metadata rows.
    fields = [
      FieldSchema(name='doc_pk', dtype=DataType.INT64, is_primary=True, auto_id=True),
      FieldSchema(name='doc_id', dtype=DataType.VARCHAR, max_length=128),
      FieldSchema(name='filename', dtype=DataType.VARCHAR, max_length=512),
      FieldSchema(name='object_name', dtype=DataType.VARCHAR, max_length=512),
      FieldSchema(name='size', dtype=DataType.INT64),
      FieldSchema(name='created_at', dtype=DataType.INT64),
      FieldSchema(name='meta_vector', dtype=DataType.FLOAT_VECTOR, dim=2),
    ]
    schema = CollectionSchema(fields=fields, description='RAG document metadata')
    name = f'{self.settings.milvus_collection}_meta'
    if name not in list_collections():
      collection = Collection(name=name, schema=schema)
    else:
      collection = Collection(name=name)
      existing_fields = {field.name for field in collection.schema.fields}
      if 'meta_vector' not in existing_fields:
        collection.drop()
        collection = Collection(name=name, schema=schema)
    # Ensure an index exists on the dummy vector field so the collection can load
    _maybe_await(
      collection.create_index(
        field_name='meta_vector',
        index_params={
          'index_type': 'FLAT',
          'metric_type': 'L2',
          'params': {},
        },
      )
    )
    collection.load()
    return collection

  def upsert(
    self,
    chunk_ids: list[str],
    document_ids: list[str],
    chunk_texts: list[str],
    embeddings: list[list[float]],
  ) -> None:
    if not embeddings:
      return
    assert len(chunk_ids) == len(document_ids) == len(embeddings) == len(chunk_texts)
    # Explicitly map fields to avoid misalignment with auto_id primary key.
    self.collection.insert(
      [
        chunk_ids,
        document_ids,
        chunk_texts,
        embeddings,
      ],
      fields=['chunk_id', 'document_id', 'chunk_text', 'embedding'],
    )
    self.collection.flush()
    self.collection.load()

  def search(self, query_embedding: list[float], limit: int) -> list[VectorHit]:
    self.collection.load()
    search_params = {'metric_type': 'COSINE', 'params': {'nprobe': 10}}
    results_raw: Any = self.collection.search(
      data=[query_embedding],
      anns_field='embedding',
      param=search_params,
      limit=limit,
      output_fields=['chunk_id', 'document_id', 'chunk_text'],
    )
    results = _maybe_await(results_raw)
    if not hasattr(results, '__getitem__'):
      return []
    hits: list[VectorHit] = []
    for res in results[0]:
      chunk_id = res.entity.get('chunk_id')
      doc_id = res.entity.get('document_id')
      text = res.entity.get('chunk_text') or ''
      hits.append(VectorHit(chunk_id=chunk_id, document_id=doc_id, text=text, score=float(res.score)))
    return hits

  def delete_document(self, document_id: str) -> None:
    """Remove vectors belonging to a document."""
    try:
      self.collection.delete(expr=f'document_id in ["{document_id}"]')
      self.collection.flush()
    except MilvusException:
      return

    # Remove metadata
    try:
      self.doc_collection.delete(expr=f'doc_id in ["{document_id}"]')
      self.doc_collection.flush()
    except MilvusException:
      return

  def upsert_document_meta(
    self, *, doc_id: str, filename: str, object_name: str, size: int, created_at: datetime
  ) -> None:
    # Delete existing rows for id to avoid dupes
    with contextlib.suppress(MilvusException):
      self.doc_collection.delete(expr=f'doc_id in ["{doc_id}"]')
    self.doc_collection.insert(
      [
        [doc_id],
        [filename],
        [object_name],
        [size],
        [int(created_at.timestamp())],
        [[0.0, 0.0]],
      ],
      fields=['doc_id', 'filename', 'object_name', 'size', 'created_at', 'meta_vector'],
    )
    self.doc_collection.flush()

  def list_documents(self) -> list[dict[str, Any]]:
    try:
      self.doc_collection.load()
      results = self.doc_collection.query(
        expr='',
        output_fields=['doc_id', 'filename', 'object_name', 'size', 'created_at'],
        limit=1000,
      )
    except MilvusException:
      return []
    docs: list[dict[str, Any]] = [
      {
        'id': row.get('doc_id'),
        'filename': row.get('filename'),
        'object_name': row.get('object_name'),
        'size': int(row.get('size') or 0),
        'created_at': datetime.fromtimestamp(int(row.get('created_at') or 0)),
      }
      for row in results or []
    ]
    # Sort by created_at desc
    docs.sort(key=operator.itemgetter('created_at'), reverse=True)
    return docs


def list_collections() -> list[str]:
  try:
    return connections.get_connection('default').list_collections()  # type: ignore[no-any-return]
  except Exception:
    return []


def _maybe_await(result: Any) -> Any:
  if asyncio.iscoroutine(result):
    return asyncio.get_event_loop().run_until_complete(result)
  return result
