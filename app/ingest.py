import contextlib
import mimetypes
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from chonkie.chunker.recursive import RecursiveChunker
from markitdown import MarkItDown

from app.config import get_settings
from app.embeddings import EmbeddingClient, get_embedding_client
from app.storage import MinioClient
from app.vectorstore import MilvusVectorStore


class IngestionPipeline:
  """End-to-end pipeline from raw file to vectorized storage."""

  def __init__(
    self,
    minio: MinioClient,
    vector_store: MilvusVectorStore,
    embedder: EmbeddingClient | None = None,
  ) -> None:
    self.minio = minio
    self.vector_store = vector_store
    self.embedder = embedder or get_embedding_client()
    self.settings = get_settings()
    self.chunker = RecursiveChunker(chunk_size=self.settings.chunk_size)
    self.converter = MarkItDown()

  def ingest_upload(self, file_bytes: bytes, filename: str, content_type: str | None = None) -> 'DocumentMeta':
    content_type = content_type or mimetypes.guess_type(filename)[0] or 'application/octet-stream'
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
      tmp.write(file_bytes)
      tmp.flush()
      tmp_path = Path(tmp.name)
    try:
      return self._process_file(tmp_path, filename=filename, content_type=content_type)
    finally:
      with contextlib.suppress(Exception):
        tmp_path.unlink()

  def _process_file(self, path: Path, filename: str, content_type: str) -> 'DocumentMeta':
    object_name = f'uploads/{uuid4()}_{Path(filename).name}'
    self.minio.upload(path, object_name=object_name, content_type=content_type)

    converted = self.converter.convert(str(path))
    text_content = (converted.text_content or '').strip()
    if not text_content:
      msg = 'Unable to extract text from document'
      raise ValueError(msg)

    doc_id = str(uuid4())
    created_at = datetime.utcnow()

    chunks = self.chunker.chunk(text_content)
    chunk_texts: list[str] = []
    chunk_ids: list[str] = []
    for chunk in chunks:
      clean_text = chunk.text.strip()
      if not clean_text:
        continue
      chunk_texts.append(clean_text)
      chunk_ids.append(chunk.id)

    if not chunk_ids:
      msg = 'No usable text chunks found in document'
      raise ValueError(msg)

    embeddings = self.embedder.embed_batch(chunk_texts)
    self.vector_store.upsert(
      chunk_ids=chunk_ids,
      document_ids=[doc_id] * len(chunk_ids),
      chunk_texts=chunk_texts,
      embeddings=embeddings,
    )
    self.vector_store.upsert_document_meta(
      doc_id=doc_id,
      filename=filename,
      object_name=object_name,
      size=path.stat().st_size,
      created_at=created_at,
    )
    return DocumentMeta(
      id=doc_id, filename=filename, object_name=object_name, size=path.stat().st_size, created_at=created_at
    )


@dataclass
class DocumentMeta:
  id: str
  filename: str
  object_name: str
  size: int
  created_at: datetime
