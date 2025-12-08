from functools import lru_cache

import redis

from .config import get_settings
from .embeddings import EmbeddingClient, get_embedding_client
from .storage import MinioClient
from .vectorstore import MilvusVectorStore


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
  settings = get_settings()
  return redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)


@lru_cache(maxsize=1)
def get_minio_client() -> MinioClient:
  return MinioClient()


@lru_cache(maxsize=1)
def get_vector_store() -> MilvusVectorStore:
  return MilvusVectorStore()


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingClient:
  return get_embedding_client()
