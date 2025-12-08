from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  """Centralized application settings loaded from environment variables."""

  postgres_url: str = Field(default='', alias='POSTGRES_URL')
  server_host: str = Field(default='0.0.0.0', alias='FASTAPI_HOST')
  server_port: int = Field(default=8000, alias='FASTAPI_PORT')
  redis_host: str = Field(default='127.0.0.1', alias='REDIS_HOST')
  redis_port: int = Field(default=6379, alias='REDIS_PORT')

  minio_host: str = Field(default='127.0.0.1', alias='MINIO_HOST')
  minio_port: int = Field(default=9000, alias='MINIO_PORT')
  minio_root_user: str = Field(default='minioadmin', alias='MINIO_ROOT_USER')
  minio_root_password: str = Field(default='minioadmin', alias='MINIO_ROOT_PASSWORD')
  minio_region: str = Field(default='us-east-1', alias='MINIO_REGION')
  minio_bucket: str = Field(default='rag-bucket', alias='MINIO_BUCKET')

  milvus_host: str = Field(default='127.0.0.1', alias='MILVUS_HOST')
  milvus_grpc_port: int = Field(default=19530, alias='MILVUS_GRPC_PORT')
  milvus_http_port: int = Field(default=9091, alias='MILVUS_HTTP_PORT')
  milvus_collection: str = Field(default='rag_chunks', alias='MILVUS_COLLECTION')

  openai_base_url: str = Field(default='', alias='OPENAI_BASE_URL')
  openai_api_key: str = Field(default='', alias='OPENAI_API_KEY')
  ollama_chat_model: str = Field(default='qwen3-vl:8b-instruct', alias='OLLAMA_CHAT_MODEL')
  ollama_embed_model: str = Field(default='embeddinggemma:300m', alias='OLLAMA_EMBED_MODEL')

  cache_ttl_seconds: int = Field(default=3600, alias='CACHE_TTL_SECONDS')
  retrieval_top_k: int = Field(default=6, alias='RETRIEVAL_TOP_K')
  chunk_size: int = Field(default=1200, alias='CHUNK_SIZE')

  upload_dir: Path = Field(default=Path('uploaded_files'), alias='UPLOAD_DIR')

  model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore', populate_by_name=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
  settings = Settings()

  # Ensure SQLAlchemy uses the psycopg (v3) driver even if the URL is missing it.
  if settings.postgres_url.startswith('postgresql://'):
    settings.postgres_url = 'postgresql+psycopg://' + settings.postgres_url[len('postgresql://') :]
  elif settings.postgres_url.startswith('postgres://'):
    settings.postgres_url = 'postgresql+psycopg://' + settings.postgres_url[len('postgres://') :]

  settings.upload_dir.mkdir(parents=True, exist_ok=True)
  return settings
