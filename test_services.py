from __future__ import annotations

import contextlib
import time
from pathlib import Path

import psycopg
import redis
from environs import Env
from minio import Minio
from openai import OpenAI
from pymilvus import connections, utility

ROOT = Path(__file__).resolve().parent.parent
env = Env()
env.read_env(ROOT / '.env')


def wait(delay: float = 2.0) -> None:
  time.sleep(delay)


def test_postgres() -> None:
  host = env.str('POSTGRES_HOST', '127.0.0.1')
  port = env.int('POSTGRES_PORT', 5432)
  user = env.str('POSTGRES_USER', 'rag_user')
  password = env.str('POSTGRES_PASSWORD', 'rag_password')
  dbname = env.str('POSTGRES_DB', 'rag_db')

  for attempt in range(10):
    try:
      conn = psycopg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
        connect_timeout=5,
      )
      with conn.cursor() as cur:
        cur.execute('SELECT 1;')
        cur.fetchone()
      conn.close()
      print(f'PostgreSQL connection successful on {host}:{port}')
      return
    except Exception as exc:
      if attempt == 9:
        raise
      print(f'PostgreSQL not ready yet ({exc}), retrying...')
      wait()


def test_redis() -> None:
  host = env.str('REDIS_HOST', '127.0.0.1')
  port = env.int('REDIS_PORT', 6379)
  client = redis.Redis(host=host, port=port, decode_responses=True, socket_connect_timeout=5)

  for attempt in range(10):
    try:
      client.ping()
      print(f'Redis connection successful on {host}:{port}')
      return
    except Exception as exc:
      if attempt == 9:
        raise
      print(f'Redis not ready yet ({exc}), retrying...')
      wait()


def test_minio() -> None:
  host = env.str('MINIO_HOST', '127.0.0.1')
  endpoint = f'{host}:{env.str("MINIO_PORT", "9000")}'
  access_key = env.str('MINIO_ROOT_USER', 'minioadmin')
  secret_key = env.str('MINIO_ROOT_PASSWORD', 'minioadmin')
  region = env.str('MINIO_REGION', 'us-east-1')
  bucket = env.str('MINIO_BUCKET', 'rag-bucket')

  client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False, region=region)

  for attempt in range(10):
    try:
      if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"Created MinIO bucket '{bucket}'")
      else:
        print(f"MinIO bucket '{bucket}' is accessible")
      return
    except Exception as exc:
      if attempt == 9:
        raise
      print(f'MinIO not ready yet ({exc}), retrying...')
      wait(3.0)


def test_milvus() -> None:
  host = env.str('MILVUS_HOST', '127.0.0.1')
  port = env.int('MILVUS_GRPC_PORT', 19530)

  for attempt in range(20):
    with contextlib.suppress(Exception):
      connections.disconnect(alias='default')

    try:
      connections.connect(alias='default', host=host, port=port, timeout=5)
      version = utility.get_server_version()
      print(f'Milvus connection successful on {host}:{port} (server {version})')
      connections.disconnect(alias='default')
      return
    except Exception as exc:
      if attempt == 19:
        raise
      print(f'Milvus not ready yet ({exc}), retrying...')
      wait(3.0)


def test_ollama_models() -> None:
  base_url = env.str('OPENAI_BASE_URL', 'http://127.0.0.1:11434/v1')
  api_key = env.str('OPENAI_API_KEY', 'ollama')
  chat_model = env.str('OLLAMA_CHAT_MODEL', 'qwen3-vl:8b-instruct')
  embed_model = env.str('OLLAMA_EMBED_MODEL', 'embeddinggemma:300m')

  client = OpenAI(base_url=base_url, api_key=api_key)

  def _chat(model: str) -> None:
    resp = client.chat.completions.create(
      model=model,
      messages=[
        {'role': 'system', 'content': 'You are a test assistant.'},
        {'role': 'user', 'content': "Reply with 'pong'."},
      ],
      max_tokens=5,
    )
    content = (resp.choices[0].message.content or '').strip()
    print(f"Ollama chat model '{model}' reply: {content}")

  def _embed(model: str) -> None:
    resp = client.embeddings.create(
      model=model,
      input='ping',
    )
    dims = len(resp.data[0].embedding) if resp.data else 0
    print(f"Ollama embedding model '{model}' produced vector of dim {dims}")

  for attempt in range(5):
    try:
      _chat(chat_model)
      _embed(embed_model)
      return
    except Exception as exc:
      if attempt == 4:
        raise
      print(f'Ollama endpoint not ready yet ({exc}), retrying...')
      wait(2.0)


def main() -> None:
  print('Checking PostgreSQL...')
  test_postgres()
  print('Checking Redis...')
  test_redis()
  print('Checking MinIO...')
  test_minio()
  print('Checking Milvus...')
  test_milvus()
  print('Checking Ollama models via OpenAI client...')
  test_ollama_models()
  print('All services verified successfully.')


if __name__ == '__main__':
  main()
