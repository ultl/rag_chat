from functools import lru_cache

from openai import OpenAI

from .config import get_settings


class EmbeddingClient:
  """Helper to generate embeddings via the OpenAI-compatible Ollama endpoint."""

  def __init__(self) -> None:
    settings = get_settings()
    self.model = settings.ollama_embed_model
    self._client = OpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key)
    self._dimension: int | None = None

  def embed(self, text: str) -> list[float]:
    response = self._client.embeddings.create(model=self.model, input=text)
    return response.data[0].embedding  # type: ignore[no-any-return]

  def embed_batch(self, texts: list[str]) -> list[list[float]]:
    if not texts:
      return []
    response = self._client.embeddings.create(model=self.model, input=texts)
    return [item.embedding for item in response.data]  # type: ignore[no-any-return]

  @property
  def dimension(self) -> int:
    if self._dimension is None:
      vector = self.embed('dimension-probe')
      self._dimension = len(vector)
    return self._dimension


@lru_cache(maxsize=1)
def get_embedding_client() -> EmbeddingClient:
  return EmbeddingClient()
