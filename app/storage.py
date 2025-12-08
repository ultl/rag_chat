from pathlib import Path

from minio import Minio

from .config import get_settings


class MinioClient:
  """Minimal MinIO wrapper for storing uploaded files."""

  def __init__(self) -> None:
    settings = get_settings()
    endpoint = f'{settings.minio_host}:{settings.minio_port}'
    self.bucket = settings.minio_bucket
    self._client = Minio(
      endpoint=endpoint,
      access_key=settings.minio_root_user,
      secret_key=settings.minio_root_password,
      secure=False,
      region=settings.minio_region,
    )
    self._ensure_bucket()

  def _ensure_bucket(self) -> None:
    if not self._client.bucket_exists(self.bucket):
      self._client.make_bucket(self.bucket)

  def upload(self, file_path: Path, object_name: str, content_type: str | None = None) -> str:
    resolved_type = content_type or 'application/octet-stream'
    self._client.fput_object(
      bucket_name=self.bucket,
      object_name=object_name,
      file_path=str(file_path),
      content_type=resolved_type,
    )
    return object_name

  def presigned_url(self, object_name: str, expires: int = 3600) -> str:
    from datetime import timedelta

    return self._client.presigned_get_object(self.bucket, object_name, expires=timedelta(seconds=expires))

  def remove(self, object_name: str) -> None:
    self._client.remove_object(self.bucket, object_name)
