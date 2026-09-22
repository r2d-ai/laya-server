from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LAYA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "info"

    device: str = "cuda"
    strict_cuda: bool = True
    preload: str = "english,multilingual,typed-decisions"
    max_loaded: int = Field(default=3, ge=1, le=3)
    default_model: str = "english"
    auto_task_detection: bool = True
    max_concurrency: int = Field(default=2, ge=1, le=64)

    hf_token: str | None = None
    api_key: str | None = None

    @property
    def preload_models(self) -> list[str]:
        return [item.strip() for item in self.preload.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
