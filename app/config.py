"""
Centralized configuration via Pydantic Settings.

All values are sourced from environment variables (or a `.env` file).
No secrets are ever hardcoded.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application settings sourced from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── RabbitMQ ─────────────────────────────────────────────
    rabbitmq_host: str = Field(
        default="rabbitmq",
        description="RabbitMQ server hostname.",
    )
    rabbitmq_port: int = Field(
        default=5672,
        ge=1,
        le=65535,
        description="RabbitMQ AMQP port.",
    )
    rabbitmq_default_user: str = Field(
        default="guest",
        description="RabbitMQ authentication username.",
    )
    rabbitmq_default_pass: SecretStr = Field(
        default=SecretStr("guest"),
        description="RabbitMQ authentication password.",
    )
    rabbitmq_vhost: str = Field(
        default="/",
        description="RabbitMQ virtual host.",
    )

    # ── Redis ────────────────────────────────────────────────
    redis_url: str = Field(
        default="redis://redis:6379/0",
        description="Redis connection URL.",
    )

    # ── PostgreSQL ───────────────────────────────────────────
    postgres_dsn: str = Field(
        default="postgresql+asyncpg://mas_user:changeme@postgres:5432/mas_events",
        description="Async PostgreSQL connection DSN.",
    )

    # ── FastAPI ──────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="API bind host.")  # noqa: S104
    api_port: int = Field(default=8000, ge=1, le=65535, description="API bind port.")
    api_log_level: str = Field(default="info", description="Uvicorn log level.")

    @property
    def rabbitmq_url(self) -> str:
        """Build the AMQP connection URL from individual components."""
        password = self.rabbitmq_default_pass.get_secret_value()
        return (
            f"amqp://{self.rabbitmq_default_user}:{password}"
            f"@{self.rabbitmq_host}:{self.rabbitmq_port}"
            f"{self.rabbitmq_vhost}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
