"""Typed configuration from environment variables.

The Spring analogue is `@ConfigurationProperties`: one typed object, validated
at startup, injected everywhere. The difference is that Pydantic validates at
RUNTIME, so a malformed env var fails the process on boot rather than at 3am
when the code path is first hit.

FAIL FAST, LOUDLY, AT STARTUP

A service that boots with `RATE_LIMIT_PER_MINUTE=abc` and only discovers the
problem on the 200th request is worse than one that refuses to start. Every
field here is typed and bounded, so a bad value is a startup crash with a
precise message.

NEVER LOG THIS OBJECT. It holds API keys. `SecretStr` makes that safe by
default: printing it yields `**********`, and you must call
`.get_secret_value()` deliberately. That single type prevents the most common
credential leak there is -- a debug log of the config object.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, validated on construction."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # unknown env vars are not our problem
        case_sensitive=False,
    )

    # -- service ----------------------------------------------------------
    app_name: str = "genai-service"
    environment: Literal["local", "dev", "staging", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"

    # -- provider ---------------------------------------------------------
    # Defaults to `fake` so the service boots and serves with NO credentials.
    # A reference implementation nobody can run is not a reference.
    llm_provider: Literal["fake", "openai", "anthropic"] = "fake"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    llm_max_retries: int = Field(default=3, ge=0, le=10)

    openai_api_key: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None

    # -- limits -----------------------------------------------------------
    api_keys: str = "dev-key-1,dev-key-2"
    rate_limit_per_minute: int = Field(default=60, gt=0)
    rate_limit_burst: int = Field(default=10, gt=0)
    max_prompt_chars: int = Field(default=32_000, gt=0)
    max_body_bytes: int = Field(default=256_000, gt=0)
    max_output_tokens: int = Field(default=2048, gt=0)

    # -- cost -------------------------------------------------------------
    daily_budget_usd: float = Field(default=5.0, gt=0)
    tenant_daily_budget_usd: float = Field(default=1.0, gt=0)

    # -- infrastructure ---------------------------------------------------
    redis_url: Optional[str] = None
    database_url: Optional[str] = None
    cors_origins: str = "*"

    @field_validator("api_keys")
    @classmethod
    def _reject_empty_key_list(cls, value: str) -> str:
        if not [k for k in value.split(",") if k.strip()]:
            raise ValueError("API_KEYS must contain at least one key")
        return value

    @property
    def valid_api_keys(self) -> set:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def cors_origin_list(self) -> list:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment in ("staging", "prod")

    def provider_key(self) -> Optional[str]:
        """Resolve the active provider's key, unwrapped at the point of use."""
        secret = self.openai_api_key if self.llm_provider == "openai" else self.anthropic_api_key
        return secret.get_secret_value() if secret else None

    def startup_check(self) -> list:
        """Warnings worth shouting about on boot.

        Returned rather than raised: a misconfigured non-prod service should
        still start, but nobody should be able to claim they were not told.
        """
        warnings: list = []
        if self.llm_provider != "fake" and not self.provider_key():
            warnings.append(f"llm_provider={self.llm_provider} but no API key is configured")
        if self.is_production:
            if self.llm_provider == "fake":
                warnings.append("running in production with the FAKE provider")
            if "dev-key-1" in self.valid_api_keys:
                warnings.append("default development API key is enabled in production")
            if self.cors_origins == "*":
                warnings.append("CORS is open to all origins in production")
        return warnings


@lru_cache
def get_settings() -> Settings:
    """Cached singleton.

    `lru_cache` makes this a singleton without a global, and -- more usefully --
    lets tests swap configuration by calling `get_settings.cache_clear()`.
    This is the FastAPI equivalent of a Spring singleton bean.
    """
    return Settings()
