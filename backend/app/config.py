from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent


def _hydrate_env_from_dotenv_files() -> None:
    """
    Merge .env files in order. Later files win only for non-empty values, so
    placeholder keys in `.env.local` (e.g. SHOPIFY_ACCESS_TOKEN= with no value)
    do not erase credentials from `.env`.
    """
    paths = (
        _REPO_ROOT / ".env",
        _BACKEND_DIR / ".env",
        _BACKEND_DIR / ".env.local",
    )
    merged: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        for raw_key, raw_val in dotenv_values(path).items():
            if not raw_key:
                continue
            key = raw_key.strip()
            if raw_val is None:
                continue
            val = str(raw_val).strip()
            if val == "":
                continue
            merged[key] = val
    for key, val in merged.items():
        os.environ[key] = val


_hydrate_env_from_dotenv_files()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    shopify_shop_name: str = ""
    shopify_access_token: str = ""
    shopify_api_version: str = "2025-07"
    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    gemini_model: str = "gemini-2.5-flash"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    shopify_max_pages_per_request: int = 40
    shopify_request_timeout_seconds: float = 60.0
    shopify_max_retries: int = 5

    @field_validator("gemini_model", mode="before")
    @classmethod
    def _gemini_model_fallback(cls, v: object) -> str:
        if v is None or (isinstance(v, str) and not str(v).strip()):
            return "gemini-2.5-flash"
        raw = str(v).strip()
        # Google has retired gemini-2.0-flash for new API users; use current Flash.
        slug = raw.lower().removeprefix("models/").strip()
        if slug.startswith("gemini-2.0-flash"):
            return "gemini-2.5-flash"
        return raw


@lru_cache
def get_settings() -> Settings:
    return Settings()
