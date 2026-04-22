"""FastAPI entrypoint for the Shopify AI Agent."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agent_service import run_agent_turn
from app.config import get_settings
from app.output_sanitize import sanitize_agent_text
from app.shopify import normalize_shop_host

app = FastAPI(title="Shopify Store Analyst Agent", version="1.0.0")


def _is_gemini_quota_error(exc: BaseException) -> bool:
    """Detect Google Gemini free-tier / billing quota errors for clearer API responses."""
    parts: list[str] = []
    e: BaseException | None = exc
    for _ in range(6):
        if e is None:
            break
        parts.append(str(e).lower())
        e = e.__cause__
    blob = " ".join(parts)
    if "resourceexhausted" in blob.replace(" ", ""):
        return True
    if "429" in blob and ("quota" in blob or "rate" in blob):
        return True
    if "quota exceeded" in blob or "exceeded your current quota" in blob:
        return True
    return False


@app.on_event("startup")
def _warm_settings() -> None:
    get_settings()


def _cors_origins() -> list[str]:
    raw = get_settings().cors_origins
    return [o.strip() for o in raw.split(",") if o.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str


def _shop_credentials_from_env() -> tuple[str, str]:
    """Single-store deployment: shop host and token come only from server environment."""
    settings = get_settings()
    token = (settings.shopify_access_token or "").strip()
    if not token:
        raise HTTPException(status_code=500, detail="Shopify access token is not configured on the server.")
    host = normalize_shop_host(settings.shopify_shop_name)
    if not host:
        raise HTTPException(
            status_code=500,
            detail="SHOPIFY_SHOP_NAME is not configured on the server.",
        )
    return host, token


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    shop_host, token = _shop_credentials_from_env()
    history = [{"role": m.role, "content": m.content} for m in body.history]
    try:
        raw = run_agent_turn(
            user_message=body.message,
            history=history,
            shop_host=shop_host,
            access_token=token,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        if _is_gemini_quota_error(e):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Gemini API quota exceeded (HTTP 429). Wait and retry, try another GEMINI_MODEL in .env "
                    "(for example gemini-2.5-flash or gemini-2.5-flash-lite), or check billing and limits at "
                    "https://ai.google.dev/gemini-api/docs/rate-limits"
                ),
            ) from e
        raise HTTPException(status_code=500, detail=f"Agent error: {e!s}") from e

    cleaned = sanitize_agent_text(raw)
    if not cleaned.strip() and (raw or "").strip():
        cleaned = (
            "The reply was empty after safety formatting. Try asking again in plain language, "
            "or check that the model returned text (not only tool calls)."
        )
    return ChatResponse(reply=cleaned)
