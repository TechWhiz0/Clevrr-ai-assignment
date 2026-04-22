"""Shopify API client + LangChain tools in one place."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import httpx
from langchain_core.callbacks.manager import CallbackManagerForToolRun
from langchain_core.tools import tool
from langchain_experimental.tools import PythonAstREPLTool

from app.config import Settings

# Assignment: unsafe Shopify operations (non-GET / writes) must use this exact reply.
FORBIDDEN_SHOPIFY_OPERATION = "This operation is not permitted."

# Shopify / edge transient responses worth retrying (bounded exponential backoff).
_TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({429, 502, 503, 504})
_ENDPOINT_SAFE_RE = re.compile(r"^[a-zA-Z0-9_./-]+$")

# Shell / eval — block with the assignment-mandated phrase.
_UNSAFE_PYTHON_PATTERNS = (
    "import os",
    "import subprocess",
    "subprocess.",
    "os.system",
    "os.popen",
    "__import__",
    "eval(",
    "exec(",
    "compile(",
    "open(",
    "socket.",
)

# Any HTTP client or mutating verb → non-GET / unsafe for Shopify (assignment wording).
_HTTP_OR_MUTATION_PATTERNS = (
    "requests.",
    "httpx.",
    "urllib.request",
    "aiohttp",
    "http.client",
    ".post(",
    ".put(",
    ".delete(",
    ".patch(",
)


def normalize_shop_host(shop: str) -> str:
    s = (shop or "").strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = s.rstrip("/")
    if not s.endswith(".myshopify.com"):
        if "." not in s:
            s = f"{s}.myshopify.com"
    return s


def _extract_page_info_from_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        section = part.strip()
        if 'rel="next"' not in section and "rel='next'" not in section:
            continue
        m = re.search(r"<([^>]+)>", section)
        if not m:
            continue
        url = m.group(1)
        qs = parse_qs(urlparse(url).query)
        infos = qs.get("page_info")
        if infos:
            return infos[0]
    return None


def _retry_sleep_seconds(response: httpx.Response, attempt: int, max_sleep: float) -> float:
    """Prefer Retry-After when present (seconds); else exponential backoff."""
    ra = (response.headers.get("Retry-After") or "").strip()
    if ra.isdigit():
        return min(float(ra), max_sleep)
    return min(2.0**attempt, max_sleep)


def validate_get_endpoint_and_query(
    endpoint: str,
    query: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Enforce safe GET-only usage before any HTTP call.
    Returns an error payload dict if invalid; otherwise None.
    """
    raw = (endpoint or "").strip()
    if not raw:
        return {"error": "missing_endpoint", "detail": "endpoint is required"}

    if ".." in raw or "\n" in raw or "\r" in raw:
        return {"error": "invalid_endpoint", "detail": "path must not contain traversal or newlines"}

    if "://" in raw.lower():
        return {"error": "invalid_endpoint", "detail": "full URLs are not allowed; use a resource path like orders"}

    path_check = raw.removesuffix(".json").strip("/")
    if not path_check or not _ENDPOINT_SAFE_RE.fullmatch(path_check):
        return {
            "error": "invalid_endpoint",
            "detail": "endpoint may only contain letters, digits, /, -, _ and .json suffix",
        }

    qp = query or {}
    forbidden_keys = frozenset(k.lower() for k in ("_method", "method", "http_method"))
    for key in qp:
        if str(key).lower() in forbidden_keys:
            return {
                "error": "forbidden_query_param",
                "detail": FORBIDDEN_SHOPIFY_OPERATION,
            }

    return None


class ShopifyGetClient:
    """HTTP GET only — no POST/PUT/PATCH/DELETE."""

    def __init__(self, settings: Settings, shop_host: str, access_token: str) -> None:
        self._settings = settings
        self._shop = normalize_shop_host(shop_host)
        self._token = access_token
        self._base = f"https://{self._shop}/admin/api/{settings.shopify_api_version}"

    def _headers(self) -> dict[str, str]:
        return {
            "X-Shopify-Access-Token": self._token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def get_json(
        self,
        endpoint: str,
        query: dict[str, Any] | None = None,
        *,
        auto_paginate: bool = True,
    ) -> dict[str, Any]:
        bad = validate_get_endpoint_and_query(endpoint, query)
        if bad is not None:
            return bad

        path = endpoint.strip().lstrip("/")
        if not path.endswith(".json"):
            path = f"{path}.json"

        merged: dict[str, Any] = dict(query or {})
        if auto_paginate and "page_info" not in merged:
            merged.setdefault("limit", "250")

        all_rows: list[Any] | None = None
        root_key: str | None = None
        page_info: str | None = None
        pages = 0

        max_retries = max(1, self._settings.shopify_max_retries)
        max_sleep = max(5.0, float(self._settings.shopify_request_timeout_seconds))

        timeout = httpx.Timeout(self._settings.shopify_request_timeout_seconds)
        with httpx.Client(timeout=timeout) as client:
            while True:
                pages += 1
                params = dict(merged)
                if page_info:
                    params = {"limit": str(params.get("limit", "250")), "page_info": page_info}

                url = f"{self._base}/{path}"
                attempt = 0
                resp: httpx.Response | None = None
                while True:
                    attempt += 1
                    try:
                        resp = client.get(url, headers=self._headers(), params=params)
                    except httpx.RequestError as e:
                        if attempt >= max_retries:
                            return {"error": "shopify_request_error", "detail": str(e)}
                        time.sleep(min(2.0**attempt, max_sleep))
                        continue

                    assert resp is not None
                    if resp.status_code in _TRANSIENT_STATUS_CODES:
                        if attempt >= max_retries:
                            return {
                                "error": "shopify_transient_or_rate_limit",
                                "status_code": resp.status_code,
                                "detail": resp.text[:2000]
                                if resp.status_code != 429
                                else "Shopify returned HTTP 429 after retries.",
                            }
                        time.sleep(_retry_sleep_seconds(resp, attempt, max_sleep))
                        continue

                    if resp.status_code >= 400:
                        return {
                            "error": "shopify_http_error",
                            "status_code": resp.status_code,
                            "detail": resp.text[:2000],
                        }
                    break

                try:
                    payload = resp.json()
                except json.JSONDecodeError:
                    return {"error": "invalid_json", "detail": resp.text[:2000]}

                if isinstance(payload, dict) and payload.get("errors") is not None:
                    return {"error": "shopify_api_errors", "detail": payload.get("errors")}

                if not auto_paginate:
                    return {"data": payload, "pages_fetched": 1}

                if not isinstance(payload, dict):
                    return {"data": payload, "pages_fetched": pages}

                keys = [k for k in payload.keys() if k != "errors"]
                if len(keys) != 1:
                    return {"data": payload, "pages_fetched": pages}

                rk = keys[0]
                chunk = payload.get(rk)
                if not isinstance(chunk, list):
                    return {"data": payload, "pages_fetched": pages}

                if root_key is None:
                    root_key = rk
                    all_rows = []
                elif rk != root_key:
                    return {"data": payload, "pages_fetched": pages}

                assert all_rows is not None
                all_rows.extend(chunk)

                next_pi = _extract_page_info_from_link(resp.headers.get("link"))
                if not next_pi or not chunk:
                    return {
                        "data": {root_key: all_rows},
                        "pages_fetched": pages,
                        "total_records": len(all_rows),
                    }

                if pages >= self._settings.shopify_max_pages_per_request:
                    return {
                        "data": {root_key: all_rows},
                        "pages_fetched": pages,
                        "total_records": len(all_rows),
                        "warning": "Pagination stopped at max_pages cap; results may be incomplete.",
                    }

                page_info = next_pi


def build_shopify_get_tool(settings: Settings, shop_host: str, access_token: str):
    client = ShopifyGetClient(settings, shop_host, access_token)

    @tool("get_shopify_data")
    def get_shopify_data(spec: str) -> str:
        """Shopify Admin REST GET tool. `spec` must be a JSON string."""
        try:
            payload = json.loads(spec) if spec else {}
        except json.JSONDecodeError as e:
            return json.dumps({"error": "invalid_json_spec", "detail": str(e)})

        for forbidden in ("method", "http_method", "_method"):
            if forbidden not in payload:
                continue
            raw_m = payload.get(forbidden)
            if raw_m is None or str(raw_m).strip() == "":
                continue
            if str(raw_m).strip().upper() != "GET":
                return json.dumps({"error": "method_not_allowed", "detail": FORBIDDEN_SHOPIFY_OPERATION})

        endpoint = payload.get("endpoint")
        if not endpoint or not isinstance(endpoint, str):
            return json.dumps({"error": "missing_endpoint"})
        qp = payload.get("query_params")
        if qp is not None and not isinstance(qp, dict):
            return json.dumps({"error": "query_params_must_be_object"})
        auto = bool(payload.get("auto_paginate", True))

        pre = validate_get_endpoint_and_query(str(endpoint), qp if isinstance(qp, dict) else None)
        if pre is not None:
            return json.dumps(pre)

        result = client.get_json(str(endpoint), qp, auto_paginate=auto)
        try:
            return json.dumps(result, default=str)[:120_000]
        except (TypeError, ValueError):
            return json.dumps({"error": "serialization_error"}, default=str)

    return get_shopify_data


class GuardedPythonAstReplTool(PythonAstREPLTool):
    """PythonAstREPLTool with a light pre-check to discourage network/shell usage."""

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        lowered = query.lower()
        for pat in _HTTP_OR_MUTATION_PATTERNS:
            if pat.lower() in lowered:
                return FORBIDDEN_SHOPIFY_OPERATION
        for pat in _UNSAFE_PYTHON_PATTERNS:
            if pat.lower() in lowered:
                return FORBIDDEN_SHOPIFY_OPERATION
        return super()._run(query, run_manager=run_manager)


def build_python_analysis_tool() -> GuardedPythonAstReplTool:
    return GuardedPythonAstReplTool(
        name="python_repl",
        description=(
            "PythonAstREPLTool for analysis: aggregates, markdown tables, matplotlib charts "
            "(savefig to buffer, base64, use data:image/png;base64,... in markdown). "
            "Input is Python code as a single string. Never perform HTTP; fetch with get_shopify_data first."
        ),
    )
