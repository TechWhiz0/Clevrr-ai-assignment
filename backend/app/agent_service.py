"""Tool-using agent built on Gemini (LangChain chat model + tools), ReAct-style loop."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import Settings, get_settings
from app.shopify import FORBIDDEN_SHOPIFY_OPERATION, build_python_analysis_tool, build_shopify_get_tool

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """You are a senior Shopify analyst. You help merchants understand orders, products, and customers.

Rules:
- Every numeric fact MUST come from `get_shopify_data`. Never invent metrics.
- Shopify Admin API use is **GET-only** through `get_shopify_data`. If the user asks for POST, PUT, PATCH, DELETE, GraphQL mutations, or any write/modify/delete on Shopify—or if a tool returns a method error—your **entire** reply MUST be exactly this sentence and nothing else: """ + FORBIDDEN_SHOPIFY_OPERATION + """
- Use `python_repl` only to analyze structures you already fetched (counts, sums, grouping, markdown tables, charts). Never use it for HTTP calls to Shopify; that is GET-only via `get_shopify_data`.
- Do NOT include fenced code blocks in the final reply (no ```python, ```js, etc.). Charts must be a markdown **image** only (see Chart section).
- If Shopify returns errors or rate limits, say so. If a **time-filtered** request returns **zero orders**, do not stop immediately: run **one** broader `get_shopify_data` (e.g. `created_at_min` ~400 days ago, still `status=any`) or omit date bounds with a reasonable `limit` so you can see whether the store only has **older** sales—then explain both the window you tried and what exists (many dev stores have no data in “current” months).
- You CAN compute date ranges in `python_repl` (e.g. last 7 days from today's date) when building `created_at_min` / `created_at_max` for the next `get_shopify_data` call—do not refuse for lacking a calendar; use UTC ISO-8601 strings.
- Never call tools named `utcnow` or `timedelta`; those are not available tools. If you need dates, use `python_repl`.

**Response format (use these section headings in markdown when there is anything to show):**

## Summary
One or two sentences: what you found and whether data was limited.

## Key metrics
Bullet list of **aggregate metrics** only if supported by data (e.g. **Order count**, **Total revenue**, **Average order value**, **Unique customers**). Use the store currency when present.

## Detail
**Tabular summaries** as GitHub-flavored markdown tables for breakdowns (by product, city, week, etc.). Align numbers readably; round to 2 decimals for money where helpful.

## Recommendations
Concrete **business recommendations** (bullets): match the count the user asked for if they specified one (e.g. exactly two); otherwise 2–4 bullets grounded in the numbers above.

## Chart (optional)
For graphs or trends, use **matplotlib only** in `python_repl`: build the figure (title, axis labels), then save to an in-memory PNG and emit **one** markdown image line in your reply—**no code fences**.

Pattern (adapt variable names as needed):
import io, base64
buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight")
plt.close(fig)
b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
# Then in the markdown body: ![Weekly order volume](data:image/png;base64,<paste b64 here>)

If the question is trivially answered by a single number, you may use a shorter reply but still include **Summary** and **Key metrics** when applicable.

Shopify tips:
- Orders: always use `"status":"any"` unless the user asks otherwise so closed/cancelled/archived-era orders still appear in lists.
- `created_at_min` / `created_at_max` must be ISO-8601 UTC. For **last calendar month**, use that month’s start 00:00:00Z through the month’s end (exclusive next month 00:00:00Z is clearest). **State the exact UTC range** you queried in ## Summary so the user can see why a window might be empty.
- Prefer `total_price` on orders for revenue; use `shipping_address` / `billing_address` city when present for city breakdowns.
- **Top products:** each order includes `line_items`. In `python_repl`, iterate orders, sum `quantity` (and optional line revenue from `price` × quantity or `discount_allocations`) grouped by `name` or `sku`, then sort descending.
- Use `get_shopify_data` with auto_paginate true for broad questions unless you intentionally need one page.

When calling `get_shopify_data`, pass a JSON string in the `spec` field, for example:
{"endpoint":"orders","query_params":{"status":"any","created_at_min":"2025-04-01T00:00:00Z"},"auto_paginate":true}
"""


def _format_history_block(history: list[dict[str, str]] | None) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for turn in history[-12:]:
        role = (turn.get("role") or "").strip()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    if not lines:
        return ""
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"


def _tool_map(tools: list[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t in tools:
        name = getattr(t, "name", None)
        if isinstance(name, str) and name:
            out[name] = t
    return out


def _text_from_message_content(content: Any) -> str:
    """Normalize LangChain / Gemini message payloads to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    parts.append(str(block.get("text", "")))
                # Gemini reasoning / thought summaries — skip; only visible text counts.
                elif btype == "thinking":
                    pass
                elif btype == "code_execution_result":
                    parts.append(str(block.get("code_execution_result", "")))
                elif "text" in block:
                    parts.append(str(block["text"]))
            else:
                t = getattr(block, "text", None)
                if isinstance(t, str):
                    parts.append(t)
                else:
                    parts.append(str(block))
        return "".join(parts).strip()
    return str(content).strip()


def _is_forbidden_shopify_tool_result(tool_name: str, observation: str) -> bool:
    """Unsafe Shopify / REPL patterns must surface only the assignment-mandated sentence."""
    if tool_name == "python_repl" and observation.strip() == FORBIDDEN_SHOPIFY_OPERATION:
        return True
    if tool_name != "get_shopify_data":
        return False
    try:
        obj = json.loads(observation)
    except json.JSONDecodeError:
        return False
    if not isinstance(obj, dict):
        return False
    if obj.get("detail") == FORBIDDEN_SHOPIFY_OPERATION:
        return True
    err = obj.get("error")
    return err in ("method_not_allowed", "forbidden_query_param")


def _conversation_has_tool_results(messages: list[Any]) -> bool:
    return any(isinstance(m, ToolMessage) for m in messages)


def _failure_from_tool_messages(messages: list[Any]) -> str:
    """Turn recent tool errors into a user-visible diagnostic."""
    details: list[str] = []
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            continue
        body = (m.content or "").strip()
        if not body:
            continue
        if body.startswith("Tool error:"):
            details.append(body[:500])
            continue
        if body.startswith("{") and body.endswith("}"):
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict):
                err = obj.get("error")
                detail = obj.get("detail")
                if err:
                    details.append(f"{err}: {detail}" if detail else str(err))
        if len(details) >= 2:
            break
    if not details:
        return (
            "The model did not return a visible answer after several tool calls. "
            "Check backend logs for tool-call results and confirm Shopify/Gemini credentials."
        )
    joined = " | ".join(d.replace("\n", " ")[:300] for d in details)
    return (
        "I could not produce a final answer because tool calls failed. "
        f"Last tool errors: {joined}"
    )


def _force_final_reply(
    llm: ChatGoogleGenerativeAI,
    messages: list[Any],
    user_question: str,
) -> str:
    """Last resort: plain LLM turn with tool JSON inlined (no tools), so a stuck loop can still answer."""
    chunks: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            body = (m.content or "")[:25_000]
            chunks.append(f"--- tool {m.tool_call_id} ---\n{body}")
    tool_blob = "\n".join(chunks)[:80_000]
    if not tool_blob.strip():
        return ""
    sys = SystemMessage(
        content=(
            "You are finishing a Shopify analyst reply. Use ONLY numbers present in the tool excerpts below. "
            "Write markdown with ## Summary, ## Key metrics, ## Detail (GFM table if helpful), "
            "## Recommendations matching the count the merchant asked for when specified (else two bullets). "
            "If they asked for a chart and the excerpts support it, ## Chart may use "
            "![description](data:image/png;base64,...) from matplotlib; if you cannot emit valid base64, "
            "say so briefly and rely on the table. No fenced code blocks, no invented metrics."
        )
    )
    human = HumanMessage(content=f"Merchant question:\n{user_question}\n\nTool excerpts:\n{tool_blob}")
    out = llm.invoke([sys, human])
    return _text_from_message_content(out.content)


def _tool_call_parts(tc: Any) -> tuple[str | None, dict[str, Any], str]:
    """Support both dict tool_calls and LangChain ToolCall objects."""
    if isinstance(tc, dict):
        name = tc.get("name")
        args = tc.get("args")
        tid = str(tc.get("id") or "tool")
        if not isinstance(args, dict):
            args = {}
        return (name if isinstance(name, str) else None, args, tid)
    name = getattr(tc, "name", None)
    args = getattr(tc, "args", None)
    tid = getattr(tc, "id", None)
    if not isinstance(args, dict):
        args = {}
    return (
        name if isinstance(name, str) else None,
        args,
        str(tid) if tid is not None else "tool",
    )


def _synthetic_tool_result(name: str, args: dict[str, Any]) -> str | None:
    """
    Handle pseudo-tool calls some models emit (utcnow/timedelta) so the loop can recover.
    """
    if name == "utcnow":
        now = datetime.now(timezone.utc).replace(microsecond=0)
        return json.dumps({"utcnow": now.isoformat().replace("+00:00", "Z")})
    if name == "timedelta":
        days = float(args.get("days", 0) or 0)
        hours = float(args.get("hours", 0) or 0)
        minutes = float(args.get("minutes", 0) or 0)
        seconds = float(args.get("seconds", 0) or 0)
        delta = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        total_seconds = int(delta.total_seconds())
        return json.dumps({"timedelta_seconds": total_seconds})
    return None


def _derive_orders_time_window(user_message: str) -> tuple[str | None, str | None]:
    """Best-effort UTC window extraction for common merchant time phrases."""
    q = (user_message or "").lower()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    if "last 7 days" in q or "past 7 days" in q:
        start = now - timedelta(days=7)
        return (start.isoformat().replace("+00:00", "Z"), now.isoformat().replace("+00:00", "Z"))
    if "this month" in q or "current month" in q:
        start = now.replace(day=1, hour=0, minute=0, second=0)
        return (start.isoformat().replace("+00:00", "Z"), now.isoformat().replace("+00:00", "Z"))
    if "last month" in q:
        this_month_start = now.replace(day=1, hour=0, minute=0, second=0)
        prev_month_end = this_month_start - timedelta(seconds=1)
        prev_month_start = prev_month_end.replace(day=1, hour=0, minute=0, second=0)
        return (
            prev_month_start.isoformat().replace("+00:00", "Z"),
            this_month_start.isoformat().replace("+00:00", "Z"),
        )
    return (None, None)


def run_agent_turn(
    *,
    user_message: str,
    history: list[dict[str, str]] | None,
    shop_host: str,
    access_token: str,
    settings: Settings | None = None,
) -> str:
    settings = settings or get_settings()
    logger.info("agent_turn_start question=%r history_turns=%d", user_message[:200], len(history or []))
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    # Disable thinking for gemini-2.5-* models: thinking tokens can produce empty-content
    # responses in the ReAct loop when the model exhausts its budget on reasoning.
    _llm_kwargs: dict[str, Any] = {
        "model": settings.gemini_model,
        "google_api_key": settings.gemini_api_key,
        "temperature": 0.2,
    }
    if settings.gemini_model.lower().startswith("gemini-2.5"):
        _llm_kwargs["thinking_budget"] = 0
    llm = ChatGoogleGenerativeAI(**_llm_kwargs)

    tools = [
        build_shopify_get_tool(settings, shop_host, access_token),
        build_python_analysis_tool(),
    ]
    tool_by_name = _tool_map(tools)
    llm_with_tools = llm.bind_tools(tools)

    history_block = _format_history_block(history)
    user_block = f"{history_block}Current question: {user_message}"

    messages: list[Any] = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=user_block),
    ]

    max_steps = 56
    nudge_attempts = 0
    # Consecutive assistant turns with no tool_calls and no visible text (reset when model asks for tools).
    consecutive_empty = 0
    # Total such empty turns — NOT reset on tool rounds — stops tool/empty/tool/empty from burning the step budget forever.
    total_empty_turns = 0
    forced_final_attempted = False
    auto_prefetch_attempted = False
    for _ in range(max_steps):
        has_tools = _conversation_has_tool_results(messages)
        # After two empty turns in a row while we already have data, bind no tools so the model must write text.
        force_text_only = has_tools and consecutive_empty >= 2
        # First-turn empty: still allow get_shopify_data (no ToolMessages yet).
        if not has_tools:
            force_text_only = False
        chat_model = llm if force_text_only else llm_with_tools
        ai_msg: AIMessage = chat_model.invoke(messages)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if tool_calls:
            logger.info("agent_tool_calls count=%d", len(tool_calls))
        if tool_calls:
            consecutive_empty = 0
        if not tool_calls:
            text = _text_from_message_content(ai_msg.content)
            if text:
                logger.info("agent_turn_success visible_chars=%d", len(text))
                return text
            consecutive_empty += 1
            total_empty_turns += 1
            logger.warning(
                "agent_empty_assistant_turn consecutive_empty=%d total_empty=%d has_tool_results=%s",
                consecutive_empty,
                total_empty_turns,
                has_tools,
            )
            if (not has_tools) and (not auto_prefetch_attempted) and total_empty_turns >= 2:
                auto_prefetch_attempted = True
                start_iso, end_iso = _derive_orders_time_window(user_message)
                query_params: dict[str, Any] = {"status": "any", "limit": "250"}
                if start_iso:
                    query_params["created_at_min"] = start_iso
                if end_iso:
                    query_params["created_at_max"] = end_iso
                spec = {
                    "endpoint": "orders",
                    "query_params": query_params,
                    "auto_paginate": True,
                }
                try:
                    obs = tool_by_name["get_shopify_data"].invoke({"spec": json.dumps(spec)})
                except Exception as e:
                    obs = f"Tool error: auto-prefetch orders failed: {e!s}"
                if not isinstance(obs, str):
                    obs = str(obs)
                logger.warning(
                    "agent_auto_prefetch_orders used start=%r end=%r chars=%d",
                    start_iso,
                    end_iso,
                    len(obs),
                )
                messages.append(ToolMessage(content=obs, tool_call_id="auto_prefetch_orders"))
                continue
            if (
                not forced_final_attempted
                and total_empty_turns >= 1
                and has_tools
            ):
                forced_final_attempted = True
                forced = _force_final_reply(llm, messages, user_message)
                if forced.strip():
                    logger.warning("agent_forced_final_reply_used chars=%d", len(forced))
                    return forced
            # Empty assistant turn: nudge toward tools if nothing fetched yet, else toward a final reply.
            if nudge_attempts < 8:
                nudge_attempts += 1
                if has_tools:
                    nudge_body = (
                        "You must reply now in plain language for the merchant (no more tool calls). "
                        "Use the tool results already in this thread. Include Summary, Key metrics, "
                        "and exactly the number of recommendations the user asked for (default two). "
                        "If they asked for a graph, add one matplotlib chart as markdown "
                        "![description](data:image/png;base64,...) with no code fences. "
                        "If tools failed, say what failed in one short paragraph."
                    )
                else:
                    nudge_body = (
                        "Your last reply had no user-visible text. Continue: call get_shopify_data "
                        "for the orders the user asked about (created_at_min / created_at_max in ISO-8601 UTC), "
                        "use python_repl for tables and matplotlib charts (PNG base64 in markdown image syntax), "
                        "then answer with Summary, Key metrics, Detail, Recommendations, and optional Chart image."
                    )
                messages.append(HumanMessage(content=nudge_body))
                continue
            fallback = _failure_from_tool_messages(messages)
            logger.error("agent_turn_failed_fallback=%r", fallback)
            return fallback

        for tc in tool_calls:
            name, args, tid = _tool_call_parts(tc)
            if not name or name not in tool_by_name:
                if name:
                    synthetic = _synthetic_tool_result(name, args)
                    if synthetic is not None:
                        logger.info("agent_synthetic_tool_result tool=%s", name)
                        messages.append(ToolMessage(content=synthetic, tool_call_id=tid))
                        continue
                logger.warning("agent_unknown_tool name=%r tool_call_id=%s", name, tid)
                messages.append(
                    ToolMessage(
                        content=(
                            f"Unknown tool: {name!r}. Available tools: get_shopify_data, python_repl. "
                            "Use python_repl for date math (utcnow/timedelta are not tools)."
                        ),
                        tool_call_id=tid,
                    )
                )
                continue
            tool = tool_by_name[name]
            try:
                observation = tool.invoke(args)
            except Exception as e:
                observation = f"Tool error: {e!s}"
            logger.info(
                "agent_tool_result tool=%s chars=%d",
                name,
                len(observation) if isinstance(observation, str) else len(str(observation)),
            )
            if not isinstance(observation, str):
                observation = str(observation)
            if _is_forbidden_shopify_tool_result(name, observation):
                logger.warning("agent_forbidden_operation_detected tool=%s", name)
                return FORBIDDEN_SHOPIFY_OPERATION
            if len(observation) > 120_000:
                observation = observation[:120_000] + "\n...[truncated]"
            messages.append(
                ToolMessage(
                    content=observation,
                    tool_call_id=tid,
                )
            )

    logger.error("agent_step_limit_reached max_steps=%d", max_steps)
    return "I stopped after too many tool steps. Please narrow the question (for example, a shorter date range)."
