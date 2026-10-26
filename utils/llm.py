"""
utils/llm.py — Anthropic client factory with an offline stub.

LLM_MODE=stub returns a deterministic stand-in that answers forced tool calls
(`tool_choice={"type": "tool", ...}`) with a payload generated from the tool's
JSON schema, one item per ticker found in the prompt. No network, no API key.
It exists so the full pipeline can be demonstrated offline; its outputs carry
no analytical content.

Any other value (or unset) returns the real `anthropic.Anthropic` client.
"""
from __future__ import annotations

import os
import re
from contextlib import contextmanager
from types import SimpleNamespace


def stub_enabled() -> bool:
    return os.getenv("LLM_MODE", "").strip().lower() == "stub"


def get_client():
    if stub_enabled():
        return StubClient()
    import anthropic
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ── Stub ─────────────────────────────────────────────────────────────────

_TICKER = re.compile(r"(?<![A-Za-z0-9])([A-Z]{1,5}(?:-[A-Z])?)(?=[:\s(,])")


def _seed(text: str) -> int:
    h = 0x811C9DC5
    for ch in text.encode():
        h = ((h ^ ch) * 0x01000193) & 0xFFFFFFFF
    return h


def _tickers_in(prompt: str) -> list[str]:
    import config
    universe = set(config.TICKERS)
    seen: list[str] = []
    for m in _TICKER.finditer(prompt):
        t = m.group(1)
        if t in universe and t not in seen:
            seen.append(t)
    return seen


def _value(schema: dict, name: str, ticker: str):
    kind = schema.get("type")
    h = _seed(f"{name}:{ticker}")
    if "enum" in schema:
        options = schema["enum"]
        if name in ("recommendation", "action"):
            return "BUY" if h % 3 == 0 else "HOLD"
        if name == "conviction":
            return 3 + h % 3
        return options[h % len(options)]
    if kind == "string":
        return ticker if name == "ticker" else f"[stub] {name} for {ticker or 'portfolio'}"
    if kind == "integer":
        return h % 5 + 1
    if kind == "number":
        if name == "size_pct":
            return 0.0
        return round(((h % 2001) - 1000) / 1000, 3)   # [-1, 1]
    if kind == "boolean":
        return bool(h % 2)
    if kind == "array":
        return [_value(schema.get("items", {}), f"{name}{i}", ticker) for i in range(3)]
    if kind == "object":
        return {k: _value(v, k, ticker) for k, v in schema.get("properties", {}).items()}
    return None


def _payload(tool: dict, prompt: str) -> dict:
    schema = tool["input_schema"]
    tickers = _tickers_in(prompt)
    out: dict = {}
    for key, sub in schema.get("properties", {}).items():
        if sub.get("type") == "array" and sub.get("items", {}).get("type") == "object" \
                and "ticker" in sub["items"].get("properties", {}):
            out[key] = [{k: _value(v, k, t) for k, v in sub["items"]["properties"].items()}
                        for t in tickers]
        else:
            out[key] = _value(sub, key, "")
    return out


def _message(tool: dict | None, prompt: str):
    usage = SimpleNamespace(input_tokens=0, output_tokens=0,
                            cache_creation_input_tokens=0, cache_read_input_tokens=0)
    if tool is None:
        block = SimpleNamespace(type="text", text='{"intent": "info", "answer": "[stub] offline demo"}')
        return SimpleNamespace(content=[block], stop_reason="end_turn", usage=usage, model="stub")
    block = SimpleNamespace(type="tool_use", name=tool["name"], id="stub", input=_payload(tool, prompt))
    return SimpleNamespace(content=[block], stop_reason="tool_use", usage=usage, model="stub")


def _prompt_text(messages: list) -> str:
    parts = []
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts.extend(b.get("text", "") for b in c if isinstance(b, dict))
    return "\n".join(parts)


class _Messages:
    def create(self, *, messages=None, tools=None, tool_choice=None, **_):
        tool = None
        if tools and tool_choice and tool_choice.get("type") == "tool":
            tool = next((t for t in tools if t["name"] == tool_choice.get("name")), tools[0])
        return _message(tool, _prompt_text(messages))

    @contextmanager
    def stream(self, **kwargs):
        msg = self.create(**kwargs)
        yield SimpleNamespace(get_final_message=lambda: msg)


class StubClient:
    """Duck-types the subset of `anthropic.Anthropic` the agents use."""

    def __init__(self):
        self.messages = _Messages()
