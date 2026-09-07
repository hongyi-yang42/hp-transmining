"""Minimal Anthropic-Messages-compatible HTTP client for the pilot.

Deliberately NOT a multi-provider framework: one endpoint shape (the
Anthropic ``/v1/messages`` API, which the locally configured gateway
speaks), one function, ``httpx`` (already in the venv). Credentials
come from the environment — ``ANTHROPIC_BASE_URL`` (optional),
``ANTHROPIC_AUTH_TOKEN`` (bearer) or ``ANTHROPIC_API_KEY`` (x-api-key).
Nothing is read from disk, nothing is cached, nothing is committed.

Every caller records model name, prompt version, and the inference
parameters that affect reproducibility alongside its outputs.
"""

from __future__ import annotations

import json
import os

DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"


class ModelGatewayError(RuntimeError):
    """The gateway failed or returned an unusable payload."""


def _credentials() -> tuple[str, str]:
    base = (os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if token:
        return base, f"Bearer {token}"
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return base, api_key
    raise ModelGatewayError(
        "no model credential in environment "
        "(set ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY)"
    )


def call_model(
    system: str,
    user: str,
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout_s: float = 120.0,
) -> str:
    """One Messages-API call; returns the concatenated text blocks.

    Raises :class:`ModelGatewayError` on transport, HTTP, or payload
    problems — the error carries the HTTP status and the gateway's error
    field only, never request or response bodies (they may echo corpus
    text).
    """
    import httpx  # lazy: unit tests run without network use

    base, credential = _credentials()
    headers = {
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    if credential.startswith("Bearer "):
        headers["authorization"] = credential
    else:
        headers["x-api-key"] = credential

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(f"{base}/v1/messages", headers=headers, json=payload)
    except httpx.HTTPError as exc:
        raise ModelGatewayError(f"model gateway transport failure: {type(exc).__name__}") from None

    if resp.status_code != 200:
        detail = ""
        try:
            body = resp.json()
            err = body.get("error") or {}
            detail = f"; {err.get('type', '')} {err.get('message', '')[:200]}".rstrip()
        except Exception:
            pass
        raise ModelGatewayError(f"model gateway HTTP {resp.status_code}{detail}")

    try:
        data = resp.json()
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
    except Exception:
        raise ModelGatewayError("model gateway returned an unparseable payload") from None
    if not text.strip():
        raise ModelGatewayError("model gateway returned no text content")
    return text


def parse_json_object(text: str) -> dict | None:
    """Parse a model reply as ONE JSON object; ``None`` when unusable.

    Tolerates a single markdown-fenced block (a common model habit) by
    taking the fence content; anything else — prose wrappers, arrays,
    scalars — is unusable and the caller records an invalid response.
    """
    t = text.strip()
    if t.startswith("```"):
        # ```json ... ``` (or bare ``` ... ```)
        first_newline = t.find("\n")
        if first_newline == -1:
            return None
        t = t[first_newline + 1 :]
        end = t.rfind("```")
        if end != -1:
            t = t[:end]
        t = t.strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


__all__ = [
    "DEFAULT_BASE_URL",
    "ANTHROPIC_VERSION",
    "ModelGatewayError",
    "call_model",
    "parse_json_object",
]
