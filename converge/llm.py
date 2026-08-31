"""Optional LLM re-ranking stage.

The agent is complete and competitive with **no** model call: the default
configuration costs $0.00, makes zero network requests, and answers in single-
digit milliseconds. This module exists for the case where a team wants an LLM
to break ties inside an already-small candidate set -- the one place where a
language model adds signal that lexical and structural evidence cannot.

Enable with environment variables (never commit a key):

.. code-block:: bash

    export CONVERGE_LLM=1
    export CONVERGE_LLM_MODEL=gpt-4o-mini          # any OpenAI-compatible model
    export CONVERGE_LLM_BASE_URL=https://api.openai.com/v1
    export CONVERGE_LLM_API_KEY=sk-...             # or OPENAI_API_KEY

Every failure mode -- missing key, timeout, malformed JSON, HTTP error -- falls
back to the deterministic ordering. The model can improve the slate; it can
never break the run.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o-mini"
_TIMEOUT_SECONDS = float(os.environ.get("CONVERGE_LLM_TIMEOUT", "8"))


@dataclass
class LLMResult:
    order: list[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0


def enabled() -> bool:
    return os.environ.get("CONVERGE_LLM", "").strip().lower() in {"1", "true", "yes", "on"}


def _api_key() -> str | None:
    return os.environ.get("CONVERGE_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")


def _prompt(context: dict, candidates: list[dict]) -> list[dict]:
    lines = [
        f"{position}. {item['parent_asin']} | {item['title'][:140]}"
        for position, item in enumerate(candidates, start=1)
    ]
    user = (
        "A shopper is looking for: "
        + (context.get("category") or "a clothing item")
        + ".\nStated requirements: "
        + ("; ".join(context.get("constraints") or []) or "none yet")
        + ".\nPrior preferences: "
        + (", ".join(context.get("preference_tags") or []) or "none")
        + ".\n\nCandidates:\n"
        + "\n".join(lines)
        + "\n\nReturn JSON only: {\"order\": [asin, ...]} ranking every candidate "
        "from most to least likely to be the exact product this shopper buys."
    )
    return [
        {"role": "system", "content": "You rank e-commerce products. Reply with JSON only."},
        {"role": "user", "content": user},
    ]


def rerank(context: dict, candidates: list[dict]) -> LLMResult | None:
    """Ask the model to re-order a shortlist. Returns ``None`` on any problem."""
    if not enabled() or len(candidates) < 2:
        return None
    key = _api_key()
    if not key:
        return None

    base_url = os.environ.get("CONVERGE_LLM_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
    payload = {
        "model": os.environ.get("CONVERGE_LLM_MODEL", _DEFAULT_MODEL),
        "messages": _prompt(context, candidates),
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
        order = json.loads(content).get("order") or []
        valid = {item["parent_asin"] for item in candidates}
        ordered = [str(asin) for asin in order if str(asin) in valid]
        if not ordered:
            return None
        return LLMResult(
            order=ordered,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, TypeError, OSError):
        return None
