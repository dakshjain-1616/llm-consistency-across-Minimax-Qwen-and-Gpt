"""OpenRouter API client for querying multiple LLMs."""

import os
import random
import time
import requests
from typing import Optional

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY_SECONDS", "2.0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Model identifiers on OpenRouter — updated to latest available versions (2026)
MODELS = {
    "qwen":    os.getenv("MODEL_QWEN",     "qwen/qwen3.5-397b-a17b"),
    "minimax": os.getenv("MODEL_MINIMAX",  "minimax/minimax-m2.7"),
    "gpt":     os.getenv("MODEL_GPT",      "openai/gpt-5.4-20260305"),
}

# HTTP status codes that indicate a rate-limit; back off more aggressively
_RATE_LIMIT_CODES = {429, 529}


def _classify_error(exc: Exception) -> str:
    """Return a human-readable error category for better diagnostics."""
    if isinstance(exc, requests.exceptions.Timeout):
        return "timeout"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "connection_error"
    if isinstance(exc, requests.exceptions.HTTPError):
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
        if code in (401, 403):
            return "auth_error"
        if code in _RATE_LIMIT_CODES:
            return "rate_limit"
        if code and code >= 500:
            return "server_error"
        return f"http_{code}"
    return "unknown"


def query_model(model_id: str, prompt: str, session: Optional[requests.Session] = None) -> str:
    """Send a single prompt to a model via OpenRouter and return the response text."""
    api_key = os.getenv("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
    base_url = os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("APP_REFERER", "https://github.com/consistency-heatmap"),
        "X-Title": os.getenv("APP_TITLE", "Consistency Heatmap"),
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(os.getenv("MAX_TOKENS", str(MAX_TOKENS))),
        "temperature": float(os.getenv("TEMPERATURE", str(TEMPERATURE))),
    }

    requester = session or requests
    max_retries = int(os.getenv("MAX_RETRIES", str(MAX_RETRIES)))
    retry_delay = float(os.getenv("RETRY_DELAY_SECONDS", str(RETRY_DELAY)))

    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(1, max_retries + 1):
        try:
            resp = requester.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=int(os.getenv("REQUEST_TIMEOUT_SECONDS", str(REQUEST_TIMEOUT))),
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code in (401, 403):
                raise RuntimeError(
                    f"Authentication error for model '{model_id}': check OPENROUTER_API_KEY."
                ) from exc
            if code not in _RATE_LIMIT_CODES and code and code < 500:
                raise RuntimeError(
                    f"Non-retryable HTTP {code} for model '{model_id}': {exc}"
                ) from exc
            # Rate-limit or 5xx: apply extra back-off
            backoff = retry_delay * attempt * (2.0 if code in _RATE_LIMIT_CODES else 1.0)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            backoff = retry_delay * attempt
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected response shape from model '{model_id}': {exc}"
            ) from exc

        if attempt < max_retries:
            # Exponential back-off with ±25 % jitter
            jitter = random.uniform(-0.25 * backoff, 0.25 * backoff)
            time.sleep(max(0.1, backoff + jitter))

    category = _classify_error(last_exc)
    raise RuntimeError(
        f"Failed to query model '{model_id}' after {max_retries} attempts "
        f"({category}): {last_exc}"
    ) from last_exc
