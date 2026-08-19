"""
Thin, uniform wrappers around two real LLM providers used as the
buyer-agent / seller-agent "brains" for the LLM-negotiation validation
layer (see llm_negotiation.py). Both wrappers expose the same signature:

    call(system_prompt: str, user_prompt: str) -> str

so the negotiation engine does not need to know which provider is behind
a given agent.

Provider notes (checked live against the provider APIs on 2026-08-19):
  - Gemini: the requested "gemini-3.6-flash-lite" does not exist as a
    model id; the flash-lite tier tops out at "gemini-3.5-flash-lite".
    We use the "gemini-flash-lite-latest" alias so this automatically
    tracks whichever flash-lite model Google promotes to that alias.
  - OpenRouter: "openrouter/free" auto-routes across whatever free-tier
    model is available; individual ":free" models are shared-pool and
    frequently 429 under load, so calls rotate through a short list of
    known-working free models with retry/backoff.

Keys are read from environment variables (populated from .env via
python-dotenv) and are never logged or embedded in URLs.
"""

import os
import re
import json
import time

import itertools

import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

# Multiple OpenRouter keys (OPENROUTER_API_KEY, OPENROUTER_API_KEY_2, ...)
# are rotated round-robin across calls so the experiment can draw on each
# key's separate daily free-tier quota instead of exhausting just one.
_OPENROUTER_KEYS = [os.environ[k] for k in sorted(os.environ)
                     if k.startswith("OPENROUTER_API_KEY") and os.environ[k]]
if not _OPENROUTER_KEYS:
    _OPENROUTER_KEYS = [None]
_openrouter_key_cycle = itertools.cycle(_OPENROUTER_KEYS)

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# "openrouter/free" is OpenRouter's own auto-router across whatever free
# model is currently available, so it's tried first; the named models
# behind it are an explicit fallback list for when auto-routing itself
# errors, ordered by observed reliability from a manual smoke test.
OPENROUTER_FREE_MODELS = [
    "openrouter/free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openai/gpt-oss-20b:free",  # reasoning model: burns max_tokens on hidden
                                # chain-of-thought before emitting JSON, so it
                                # is tried last, not first
]


class LLMCallError(RuntimeError):
    pass


def _candidate_json_spans(text):
    """Yield each balanced {...} substring starting at each '{' in text,
    tracking brace depth (ignoring braces inside quoted strings) so a
    malformed model reply that wraps the real object in a spurious extra
    key (observed in the wild: '{"action":{"action": ...}}') still yields
    the well-formed inner object as one candidate, tried before the
    (invalid) outer one."""
    n = len(text)
    starts = [i for i, c in enumerate(text) if c == "{"]
    for start in reversed(starts):  # innermost/latest-starting first
        depth = 0
        in_string = False
        escape = False
        for i in range(start, n):
            c = text[i]
            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:i + 1]
                    break


def _extract_first_json(text):
    """Parse the first candidate JSON object that actually parses cleanly."""
    last_err = None
    tried = False
    for candidate in _candidate_json_spans(text):
        tried = True
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            last_err = e
    if not tried:
        raise LLMCallError(f"no JSON object found in model output: {text[:200]!r}")
    raise LLMCallError(f"malformed JSON in model output: {last_err}; text={text[:300]!r}")


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30),
       retry=retry_if_exception_type((requests.RequestException, LLMCallError)))
def call_gemini(system_prompt, user_prompt, temperature=0.9, max_output_tokens=400):
    if not GEMINI_API_KEY:
        raise LLMCallError("GEMINI_API_KEY not set")
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    r = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=45)
    if r.status_code != 200:
        raise LLMCallError(f"gemini http {r.status_code}: {r.text[:300]}")
    data = r.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError) as e:
        raise LLMCallError(f"gemini unexpected response shape: {data}") from e
    if not text.strip():
        raise LLMCallError(f"gemini returned empty text (finishReason={data['candidates'][0].get('finishReason')})")
    return text


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30),
       retry=retry_if_exception_type((requests.RequestException, LLMCallError)))
def _call_openrouter_model(model, system_prompt, user_prompt, temperature, max_tokens):
    key = next(_openrouter_key_cycle)
    if not key:
        raise LLMCallError("no OPENROUTER_API_KEY(_N) set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                       headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise LLMCallError(f"openrouter[{model}] http {r.status_code}: {r.text[:300]}")
    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise LLMCallError(f"openrouter[{model}] unexpected response shape: {data}") from e
    if not text or not text.strip():
        raise LLMCallError(f"openrouter[{model}] returned empty content")
    return text


def call_openrouter(system_prompt, user_prompt, temperature=0.7, max_tokens=1500):
    """Try each free model in OPENROUTER_FREE_MODELS in turn (each with its
    own retry/backoff); only raise once every model has failed."""
    last_err = None
    for model in OPENROUTER_FREE_MODELS:
        try:
            return _call_openrouter_model(model, system_prompt, user_prompt, temperature, max_tokens)
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise LLMCallError(f"all OpenRouter free models failed; last error: {last_err}")


def get_client(provider):
    if provider == "gemini":
        return call_gemini
    if provider == "openrouter":
        return call_openrouter
    raise ValueError(f"unknown provider {provider!r}")


def call_and_parse_json(provider, system_prompt, user_prompt, **kwargs):
    """Call a provider and parse the first JSON object in its reply,
    retrying once with a stricter reminder if parsing fails."""
    client = get_client(provider)
    text = client(system_prompt, user_prompt, **kwargs)
    try:
        return _extract_first_json(text), text
    except LLMCallError:
        stricter = user_prompt + "\n\nREMINDER: reply with ONLY the JSON object, no other text."
        text2 = client(system_prompt, stricter, **kwargs)
        return _extract_first_json(text2), text2
