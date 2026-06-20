"""Vision provider layer: model-level failover, caching, retry, usage tracking.

PROVIDER_ORDER is a list of "provider" or "provider:model" entries, tried in
order. This gives both cross-provider failover (gemini -> openai) and
cross-model failover (gemini-2.5-flash -> gemini-2.5-flash-lite). Quota/billing
errors are treated as NON-transient so we fail over immediately instead of
burning the retry budget; 503/overload/rate-limit errors are retried with
exponential backoff.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

from PIL import Image

from . import config


@dataclass
class ModelResult:
    data: dict
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    error: Optional[str] = None


@dataclass
class CallStats:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    images: int = 0
    by_provider: Dict[str, int] = field(default_factory=dict)
    by_model: Dict[str, int] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def record(self, r: ModelResult, n_images: int):
        # Thread-safe: run_file may call this from multiple worker threads.
        with self._lock:
            self.images += n_images
            if r.cached:
                self.cache_hits += 1
                return
            if r.provider == "none":
                return
            self.calls += 1
            self.input_tokens += r.input_tokens
            self.output_tokens += r.output_tokens
            self.by_provider[r.provider] = self.by_provider.get(r.provider, 0) + 1
            self.by_model[r.model] = self.by_model.get(r.model, 0) + 1


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------
def encode_image(path: Path) -> bytes:
    """Open, downscale longest edge, re-encode as JPEG bytes."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        m = config.IMAGE_MAX_EDGE
        if max(w, h) > m:
            scale = m / float(max(w, h))
            im = im.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


# Lean output contract for open models (Ollama / Groq): ask ONLY for the fields we use
# and keep text short, so the JSON always finishes within the token budget (no truncation
# -> no parse failures) and generation stays fast.
OUTPUT_CONTRACT = (
    "\n\nRespond with ONE compact JSON object and NOTHING else. Use EXACTLY these keys:\n"
    '{"claim_status": "<supported|contradicted|not_enough_information>",\n'
    ' "issue_type": "<allowed issue token>",\n'
    ' "object_part": "<allowed part token>",\n'
    ' "severity": "<none|low|medium|high|unknown>",\n'
    ' "valid_image": <true|false>,\n'
    ' "supporting_image_ids": ["img_x", ...],\n'
    ' "risk_flags": ["<flag>", ...],\n'
    ' "model_detected_instruction_text": <true|false>,\n'
    ' "evidence_reason": "<max 15 words>",\n'
    ' "justification": "<max 25 words, mention image ids>"}\n'
    "Do not add any other keys, comments, or prose outside the JSON."
)


# ---------------------------------------------------------------------------
# Cache (keyed by provider, model, prompts, image bytes)
# ---------------------------------------------------------------------------
def _cache_key(provider: str, model: str, system: str, user: str, images: List[bytes]) -> str:
    h = hashlib.sha256()
    for part in [provider, model, system, user]:
        h.update(part.encode("utf-8"))
    for b in images:
        h.update(hashlib.sha256(b).digest())
    return h.hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    p = config.CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _cache_put(key: str, payload: dict) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (config.CACHE_DIR / f"{key}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Provider calls (each takes an explicit model id)
# ---------------------------------------------------------------------------
def _openai_schema() -> dict:
    from .vision_schema import RESPONSE_SCHEMA
    return RESPONSE_SCHEMA


def _call_openai(system: str, user: str, images: List[bytes], model: str) -> ModelResult:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    content = [{"type": "text", "text": user}]
    for b in images:
        content.append({"type": "image_url", "image_url": {"url": _data_url(b)}})

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "evidence_review", "schema": _openai_schema(), "strict": True},
        },
    )
    data = json.loads(resp.choices[0].message.content)
    return ModelResult(data=data, provider="openai", model=model,
                       input_tokens=resp.usage.prompt_tokens,
                       output_tokens=resp.usage.completion_tokens)


def _call_gemini(system: str, user: str, images: List[bytes], model: str) -> ModelResult:
    # New unified SDK (`google-genai`); the legacy `google-generativeai` is EOL.
    from google import genai
    from google.genai import types
    from .vision_schema import RESPONSE_SCHEMA

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    user_full = (
        user
        + "\n\nReturn ONLY a JSON object matching exactly these keys:\n"
        + json.dumps(list(RESPONSE_SCHEMA["properties"].keys()))
    )
    contents = [user_full]
    for b in images:
        contents.append(types.Part.from_bytes(data=b, mime_type="image/jpeg"))

    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system, temperature=0, response_mime_type="application/json"),
    )
    data = json.loads(resp.text)
    um = getattr(resp, "usage_metadata", None)
    return ModelResult(data=data, provider="gemini", model=model,
                       input_tokens=(getattr(um, "prompt_token_count", 0) or 0) if um else 0,
                       output_tokens=(getattr(um, "candidates_token_count", 0) or 0) if um else 0)


def _extract_json(text: str) -> dict:
    """Parse JSON tolerantly: strip code fences, else take the outermost {...}.

    A non-JSON reply raises a ValueError tagged 'transient_json' so analyze()
    retries the SAME local model rather than failing over.
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:]
        t = t.strip().rstrip("`").strip()
    try:
        return json.loads(t)
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                pass
    raise ValueError("transient_json: model did not return valid JSON")


def _call_ollama(system: str, user: str, images: List[bytes], model: str) -> ModelResult:
    """Local vision model via the Ollama native chat API (free, no billing).

    Images go in the message's `images` array as raw base64. We use plain JSON
    mode (`format="json"`) rather than full JSON-schema grammar constraint:
    grammar-constrained decoding of our nested schema is extremely slow on CPU.
    The prompt already enumerates the exact allowed tokens and we clamp every
    value downstream, so JSON mode is both fast and safe.
    """
    b64 = [base64.b64encode(b).decode("ascii") for b in images]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user + OUTPUT_CONTRACT, "images": b64},
        ],
        "stream": False,
        "format": "json",
        "keep_alive": "30m",   # keep the model resident so back-to-back calls don't reload
        "options": {
            "temperature": 0,
            "num_predict": config.OLLAMA_NUM_PREDICT,
            "num_ctx": config.OLLAMA_NUM_CTX,   # must fit multiple images
        },
    }
    req = urllib.request.Request(
        f"{config.OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=config.OLLAMA_TIMEOUT) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    content = body.get("message", {}).get("content", "")
    data = _extract_json(content)   # tolerant of fences/prose around the JSON
    return ModelResult(
        data=data, provider="ollama", model=model,
        input_tokens=body.get("prompt_eval_count", 0) or 0,
        output_tokens=body.get("eval_count", 0) or 0,
    )


def _call_groq(system: str, user: str, images: List[bytes], model: str) -> ModelResult:
    """Groq cloud vision via its OpenAI-compatible endpoint (free tier, fast).

    Same wire format as OpenAI (image_url data URIs) but JSON-object mode + the
    compact contract, since these open models don't support strict json_schema.
    """
    from openai import OpenAI

    # max_retries lets the SDK itself honor Groq's Retry-After header on 429s
    # (the correct way to ride out the per-minute token limit).
    client = OpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL,
                    max_retries=8, timeout=240.0)
    content = [{"type": "text", "text": user + OUTPUT_CONTRACT}]
    for b in images:
        content.append({"type": "image_url", "image_url": {"url": _data_url(b)}})
    # qwen3.x is a reasoning model; reasoning_effort="none" disables <think> tokens so
    # JSON-object mode validates cleanly and stays fast/cheap.
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=1200,
        reasoning_effort="none",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
    )
    data = _extract_json(resp.choices[0].message.content)
    result = ModelResult(data=data, provider="groq", model=model,
                         input_tokens=resp.usage.prompt_tokens,
                         output_tokens=resp.usage.completion_tokens)
    # Proactive throttle so the NEXT call doesn't breach the per-minute token cap.
    if config.GROQ_MIN_INTERVAL > 0:
        time.sleep(config.GROQ_MIN_INTERVAL)
    return result


def _call_mock(system: str, user: str, images: List[bytes], model: str) -> ModelResult:
    from .mock_vision import mock_response
    return ModelResult(data=mock_response(user, len(images)), provider="mock", model="mock")


_PROVIDERS = {"openai": _call_openai, "gemini": _call_gemini, "groq": _call_groq,
              "ollama": _call_ollama, "mock": _call_mock}
_DEFAULT_MODEL = {"openai": lambda: config.OPENAI_MODEL, "gemini": lambda: config.GEMINI_MODEL,
                  "groq": lambda: config.GROQ_MODEL,
                  "ollama": lambda: config.OLLAMA_MODEL, "mock": lambda: "mock"}

# Hard failures: pointless to retry, fail over to the next entry immediately.
#  - per-DAY quota cap (free tier), zero quota, no billing, bad key, missing model.
_FATAL = ("perdayperproject", "perdayper", "limit: 0", "insufficient_quota",
          "api key not valid", "invalid_api_key", "permission_denied",
          "not_found", "404")
# Retryable: per-minute rate limits, transient server errors, malformed local JSON.
_TRANSIENT = ("perminute", "rate", "timeout", "overload", "unavailable",
              "resource_exhausted", "429", "500", "502", "503", "529",
              "transient_json")


def _classify(err: Exception) -> str:
    s = str(err).lower().replace(" ", "")
    if any(t in s for t in _FATAL):
        return "fatal"      # skip retries, fail over to next entry
    if any(t in s for t in _TRANSIENT):
        return "transient"  # retry with backoff
    return "fatal"


def _parse_entry(entry: str) -> Tuple[str, str]:
    if ":" in entry:
        prov, model = entry.split(":", 1)
        return prov.strip(), model.strip()
    prov = entry.strip()
    return prov, _DEFAULT_MODEL.get(prov, lambda: prov)()


def analyze(system: str, user: str, image_paths: List[Path]) -> ModelResult:
    """Structured vision call with caching, retry and provider/model failover."""
    try:
        images = [encode_image(p) for p in image_paths]
    except Exception as e:  # corrupt/unreadable image must not crash the whole run
        return ModelResult(data={}, provider="none", model="none",
                           error=f"image_decode_failed: {e}")
    order = ["mock"] if config.USE_MOCK else list(config.PROVIDER_ORDER)

    last_err = None
    for entry in order:
        provider, model = _parse_entry(entry)
        fn = _PROVIDERS.get(provider)
        if not fn:
            continue

        key = _cache_key(provider, model, system, user, images)
        cached = _cache_get(key)
        if cached is not None:
            return ModelResult(data=cached["data"], provider=provider, model=model,
                               input_tokens=cached.get("input_tokens", 0),
                               output_tokens=cached.get("output_tokens", 0), cached=True)

        for attempt in range(config.MAX_RETRIES):
            try:
                r = fn(system, user, images, model)
                _cache_put(key, {"data": r.data, "input_tokens": r.input_tokens,
                                 "output_tokens": r.output_tokens})
                return r
            except Exception as e:  # noqa: BLE001
                last_err = e
                if _classify(e) == "transient" and attempt < config.MAX_RETRIES - 1:
                    time.sleep(config.RETRY_BASE_DELAY * (2 ** attempt))
                    continue
                break  # fatal or out of retries -> next entry (failover)

    return ModelResult(data={}, provider="none", model="none", error=str(last_err))
