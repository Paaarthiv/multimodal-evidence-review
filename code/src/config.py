"""Central configuration: paths, model ids, pricing, runtime knobs.

All secrets are read from environment variables only (loaded from code/.env).
Nothing here should ever contain a real key.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This file lives at <repo>/code/src/config.py
CODE_DIR = Path(__file__).resolve().parents[1]          # <repo>/code
REPO_ROOT = CODE_DIR.parent                              # <repo>
DATASET_DIR = REPO_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"

SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
TEST_CLAIMS_CSV = DATASET_DIR / "claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQ_CSV = DATASET_DIR / "evidence_requirements.csv"

PROMPTS_DIR = CODE_DIR / "prompts"
CACHE_DIR = CODE_DIR / ".cache"          # on-disk response cache (avoids repeat calls)
EVAL_DIR = CODE_DIR / "evaluation"

# Load .env from the code/ folder
load_dotenv(CODE_DIR / ".env")

# ---------------------------------------------------------------------------
# Provider / model config
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Groq (free tier: 1000 req/day, no credit card; OpenAI-compatible endpoint).
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")  # 27B multimodal, not deprecated
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
# Proactive pause after each Groq call to stay under the free 6k-tokens/min limit.
GROQ_MIN_INTERVAL = float(os.getenv("GROQ_MIN_INTERVAL", "8"))

# Ollama (local, free, no billing) — primary for this deployment.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "600"))  # local inference can be slow
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "550"))  # output token cap (keeps gen short)
# Context window must fit multiple images (each ~1.5-2k tokens). Default 4096 is too
# small for multi-image rows and causes HTTP 400; 8192 fits up to 3 images at 768px.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

# Primary -> fallback order. Entries may be "provider" or "provider:model".
PROVIDER_ORDER = [p.strip() for p in os.getenv(
    "PROVIDER_ORDER",
    "groq:qwen/qwen3.6-27b,ollama:qwen2.5vl:7b",
).split(",") if p.strip()]

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
USE_MOCK = os.getenv("USE_MOCK", "0") == "1"

# Retry / backoff
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "2.0"))  # seconds, exponential
# Whole-row retry passes for rows that failed for infra reasons (quota/overload).
ROW_RETRY_PASSES = int(os.getenv("ROW_RETRY_PASSES", "1"))
ROW_RETRY_DELAY = float(os.getenv("ROW_RETRY_DELAY", "3.0"))    # seconds between passes

# ---------------------------------------------------------------------------
# Pricing assumptions (USD per 1M tokens) — used ONLY for the cost report.
# These are documented assumptions in evaluation_report.md, not live lookups.
# ---------------------------------------------------------------------------
PRICING = {
    "qwen/qwen3.6-27b": {"input": 0.0, "output": 0.0},   # Groq free tier
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
}

# Image downscale target (longest edge). Smaller = cheaper/faster, still legible.
# 512 keeps major damage legible while keeping multi-image CPU inference tractable
# (vision-token count scales with resolution; 512 ~halves it vs 768).
IMAGE_MAX_EDGE = int(os.getenv("IMAGE_MAX_EDGE", "512"))
