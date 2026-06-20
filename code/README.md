# Multi-Modal Evidence Review

A system that verifies damage claims (car / laptop / package) by reviewing the
**submitted images as the primary source of truth**, using the claim conversation
to decide *what to check*, user history for *risk context only*, and a minimum
evidence checklist to decide whether the images are *sufficient*.

For each input claim it produces one `output.csv` row with the exact 14-column
schema from `problem_statement.md`.

---

## TL;DR — how to run

```bash
# 1. install
cd code
python -m venv .venv && .venv/Scripts/activate     # (Windows)  or  source .venv/bin/activate
pip install -r requirements.txt

# 2. add keys  (NEVER commit this file)
cp .env.example .env        # then edit: OPENAI_API_KEY=..., GEMINI_API_KEY=...

# 3. evaluate on the labelled sample (compares OpenAI vs Gemini)
python evaluation/main.py

# 4. produce final predictions for the test set
python main.py              # dataset/claims.csv -> ../output.csv

# offline plumbing check (no keys, no API calls):
USE_MOCK=1 python main.py --limit 5
```

---

## Architecture

One **vision call per claim** (all of a case's images batched into a single call),
wrapped in deterministic Python. This keeps cost, latency and request volume low
while keeping the *policy* auditable and reproducible.

```
claims.csv ─► parse claim text ─► select evidence standard (evidence_requirements.csv)
                                        │
images ────────────────► [VISION CALL] OpenAI (primary) ─fallback─► Gemini
   (downscaled, 1 call)        returns strict structured JSON per image + aggregate
                                        │
user_history.csv ─► history risk ─► [FUSION + SCHEMA GUARD] ─► output.csv
injection guard ──────────────────────┘  (deterministic policy + enum clamping)
```

**Why this shape**
- *Images first*: the model is instructed to decide from pixels, not words.
- *Conversation = what to check*: the claim (often Hindi / Spanish / Chinese /
  code-switched) is extracted inside the same call.
- *History never overrides visuals*: history only **adds** `user_history_risk` /
  `manual_review_required`; it can't flip a clear visual verdict (enforced in
  `src/fusion.py`).
- *Untrusted text*: any "approve this / ignore previous instructions / the note
  says approve" steering — in the chat **or inside an image** — is flagged
  `text_instruction_present` and never changes the decision (defense-in-depth:
  model-side instruction + a deterministic detector in `src/injection_guard.py`).

## Decision policy (deterministic, validated on the 20 gold rows)

- `evidence_standard_met` = `true` unless `claim_status == not_enough_information`.
- `not_enough_information` ⇒ `severity = unknown`, `issue_type = unknown`,
  `supporting_image_ids = none`.
- `claim_status`:
  - **supported** — an image clearly shows the claimed issue on the claimed part.
  - **contradicted** — claimed part visible but no/such damage, milder than claimed
    (`claim_mismatch`), or a different object (`wrong_object`).
  - **not_enough_information** — claimed part not visible / blurry / cropped / contents unseen.
- Whenever `user_history_risk` applies, `manual_review_required` is added too.
- Every emitted value is clamped to the allowed enum set in `src/schema.py`
  (object_part is validated per `claim_object`), so a hallucinated label can never
  reach `output.csv`.

## Reliability & cost controls

- **Provider failover**: OpenAI → Gemini on hard errors (configurable via `PROVIDER_ORDER`).
- **Retry**: exponential backoff on transient/429/5xx.
- **Bounded concurrency** (`MAX_CONCURRENCY`) to respect RPM/TPM limits.
- **On-disk cache** keyed by (provider, model, prompt, image bytes): re-runs and
  eval iterations never repay for identical calls.
- **Image downscaling** to ≤1024px longest edge to cut image tokens & latency.

## File map

```
code/
├── main.py                  # entry point: claims.csv -> output.csv
├── evaluation/
│   ├── main.py              # entry point: score on sample + compare providers
│   ├── report.py            # renders evaluation_report.md
│   └── evaluation_report.md # metrics + operational analysis (generated)
├── prompts/system.txt       # the reviewer system prompt
├── src/
│   ├── config.py            # paths, models, pricing, knobs (keys via env only)
│   ├── schema.py            # allowed enums, ordering, clamping, CSV column order
│   ├── vision_schema.py     # structured-output JSON schema + per-claim prompt
│   ├── providers.py         # OpenAI + Gemini + mock, caching, retry, failover
│   ├── pipeline.py          # per-claim orchestration + batch runner
│   ├── fusion.py            # decision policy + schema guard
│   ├── injection_guard.py   # deterministic prompt-injection detector
│   ├── evidence.py          # evidence_requirements.csv lookup
│   ├── history.py           # user_history.csv risk context
│   ├── metrics.py           # per-column scoring vs gold
│   ├── cost.py              # token -> USD estimate
│   ├── io_utils.py          # CSV / image-path helpers
│   └── mock_vision.py       # offline stub (no API) for plumbing tests
└── requirements.txt
```

## Providers

The vision layer is provider-agnostic and tries `PROVIDER_ORDER` entries
(`provider` or `provider:model`) left-to-right, failing over on hard errors:

- **`ollama` (local, free, default primary)** — runs an open-source vision model
  (`qwen2.5vl:7b`) on your machine via the Ollama API. No API key, no billing,
  no rate limits. Requires `ollama serve` running and `ollama pull qwen2.5vl:7b`.
  Chosen here because paid cloud billing was unavailable; Qwen2.5-VL is also strong
  at OCR, which strengthens in-image instruction-text detection.
- **`gemini`** — `google-genai` SDK. (Free tier is only 20 req/day/model.)
- **`openai`** — `gpt-4o` via OpenAI SDK with strict structured outputs.

## Configuration (env vars, read from `code/.env`)

| Var | Default | Meaning |
|---|---|---|
| `PROVIDER_ORDER` | `ollama:qwen2.5vl:7b,gemini:gemini-2.5-flash,openai:gpt-4o` | primary → fallback |
| `OLLAMA_HOST` | `http://localhost:11434` | local Ollama server |
| `OLLAMA_MODEL` | `qwen2.5vl:7b` | local vision model |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | – / `gpt-4o` | OpenAI key + model |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | – / `gemini-2.5-flash` | Gemini key + model |
| `MAX_CONCURRENCY` | `1` (local) | parallel claims |
| `USE_MOCK` | `0` | `1` = offline stub, no model |
| `IMAGE_MAX_EDGE` | `1024` | image downscale target |

## Notes for evaluation integrity

- No hardcoded per-case answers. `sample_claims.csv` is used only to score and to
  iterate the prompt/policy — never as a lookup table.
- The system reads the provided CSVs and local images and writes the exact schema.
