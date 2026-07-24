# Comprehensive Repo Review — Document Delta & Grounded Chat

> Reviewed against `2026-07-23-applied-ai-engineer-delta-chat-assignment.html` and the **internal hiring rubric** (included in the assignment HTML).

---

## How To Run (Current State)

### Prerequisites

```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler
```

### Setup

```bash
# Clone and enter the repo
cd pathnovo-assessment

# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
uv sync

# Configure API key
cp .env.example .env
# Edit .env and set: NVIDIA_API_KEY=your-key-here
```

### Run Commands

```bash
# Run delta computation (produces delta report)
uv run python -m src.main \
  --pid-a "data/samples/pair_01/Export Gas Compressor-P&ID (1).pdf" \
  --pid-b "data/samples/pair_01/Lift Gas compressor-P&ID.pdf"

# Run with interactive chat mode
uv run python -m src.main \
  --pid-a "data/samples/pair_01/Export Gas Compressor-P&ID (1).pdf" \
  --pid-b "data/samples/pair_01/Lift Gas compressor-P&ID.pdf" \
  --chat

# Run tests
uv run pytest tests/ -v

# Run evaluation harness
uv run python -m eval.run_eval
```

### Or via Makefile (after fixing Makefile to use `uv run`)

```bash
make run    # Delta pipeline
make chat   # Delta + interactive chat
make test   # Unit tests
make eval   # Evaluation scorecard
```

---

## Rubric-Mapped Assessment

The assignment includes an **internal rubric scored 1–5 per dimension, weighted to 100**. Here's how the current codebase scores and what to fix.

---

## 1. Pipeline Design (15% weight) — Current: ~3/5

> *"Clean canonical-representation seam; formats plug in behind one interface; abstraction would genuinely survive adding a 4th format."*

### ✅ What's Good
- Clean `FormatAdapter` ABC with `can_handle()` + `ingest()` interface
- `CanonicalDocument` Pydantic model is well-designed and format-agnostic
- Registry with ordered adapter fallback chain
- DWG stub is a real adapter behind the same seam (not hypothetical)
- Two formats working end-to-end (native PDF + scanned PDF via OCR)

### 🔴 Issues to Fix

| Issue | File | Fix |
|---|---|---|
| `can_handle()` opens entire PDF (expensive, file opened twice) | `src/ingest/pdf_native.py` | Cache the `fitz.open()` result or do a quick first-page check |
| `fitz.open()` not in context manager — resource leak on exception | `src/ingest/pdf_native.py` | Use `with fitz.open(file_path) as doc:` or try/finally |
| Drawing extraction silently swallowed (`except Exception: pass`) | `src/ingest/pdf_native.py` L88 | Log the exception, don't silently swallow |
| `drawing.get("type", "path")` — PyMuPDF drawings don't have a `type` key | `src/ingest/pdf_native.py` L79 | Classify from `drawing["items"]` path commands |
| DPI=300 hardcoded in scanned adapter | `src/ingest/pdf_scanned.py` | Make configurable via `settings.ocr_dpi` |
| `convert_from_path()` loads ALL pages into memory at once | `src/ingest/pdf_scanned.py` | Process page-by-page using `first_page`/`last_page` params |
| No BoundingBox validation (accepts any float, docstring says 0-1) | `src/canonical/model.py` | Add `Field(ge=0.0, le=1.0)` validators |
| `block_type`, `element_type`, `format` are raw strings | `src/canonical/model.py` | Use `Literal` or `Enum` types to enforce valid values |

---

## 2. Delta Quality (20% weight — HIGHEST) — Current: ~2.5/5

> *"Real alignment + classification with locations and confidence; handles moved/modified content; sensible on their own eval set."*

### ✅ What's Good
- Fuzzy text matching + spatial IoU alignment is a sound approach
- Greedy bipartite matching with `used_a`/`used_b` sets prevents double-matching
- Item type classification (label, dimension, note, text)
- Confidence scores on every delta item
- Structured `DeltaItem` with change_type, location, old/new values

### 🔴 Critical Issues

| Issue | Impact | Fix |
|---|---|---|
| **Drawing elements completely ignored in delta** | Geometric changes (pipe routing, symbol changes) are invisible. The rubric says *"text, dimension, note, **geometry**, table cell"* | Add `DrawingElement` comparison in `engine.py` — at minimum compare bounding box overlap and element counts per page |
| **O(n²) alignment will be slow on real P&IDs** | P&IDs can have 200+ labels per page. 200×200 = 40K comparisons per page | Document this limitation; consider spatial bucketing or KD-tree for proximity filtering |
| **`assert` used for control flow** | Will silently disappear under `python -O` | Replace `assert` on line 140 of `src/delta/engine.py` with an `if` check |
| **Hardcoded magic numbers in alignment** | `t_sim >= 0.25` and `bbox_iou > 0.5` in `src/delta/align.py` L80 | Extract to named constants or make configurable |
| **Dimension regex too narrow** | Won't match `1'-6"`, `Ø12`, `NPS 4` | Expand regex in `classify_item_type()` |
| **Label regex too narrow** | Won't match `FIC-1001A`, `PSV-100A/B` | Expand regex in `classify_item_type()` |
| **Confidence for page-level additions misleadingly high** | Native PDF text blocks have confidence=1.0 used as-is for "this entire page was added" — that's OCR quality, not structural confidence | Separate OCR confidence from structural confidence |

### 💡 High-Impact Improvement
The rubric says: *"matching is the hard part, not diffing"*. Your alignment strategy is the core differentiator. Consider documenting your alignment approach more explicitly — add comments or a docstring explaining the scoring formula, why you chose greedy over Hungarian, and known failure modes. **This is exactly what the interviewer will ask about.**

---

## 3. Grounded Chat (15% weight) — Current: ~2.5/5

> *"Retrieval spans both PIDs + report; answers cite specific sources; refuses/hedges when unsupported instead of hallucinating."*

### ✅ What's Good
- RAG pipeline: ChromaDB retrieval → system prompt → Nemotron → citation parsing
- System prompt instructs LLM to cite sources as `[PID A, Page X]` or `[Delta Report, Item #N]`
- System prompt instructs refusal when context is insufficient
- Citation parsing with regex

### 🔴 Critical Issues

| Issue | Impact | Fix |
|---|---|---|
| **Fallback answer hardcodes `[PID A, Page 1]`** | When LLM fails, citations are always wrong | Build fallback from actual retrieved chunk metadata |
| **`"dummy_key_for_testing"` silently masks missing API key** | App starts, attempts real API call with invalid key, silently degrades | Fail fast with clear error message, or explicit offline mode flag |
| **Fallback embedding is 128-dim; real model is ~2048-dim** | If mixed in same ChromaDB collection, cosine similarity is meaningless | Match real model dimensions, or enforce all-fallback mode |
| **No LLM telemetry tracked** | `Trace.log_llm_call()` exists but `answer_question()` never calls it | Wire token counts, model, prompt/response into trace |
| **`page: 1` hardcoded for all delta report chunks** | Citation page numbers for delta items will always be wrong | Extract page from delta item metadata during chunking |
| **Embedding calls are sequential, not batched** | Hundreds of API calls for large documents | Batch embeddings using OpenAI batch API |
| **Citation parsing is fragile** | Won't match `[PID A, page 1]` (lowercase) despite `re.IGNORECASE` because the `Page` literal in the pattern requires capital P | Fix regex: `r"\[(PID [AB]),\s*[Pp]age\s+(\d+)\]"` |
| **No conversation memory** | Each question is independent — no multi-turn context | At minimum, document this as a known limitation |

---

## 4. Observability (15% weight) — Current: ~2/5

> *"End-to-end traces, token/cost telemetry, structured logs, inspectable metrics; failures visible, not swallowed."*

### ✅ What's Good
- `Tracer` with nested `Span` context managers and timing
- `Trace` saves to JSON files with stage hierarchy
- Structured JSON logging formatter with correlation ID support
- `log_llm_call()` method captures model, prompt, response, tokens, cost

### 🔴 Critical Issues — This is currently the weakest area

| Issue | Impact | Fix |
|---|---|---|
| **Logger is set up but NEVER USED** | `main.py` calls `setup_logging()` but uses `print()` everywhere. No module in `src/` ever calls `get_logger()` | Replace all `print()` with logger calls throughout `main.py` and key modules |
| **`correlation_id` is never set** | The JSON formatter checks for it, but no code ever sets it — dead feature | Set `correlation_id = trace.trace_id` on log records |
| **LLM telemetry infrastructure built but not wired** | `log_llm_call()` exists on `Trace` but `answer.py` never calls it | Wire into `AnswerEngine.answer_question()` after LLM calls |
| **Chat mode operations not traced** | Only the initial pipeline (ingest → delta → report) is traced; chat Q&A is untraced | Wrap chat loop iterations in `tracer.span("chat_question")` |
| **ALL exception handlers silently swallow errors** | `except Exception: pass` or `except Exception:` with fallback — 8 instances across the codebase | Log every exception with `logger.exception()` or `logger.error()` |
| **No metrics surfaced** | Latency, token cost, delta counts, retrieval hit stats — none are surfaced in any inspectable format | At minimum, add timing summary to trace output and delta counts to report |

> **CAUTION:** The rubric explicitly says: *"failures visible, not swallowed"*. Silent exception swallowing is an **auto-flag** for the hiring panel.

---

## 5. Evaluation Rigor (20% weight — HIGHEST) — Current: ~2/5

> *"Labeled dataset; meaningful delta + groundedness metrics; runnable scorecard; comparable across runs; candid failure reporting."*

### ✅ What's Good
- Eval harness exists and is runnable via `python -m eval.run_eval`
- Delta P/R/F1 metrics implemented with fuzzy matching
- Groundedness and citation accuracy metrics exist
- Scorecard output format is clear

### 🔴 Critical Issues

| Issue | Impact | Fix |
|---|---|---|
| **Only 1 expected delta in ground truth** | P/R/F1 is meaningless with 1 item — precision will be near 0 if many deltas are detected, recall will be 1 if it's detected | **Expand ground truth**: manually label 10-20 real changes between the two P&IDs |
| **No `pair_02` dataset** | Plan mentions it but it doesn't exist; only 1 document pair | Synthesize a second pair (edit a PDF, re-export) for robustness |
| **No unit tests for metric functions** | `eval/metrics.py` has zero dedicated tests | Add `tests/test_eval_metrics.py` |
| **Duplicate `text_similarity` function** | `eval/metrics.py` reimplements it instead of importing from `src.delta.align` | Import from `src.delta.align` |
| **Results not persisted** | Scorecard is printed to stdout only — not regression-comparable across runs | Save to `output/eval_results.json` with timestamp |
| **No candid failure table** | The rubric says *"a candid failure table is a strong positive signal"* and the scorecard template in the plan includes a "Known Failures" section | Add known failure cases with honest analysis |
| **Groundedness metric is simplistic** | Word overlap ≥ 30% with 4+ char words — will false-positive on common engineering vocabulary | Consider n-gram overlap or embedding similarity |
| **`run_eval.py` has no error handling** | Any failure crashes the entire evaluation | Add try/except with meaningful error messages |

> **IMPORTANT:** The rubric auto-flags: *"Eval that can't detect a regression"* and *"'It works' claims with no trace/metric to back them"*. The current eval with 1 ground truth item can't meaningfully detect regressions.

---

## 6. Engineering & Docs (10% weight) — Current: ~1.5/5

> *"Reproducible run, clear README with trade-offs and cuts, tests where they matter, no secrets, readable code."*

### 🔴 Critical Issues

| Issue | Impact | Fix |
|---|---|---|
| **README.md is empty** (just the title) | The rubric says README must cover: *"How to run, key design decisions and trade-offs, what you deliberately cut, your observability + eval approach, and what you'd do next"* | **Write a proper README** — this is mandatory |
| **Root `main.py` is a dead stub** | Confusing — prints "Hello from pathnovo-assessment!" | Delete it or make it a thin wrapper |
| **`pytest` in production dependencies** | Should be in `[dependency-groups] dev` | Move to dev dependencies |
| **`.env.example` only lists 3 of 12+ configurable vars** | Reviewer won't know all the knobs they can turn | Document all env vars with defaults and comments |
| **No `docker-compose.yml`** | Assignment says *"optional but appreciated"* | Consider adding for reproducibility |
| **`pyproject.toml` has placeholder description** | `"Add your description here"` | Update it |
| **Makefile uses `python3` instead of `uv run`** | Won't use the project's venv | Change to `uv run python -m ...` |
| **No DEMO.md or screen recording** | Assignment requires *"a 2–4 minute walkthrough (screen recording or DEMO.md)"* | Create a DEMO.md showing: one delta + one grounded chat exchange + eval scorecard |
| **ScannedPDFAdapter has zero test coverage** | One of only 2 working formats, yet untested | Add tests (even if they need `pytest.mark.skipif` for Tesseract) |
| **Chat tests call real NVIDIA API** | Will fail without API key; no mocking | Mock `get_embedding` and `get_llm_client` in tests |

---

## 7. Communication & Scoping (5% weight) — Current: ~2/5

> *"Deliberate scope cuts explained; demo is crisp; 'what's next' shows product + technical maturity."*

### 🔴 Issues
- No scope cuts documented anywhere
- No "what's next with more time" section
- No DEMO.md or screen recording
- IMPLEMENTATION_PLAN.md exists but is a pre-implementation plan, not a post-implementation reflection

### 💡 Fix
Add these sections to the README:
1. **Scope Cuts**: DWG parsing stubbed (why), no delta markup overlay (why), single-threaded pipeline (why)
2. **What I'd Do Next**: Batch embeddings, drawing element comparison, multi-turn chat, cost/latency budget, proper eval dataset with 50+ labeled items
3. **Trade-offs**: Greedy matching vs. Hungarian algorithm, deterministic alignment vs. LLM-assisted alignment, homegrown tracer vs. OpenTelemetry

---

## Priority-Ordered Action Plan

### 🔴 P0 — Must Fix (blocks rubric pass)

1. **Write a proper README.md** — setup, run instructions, architecture, design decisions, trade-offs, scope cuts, what's next
2. **Expand ground truth** — label 10-20 real deltas in `pair_01_expected.json` and add candid failure table
3. **Wire observability** — replace `print()` with logger, connect `log_llm_call()` in answer.py, set `correlation_id`, log exceptions instead of swallowing
4. **Create DEMO.md** — show one delta + one chat exchange + eval scorecard

### 🟡 P1 — Should Fix (significant rubric impact)

5. **Add drawing element comparison** in delta engine (even basic bbox overlap counts)
6. **Fix fallback embedding dimensions** to match real model, or add explicit offline mode
7. **Fix hardcoded fallback citation** `[PID A, Page 1]` in answer.py
8. **Mock API calls in tests** so `test_chat.py` works without NVIDIA_API_KEY
9. **Move pytest to dev dependencies** in pyproject.toml
10. **Delete root `main.py`** stub
11. **Add `tests/test_eval_metrics.py`** unit tests
12. **Update `.env.example`** with all configurable variables

### 🟠 P2 — Nice to Have (polish)

13. **Fix `fitz.open()` resource leak** — use context manager
14. **Add input validation** in `main.py` (check file exists)
15. **Fix citation regex** to be case-insensitive on "Page"
16. **Add Makefile targets**: `install`, `lint`, `clean`
17. **Synthesize `pair_02`** data for second eval dataset
18. **Add `conftest.py`** with shared test fixtures
19. **Fix Makefile** to use `uv run` instead of bare `python3`

---

## Rubric Score Estimate (Current → With P0 Fixes)

| Dimension | Weight | Current | After P0 | What a 5 needs |
|---|---|---|---|---|
| Pipeline Design | 15% | 3/5 | 3.5/5 | Fix resource mgmt, validate models |
| **Delta Quality** | **20%** | **2.5/5** | **3/5** | Drawing element comparison, richer classification |
| Grounded Chat | 15% | 2.5/5 | 3/5 | Wire telemetry, fix fallbacks |
| Observability | 15% | 2/5 | 3.5/5 | Use logger, wire traces, log exceptions |
| **Eval Rigor** | **20%** | **2/5** | **3.5/5** | Expand ground truth, failure table, persist results |
| Engineering & Docs | 10% | 1.5/5 | 3.5/5 | README, DEMO.md, clean up dead code |
| Communication | 5% | 2/5 | 3.5/5 | Scope cuts, trade-offs, what's next |
| **Weighted Total** | | **~46/100** | **~67/100** | |

> **WARNING:** The two **heaviest dimensions** (Delta Quality 20% + Eval Rigor 20%) are currently the weakest areas. Investing time here has the highest ROI.
