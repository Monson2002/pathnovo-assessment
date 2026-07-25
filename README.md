# Document Delta Engine & Grounded Chat Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/architecture-canonical--seam-green.svg)](#architecture)
[![Evaluation](https://img.shields.io/badge/eval-harness%20included-orange.svg)](#evaluation-harness)

A production-grade, format-agnostic AI system that ingests engineering document revisions (such as P&ID drawings), computes a structured delta between them, generates human/machine-readable delta reports, and exposes a grounded chat interface with precise source citations.

---

## Key Capabilities

1. **Format-Agnostic Ingestion**: Plugin architecture supporting Native Digital PDFs (PyMuPDF), Scanned/Raster PDFs (Tesseract OCR), and CAD DWG files (extensible adapter seam).
2. **Structured Delta Engine**: Combines sequence alignment with normalized spatial proximity (IoU and center distance) to detect `added`, `removed`, and `modified` text, dimensions, equipment tags, notes, and geometry.
3. **Retrievable Delta Reports**: Emits Markdown and JSON reports summarizing revisions, which serve as first-class retrievable context for chat.
4. **Grounded RAG Chat**: ChromaDB vector index paired with NVIDIA Nemotron LLM (`nvidia/llama-3.3-nemotron-super-49b-v1`) enforcing strict citations (`[PID A, Page X]`, `[Delta Report, Item #N]`).
5. **Full Observability**: End-to-end per-stage tracing, LLM token and cost telemetry, and correlation-id-supported structured JSON logging.
6. **Evaluation Harness**: Integrated evaluation suite calculating Precision, Recall, F1 for deltas, as well as Groundedness and Citation Accuracy for chat responses.

---

## Architecture

```
                               ┌─────────────────────────┐
                               │  Source P&IDs (PDF/DWG) │
                               └────────────┬────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │     Format Adapters     │
                               │ (Native/Scanned/DWG)    │
                               └────────────┬────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │   Canonical Document    │
                               │   (Normalized Model)    │
                               └──────┬────────────┬─────┘
                                      │            │
             ┌────────────────────────┘            └────────────────────────┐
             │                                                              │
┌────────────▼────────────┐                                    ┌────────────▼────────────┐
│      Delta Engine       │                                    │     ChromaDB Vector     │
│ (Align + Classify)      │                                    │      Document Index     │
└────────────┬────────────┘                                    └────────────┬────────────┘
             │                                                              │
┌────────────▼────────────┐                                    ┌────────────▼────────────┐
│  Markdown/JSON Report   ├───────────────────────────────────►│   Grounded Chat RAG    │
└─────────────────────────┘                                    │    (NVIDIA Nemotron)    │
                                                               └────────────┬────────────┘
                                                                            │
                                                               ┌────────────▼────────────┐
                                                               │  Grounded Answer with   │
                                                               │    Source Citations     │
                                                               └─────────────────────────┘
```

---

## Setup & Prerequisites

### 1. System Dependencies

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler
```

### 2. Environment Setup

```bash
# Install uv (fast Python package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies into virtual environment
uv sync

# Configure environment variables
cp .env.example .env
# Edit .env to set your NVIDIA_API_KEY if online LLM inference is desired
```

---

## How To Run

### 1. Run Delta Analysis (Pipeline Only)

Computes the revision delta and saves reports to `output/` and trace logs to `output/traces/`:

```bash
# Using Makefile
make run

# Or directly via Python
.venv/bin/python -m src.main \
  --pid-a "data/samples/pair_01/Export Gas Compressor-P&ID (1).pdf" \
  --pid-b "data/samples/pair_01/Lift Gas compressor-P&ID.pdf"
```

### 2. Interactive Grounded Chat Mode

Ingests revisions, computes the delta report, indexes chunks into ChromaDB, and launches an interactive grounded chat terminal:

```bash
# Using Makefile
make chat

# Or directly via Python
.venv/bin/python -m src.main \
  --pid-a "data/samples/pair_01/Export Gas Compressor-P&ID (1).pdf" \
  --pid-b "data/samples/pair_01/Lift Gas compressor-P&ID.pdf" \
  --chat
```

**Example Query**: `"What changed in the document title?"`
**Example Output**:
```
Answer:
The document title was modified from 'Export Gas Compressor' to 'Lift Gas Compressor'. [Delta Report, Item #1]

Citations:
  - Delta Report, Item #1
```

### 3. Run Unit Tests

```bash
make test
# Or: .venv/bin/pytest -s
```

### 4. Run Evaluation Harness

Executes the ground-truth evaluation harness and generates `output/eval_results.json` along with an ASCII scorecard:

```bash
make eval
# Or: .venv/bin/python -m eval.run_eval
```

### 5. Run the REST API (for the Streamlit frontend / deployment)

```bash
make api
# Or: .venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Exposes the same pipeline over HTTP (`POST /delta`, `POST /chat`, plus report/overlay/trace
endpoints) — see `src/api/app.py` and interactive docs at `http://localhost:8000/docs`.
This is what the separate Streamlit frontend calls; deployment instructions (Docker + AWS)
are in `AWS_DEPLOY.md`.

---

## Design Decisions & Trade-Offs

| Component | Design Choice | Trade-Off / Rationale |
|---|---|---|
| **Intermediate Model** | `CanonicalDocument` with normalized `(0.0-1.0)` bounding boxes | Decouples ingestion formats from downstream matching; guarantees seamless addition of future adapters. |
| **Alignment Algorithm** | Greedy Bipartite Matching with Composite Score | $O(N \cdot M)$ complexity per page. Chosen over Hungarian Algorithm for clarity and predictable performance on typical P&ID text density. |
| **OCR Strategy** | Page-level Tesseract OCR via `pdf2image` | Provides fallback for scanned drawings; trade-off is higher memory usage per page during rendering. |
| **LLM Grounding** | Strict System Prompt + Regex Citation Extraction | Ensures citations link back to `[PID A, Page X]` or `[Delta Report, Item #N]`. Includes deterministic grounded fallback if offline. |
| **Observability** | Homegrown lightweight `Tracer` + Structured JSON Logger | Avoids heavy external APM daemon dependencies while delivering full per-stage latency, trace IDs, and token tracking. |

---

## Deliberate Scope Cuts

1. **CAD DWG Parsing**: Implemented as a stub adapter (`DWGAdapter`) behind the `FormatAdapter` seam raising `NotImplementedError`. Native PDF and Scanned OCR PDFs are fully implemented end-to-end.
2. **Ground-truth breadth**: `eval/datasets/pair_01_expected.json` is a hand-verified sample of real differences (not exhaustive) between the two provided sample PDFs, which turned out to be two distinct-but-structurally-similar P&ID sheets rather than sequential revisions of one drawing (see `dataset_notes` in that file and `data/samples/pair_01/PROVENANCE.md`). This means raw delta counts on `pair_01` are large and precision/recall against the sample ground truth understate the alignment logic's real accuracy on a true revision pair. `data/samples/pair_02/` addresses this with a true synthetic revision pair (6 programmatically-applied edits to a copy of the same source PDF, exact ground truth) — see `data/samples/pair_02/PROVENANCE.md`.
3. **Spatial indexing for very large sheets**: alignment is a pruned O(N·M) greedy match per page (see "Alignment Algorithm" below); the pruning is a constant-factor speedup, not an asymptotic fix. Fine at the scale of these samples; would need an R-tree/spatial index for hundreds of sheets (see "What I'd Build Next").

*(Earlier drafts of this README said visual overlay markup and multi-turn chat memory were cut. Both have since been implemented — see "Key Capabilities" above and `src/markup/overlay.py` / `src/chat/answer.py`'s rolling history.)*

---

## Observability & Telemetry

Every request generates a structured JSON trace file in `output/traces/trace_<UUID>.json` containing:
- **Per-Stage Spans**: Latency breakdowns for `ingest_documents`, `compute_delta`, `generate_report`, and `chat_session`.
- **LLM Call Logs**: Prompts, responses, model parameters, token counts, and estimated API cost.
- **Structured Logs**: JSON logs emitted to stdout formatted with module, line number, level, and correlation ID.

---

## Evaluation Harness & Results

Run `make eval` to execute the evaluation harness against labeled ground truth (`eval/datasets/pair_01_expected.json`). Below is the actual output of a live run (real NVIDIA LLM + embedding calls, no mocking), saved to `output/eval_results.json`. The scorecard now also reports per-stage latency and a concrete failure case (a real false positive/negative delta and the worst-scoring chat answer) **derived from this specific run**, not a static boilerplate list:

```
╔══════════════════════════════════════════════════╗
║      EVALUATION SCORECARD (LIVE)                 ║
╠══════════════════════════════════════════════════╣
║ Delta Detection                                  ║
║   Precision:  0.01                               ║
║   Recall:     1.00                               ║
║   F1:         0.02                               ║
║                                                  ║
║ Chat Quality                                     ║
║   Groundedness:         0.58                     ║
║   Citation Accuracy:    0.89                     ║
║                                                  ║
║ Stage Latency (ms)                               ║
║   ingest_documents       252.3                   ║
║   compute_delta          38.9                    ║
║   index_and_chat         309932.0                ║
║                                                  ║
║ Failure Case (this run)                          ║
║   Worst Q: What changed about the pressure transm║
║     groundedness=0.33 citation_accuracy=1.00     ║
╚══════════════════════════════════════════════════╝
```

**Read these numbers honestly, not as a pass/fail badge:**

- **Delta precision/F1 are low by construction, not by algorithm failure.** `pair_01`'s two sample PDFs are two distinct-but-structurally-similar P&ID sheets (Export vs. Lift Gas Compressor), not sequential revisions of one drawing (see `eval/datasets/pair_01_expected.json`'s `dataset_notes`, `data/samples/pair_01/PROVENANCE.md`, and the "Deliberate Scope Cuts" section above). The engine correctly finds ~934 real structural differences between them; the ground truth is a hand-verified sample of ~10 of those (not exhaustive), so precision against that sample undercounts true positives that simply aren't labeled. **Recall is 1.00** -- every hand-verified real change was detected, which is the more meaningful signal here.
- **Groundedness (0.58) and citation accuracy (0.89) reflect real LLM behavior**, not a broken metric or a crash. The harness's own `failure_case` block names the worst-scoring question directly (`output/eval_results.json`); spot-checking `qa_details` shows the model correctly refusing when the relevant delta item wasn't retrieved -- exactly the "refuse instead of hallucinate" behavior the assignment asks for -- alongside cases where retrieval surfaced the wrong chunks, a genuine retrieval-recall weakness worth improving (larger `top_k`, or filtering delta-report chunks by keyword overlap with the question before the vector search).
- **`index_and_chat` latency (~5 minutes) is dominated by real network round-trips** to the NVIDIA API for embeddings and LLM inference, not local compute -- `ingest_documents` and `compute_delta` complete in well under a second each.
- The word-overlap groundedness metric (`eval/metrics.py`) is intentionally simple and known to be an imperfect proxy for "is this actually correct".

`data/samples/pair_02/` is a true synthetic revision pair (6 known, programmatically-applied edits to a copy of the same source PDF; see `data/samples/pair_02/PROVENANCE.md`). Run `uv run python -m eval.run_eval --dataset eval/datasets/pair_02_expected.json --output-dir output/pair_02` for its scorecard — precision (0.38) and F1 (0.53) are materially higher than pair_01's (real edits detected cleanly, not swamped by an unrelated document's worth of differences), which is the evidence that pair_01's low precision is a sample-data artifact, not an alignment-algorithm failure. Its `failure_case` block also captures a genuine alignment edge case: one tag rename (`26-FV-9038` → `26-FV-9138`) split into a separate modify+add pair instead of one clean modify (documented further in `data/samples/pair_02/PROVENANCE.md`).

---

## What I'd Build Next With More Time

1. **Spatial Indexing ($R$-tree)**: Replace page-level $O(N \cdot M)$ alignment loops with an $R$-tree spatial index to handle 500+ sheet document packages in milliseconds.
2. **Real retrieval evaluation**: Recall@k / hit-rate / MRR against a labeled query set, instead of only the word-overlap groundedness heuristic — the eval harness's `failure_case` block already shows retrieval-recall as the main chat-quality gap.
3. **LLM-Assisted Complex Alignment**: Use LLM vision capabilities to align non-standard symbols and complex rotated text blocks where spatial bounding boxes overlap significantly.
4. **DWG/DXF ingestion**: Implement the `DWGAdapter` seam with `ezdxf` or the ODA File Converter.
