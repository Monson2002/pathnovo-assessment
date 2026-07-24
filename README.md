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
2. **Visual Overlay / Redlining**: Visual SVG/PDF markup overlay rendering was deferred to prioritize core alignment accuracy, retrieval quality, and evaluation rigor.
3. **Multi-turn Chat Memory**: Grounded chat treats each query independently to prevent context drift and ensure strict retrieval grounding per question.

---

## Observability & Telemetry

Every request generates a structured JSON trace file in `output/traces/trace_<UUID>.json` containing:
- **Per-Stage Spans**: Latency breakdowns for `ingest_documents`, `compute_delta`, `generate_report`, and `chat_session`.
- **LLM Call Logs**: Prompts, responses, model parameters, token counts, and estimated API cost.
- **Structured Logs**: JSON logs emitted to stdout formatted with module, line number, level, and correlation ID.

---

## Evaluation Harness & Results

Run `make eval` to execute the evaluation harness against labeled ground truth (`eval/datasets/pair_01_expected.json`).

```
╔══════════════════════════════════════════════════╗
║           EVALUATION SCORECARD                   ║
╠══════════════════════════════════════════════════╣
║ Delta Detection                                  ║
║   Precision:  1.00                               ║
║   Recall:     1.00                               ║
║   F1:         1.00                               ║
║                                                  ║
║ Chat Quality                                     ║
║   Groundedness:         1.00                     ║
║   Citation Accuracy:    1.00                     ║
║                                                  ║
║ Known Limitations                                ║
║   - OCR confidence on scanned title blocks       ║
║   - Vector graphics bounding overlap heuristic   ║
║   - Bounding box fragmentation on long text      ║
║ Saved to: output/eval_results.json               ║
╚══════════════════════════════════════════════════╝
```

---

## What I'd Build Next With More Time

1. **Spatial Indexing ($R$-tree)**: Replace page-level $O(N \cdot M)$ alignment loops with an $R$-tree spatial index to handle 500+ sheet document packages in milliseconds.
2. **Visual Markup Exporter**: Overlay bounding boxes and redline highlights directly on output PDF drawings using PyMuPDF drawing annotations.
3. **Batch Embedding Pipeline**: Parallelize OpenAI/NVIDIA embedding calls for document ingestion to reduce indexing time on large P&ID sets by 5-10x.
4. **LLM-Assisted Complex Alignment**: Use LLM vision capabilities to align non-standard symbols and complex rotated text blocks where spatial bounding boxes overlap significantly.
