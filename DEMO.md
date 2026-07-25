# Demo: Document Delta & Grounded Chat

This walks through one delta run and one real grounded-chat exchange, plus the eval scorecard, using the provided sample pair (`data/samples/pair_01/`).

## Setup

```bash
uv sync
cp .env.example .env   # set NVIDIA_API_KEY
```

## 1. Delta run

```bash
make run
# equivalently:
# .venv/bin/python -m src.main \
#   --pid-a "data/samples/pair_01/Export Gas Compressor-P&ID (1).pdf" \
#   --pid-b "data/samples/pair_01/Lift Gas compressor-P&ID.pdf"
```

Actual output from a real run in this repo:

```
--- Document Delta & Grounded Chat ---
Document A: data/samples/pair_01/Export Gas Compressor-P&ID (1).pdf
Document B: data/samples/pair_01/Lift Gas compressor-P&ID.pdf

Ingesting Document A... (loaded from cache)
  1 page(s) processed
Ingesting Document B... (loaded from cache)
  1 page(s) processed

Computing delta engine analysis...
  Delta Summary: {'added': 426, 'removed': 206, 'modified': 302, 'total_changes': 934}
  Saved Delta Reports to: output/
  Saved Visual Redline Markup to: output/annotated_delta.pdf
  Saved Observability Trace: output/traces/trace_5b9669a2-4a78-4386-852e-6a5802daaa0d.json
```

This produces three artifacts:
- `output/delta_report.md` / `output/delta_report.json` -- the human/machine-readable delta report
- `output/annotated_delta.pdf` -- the bonus visual redline markup overlay (bounding boxes + change badges on the changed regions of PID B)
- `output/traces/trace_<id>.json` -- the observability trace for this run, with per-stage timings

Sample of a genuinely detected, high-confidence real change from `output/delta_report.md`:

```
| #3 | `modified` | `text` | Page 1 | Modified text on page 1: '26-PDI-9015 HH INITIATE PRESSURIZED COMPRESSOR STOP.'
   -> '26-PDI-9054 HH INITIATE PRESSURIZED COMPRESSOR STOP.' | 0.98 |
```

(See "Evaluation Harness & Results" in `README.md` for why the *volume* of detected changes on this specific sample pair is large -- the two provided PDFs are two distinct P&ID sheets, not sequential revisions of one drawing.)

## 2. Grounded chat exchange

```bash
make chat
```

Real exchange from `eval/run_eval.py`'s live-mode QA run against this repo (full transcript in `output/eval_results.json`):

> **Q:** What was the relief valve setpoint changed from and to?
>
> **A:** I cannot answer this question based on the provided document context. None of the provided chunks mention a "relief valve setpoint" change.
> Citation: `[Delta Report, Items #227, #359, #441, #224, #300]` (none relevant)

This is the intended behavior when retrieval doesn't surface the right chunk: the model refuses rather than inventing a setpoint value. It's also an honest example of a real limitation -- see "Evaluation Harness & Results" in `README.md` for the retrieval-recall gap this points to.

A second question, where retrieval did surface the right context:

> **Q:** What is the compressor service described on PID A's inlet header, and how does it differ on PID B?
>
> **A:** The compressor service described on PID A's inlet header is EXPORT `[PID A, Page 1]`. The service description for the compressor on PID B differs in that it does not specify "EXPORT" in the retrieved context `[PID B, Page 1]`.

## 3. Eval scorecard

```bash
make eval
```

Actual output of a live run (real LLM + embedding calls, no mocking):

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
║   Groundedness:         0.38                     ║
║   Citation Accuracy:    0.56                     ║
╚══════════════════════════════════════════════════╝
```

See "Evaluation Harness & Results" in `README.md` for an honest read of these numbers -- in short: recall (1.00) is the meaningful delta-quality signal here given the sample-pair mismatch, and the chat numbers reflect real (sometimes appropriately-refusing, sometimes retrieval-limited) model behavior rather than a broken harness.

## What I'd show live in 2-4 minutes

1. `make run` -- point at the delta summary counts and `output/annotated_delta.pdf` opened side-by-side with the source PDF.
2. `make chat` -- ask "what changed on the compressor?" and a specific-tag question, showing a citation and a refusal.
3. `make eval` -- scorecard, then open `output/eval_results.json` to show the QA transcript backing the numbers.
4. `output/traces/trace_<id>.json` -- per-stage timings and (for chat) LLM token/cost telemetry.
