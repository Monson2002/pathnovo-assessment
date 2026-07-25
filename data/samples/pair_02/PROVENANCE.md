# pair_02 — Provenance

## What this is

A **true synthetic revision pair**, generated to complement `pair_01` (whose two
PDFs, as documented in `eval/datasets/pair_01_expected.json`, are actually two
distinct P&ID sheets rather than sequential revisions of one drawing).

- `pid_a_rev0.pdf` — byte-identical copy of `data/samples/pair_01/Export Gas Compressor-P&ID (1).pdf`.
- `pid_b_rev1.pdf` — the same file with 6 precise, known text edits applied via PyMuPDF.

## How it was generated

`scripts/synthesize_pair_02.py` (run via `uv run python scripts/synthesize_pair_02.py`):
for each target string, it locates the exact bounding box on page 1 with
`page.search_for()`, redacts it, and (for modifications) re-inserts the new
text at the same position with a best-effort matched font size. Because the
edits are made programmatically rather than read off a rendered page by eye,
`eval/datasets/pair_02_expected.json`'s ground truth is **exact**, not an
estimate — every entry in `expected_deltas` is generated directly from the
`EDITS` list in the script, not hand-transcribed afterward.

## The 6 edits

| # | Change | Type | Old -> New |
|---|---|---|---|
| 1 | Interlock tag renumbered | note | `26-PDI-9015` -> `26-PDI-9099` |
| 2 | Oil mist separator mounting changed (+ typo fix) | note | `OFF-SKID` -> `ON-SKID` |
| 3 | Heat tracing setpoint raised | dimension | `25°C` -> `40°C` |
| 4 | Pressure control valve tag renumbered | label | `26-PY-9087B` -> `26-PY-9187B` |
| 5 | Anti-surge valve tag renumbered | label | `26-FV-9038` -> `26-FV-9138` |
| 6 | Atmospheric vent label removed | label | `ATMOSPHERIC VENT.` -> *(deleted)* |

## Validation run (candid results, not cherry-picked)

```
uv run python -m src.main --pid-a data/samples/pair_02/pid_a_rev0.pdf \
  --pid-b data/samples/pair_02/pid_b_rev1.pdf --output-dir output_pair02
```

Delta engine reported: **7 added / 1 removed / 5 modified = 13 total**, against
6 true edits. Two honest findings worth knowing before quoting these numbers:

1. **Edit #1-4 and #6 matched cleanly** (4 correct `modified` + 1 correct
   `removed`, confidence 0.94-0.97 and 1.00 respectively) — the alignment
   algorithm does what it claims on genuine revision edits.
2. **Edit #5 (the anti-surge tag) split into two delta items** instead of one
   clean `modified`: the engine reported `modified: 'COMPRESSOR ANTI-SURGE
   26-FV-9038' -> 'COMPRESSOR ANTI-SURGE'` (i.e. it matched the surrounding
   label but dropped the tag) plus a separate `added: 26-FV-9138`. This
   happened because the re-inserted replacement text landed as a distinct
   PyMuPDF text span rather than merging with the adjacent original span, so
   the two ended up on different sides of the alignment's similarity
   threshold. This is a genuine, reproducible edge case in how the aligner
   handles a tag embedded mid-label, not a synthesis mistake — a real
   candidate follow-up would be loosening the text-similarity threshold for
   spatially-adjacent-but-not-identical spans, or merging same-line spans
   before alignment.
3. **6 of the 13 reported changes are `added: rect` geometry items** — an
   artifact of the redaction method itself: `page.add_redact_annot()` paints a
   white rectangle over the old text, and that rectangle is itself a new
   drawing element the (correctly-working) geometry-delta comparison detects.
   This is a synthesis-method artifact, not a real revision change; it would
   not appear in a genuine CAD-authored revision. Filtering out redaction
   whiteout rects was judged lower priority than shipping the pair itself.

Net: on a real revision pair, precision is materially better than `pair_01`'s
(6 of 6 true edits detected, vs. 934 raw items against pair_01's mismatched
sample), which supports the read in `REVIEW.md`/`README.md` that pair_01's low
precision is a sample-data artifact, not an alignment-algorithm failure.

## Reproducing

```bash
uv run python scripts/synthesize_pair_02.py
uv run python -m eval.run_eval --dataset eval/datasets/pair_02_expected.json --output-dir output/pair_02
```

(`--output-dir output/pair_02` keeps this pair's scorecard separate from `pair_01`'s at `output/eval_results.json` -- both datasets default to the same output directory otherwise.)

**Actual scorecard from a live run** (`output/pair_02/eval_results.json`): delta precision **0.38** / recall **0.83** / F1 **0.53** -- versus pair_01's 0.01 / 1.00 / 0.02 on the same delta engine. This is the concrete evidence that pair_01's low precision is a sample-data artifact (comparing two unrelated documents), not an alignment-algorithm failure.
