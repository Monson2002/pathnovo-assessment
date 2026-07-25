# `pair_01` Provenance

## Source

- `Export Gas Compressor-P&ID (1).pdf` and `Lift Gas compressor-P&ID.pdf` were
  provided as the assignment's sample data. They were **not** generated or edited by
  this project.

## Important caveat: these are not sequential revisions

Unlike a true "Rev A → Rev B" pair, these two PDFs are **two distinct P&ID sheets**
for different equipment ("Export Gas Compressor" vs. "Lift Gas Compressor") that
happen to be structurally similar (same drawing template, same tag-numbering
convention). Comparing them produces a large, real structural delta — hundreds of
distinct tag/number differences (e.g. `26-PIT-9087` vs `26-PIT-9077`,
`26-000001-001` vs `26-000006-001`) — rather than the small, localized set of edits
a genuine revision pair would show.

This was discovered during review, not by design, and is documented candidly here
and in `eval/datasets/pair_01_expected.json`'s `dataset_notes` rather than hidden.
It's also why `eval/datasets/pair_01_expected.json`'s ground truth is a
hand-verified **sample** of real, checkable differences (extracted directly from
`pdftotext -layout` output of both PDFs), not an exhaustive labeling — and why
precision against that sample reads low even though the underlying alignment logic
is correct (recall against the labeled sample is 1.00; see `README.md`'s
"Evaluation Harness & Results").

## Why it's still useful

`pair_01` is a good stress test for the delta engine at real-world scale (~900+
detected changes across two similar-but-different documents) and for chat retrieval
across a large delta report. It is a poor fixture for precision benchmarking against
a small ground-truth sample, which is exactly what `data/samples/pair_02/` (a true
synthetic revision pair with exact, complete ground truth) was built to fix — see
`data/samples/pair_02/PROVENANCE.md`.
