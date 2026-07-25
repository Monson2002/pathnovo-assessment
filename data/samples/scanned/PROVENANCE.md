# Scanned Sample Provenance

`pid_a_scanned.pdf` was synthesized to give `ScannedPDFAdapter` a real, non-mocked
test fixture (the assignment's own FAQ suggests this exact approach: "print-and-scan
(or photograph) a page").

## Method

1. Source: page 1 of `data/samples/pair_02/pid_a_rev0.pdf` (itself a byte-identical
   copy of the assignment's provided `pair_01` sample — see
   `data/samples/pair_02/PROVENANCE.md`).
2. Rendered to a 300 DPI raster image with PyMuPDF (`page.get_pixmap()`), simulating
   a scan/photograph of the physical drawing.
3. Wrote a **new** single-page PDF containing only that raster image — no text layer,
   no vector graphics. This is what makes it a genuine scanned-document fixture:
   `NativePDFAdapter.can_handle()` correctly returns `False` (no extractable text),
   so ingestion falls through to `ScannedPDFAdapter`'s Tesseract OCR path.

Script used is inline in the git history of the commit that added this file (a short
PyMuPDF snippet — render + re-insert as a flat image, no redaction/editing involved,
unlike `scripts/synthesize_pair_02.py`).

## Validation

Verified end-to-end via `tests/test_ingest.py::test_scanned_pdf_adapter_dispatch_and_ocr`:
- `NativePDFAdapter.can_handle()` → `False`
- `ScannedPDFAdapter.can_handle()` → `True`
- Full `detect_and_ingest()` dispatch → `metadata.format == "scanned_pdf"`, with 1340
  OCR'd text blocks recovered from the rasterized page, each with a valid normalized
  bounding box.
