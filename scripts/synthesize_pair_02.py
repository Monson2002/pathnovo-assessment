"""One-off generator for data/samples/pair_02: a TRUE synthetic revision pair.

Unlike pair_01 (two distinct P&ID sheets, documented as such), pair_02 is
built by taking a single real source PDF and applying a small set of known,
precise text edits with PyMuPDF to produce a "Rev B". Because we make the
edits ourselves, the resulting ground truth is exact (not estimated from
reading pdftotext output), which is what pair_01's ground truth structurally
can't be. See data/samples/pair_02/PROVENANCE.md for the write-up.

Run with: uv run python scripts/synthesize_pair_02.py
"""

import json
import os
import shutil

import fitz  # PyMuPDF

SRC_PDF = "data/samples/pair_01/Export Gas Compressor-P&ID (1).pdf"
OUT_DIR = "data/samples/pair_02"
PID_A = os.path.join(OUT_DIR, "pid_a_rev0.pdf")
PID_B = os.path.join(OUT_DIR, "pid_b_rev1.pdf")

# (old_text, new_text_or_None, change_type, item_type, description)
# new_text=None means the text is deleted outright (a "removed" change).
EDITS = [
    (
        "26-PDI-9015 HH INITIATE PRESSURIZED COMPRESSOR STOP.",
        "26-PDI-9099 HH INITIATE PRESSURIZED COMPRESSOR STOP.",
        "modified",
        "note",
        "Interlock instrument tag renumbered from 26-PDI-9015 to 26-PDI-9099",
    ),
    (
        "LUBE OIL RESERVOIR VENT. OIL MIST SEPERATOR INSTALLED OFF-SKID.",
        "LUBE OIL RESERVOIR VENT. OIL MIST SEPARATOR INSTALLED ON-SKID.",
        "modified",
        "note",
        "Oil mist separator mounting changed from OFF-SKID to ON-SKID (also fixes SEPERATOR typo)",
    ),
    (
        "SAFETY CRITICAL HEAT TRACING - HYDRATE MITIGATION (25°C).",
        "SAFETY CRITICAL HEAT TRACING - HYDRATE MITIGATION (40°C).",
        "modified",
        "dimension",
        "Heat tracing setpoint raised from 25°C to 40°C",
    ),
    (
        "26-PY-9087B",
        "26-PY-9187B",
        "modified",
        "label",
        "Pressure control valve tag renumbered from 26-PY-9087B to 26-PY-9187B",
    ),
    (
        "26-FV-9038",
        "26-FV-9138",
        "modified",
        "label",
        "Anti-surge valve tag renumbered from 26-FV-9038 to 26-FV-9138",
    ),
    (
        "ATMOSPHERIC VENT.",
        None,
        "removed",
        "label",
        "Atmospheric vent label removed",
    ),
]


def find_span_font(page, rect):
    """Best-effort lookup of the font name/size of the text occupying `rect`,
    so the replacement text visually matches the original."""
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                span_rect = fitz.Rect(span["bbox"])
                if span_rect.intersects(rect):
                    return (
                        span.get("font", "helv"),
                        span.get("size", 5.0),
                        span.get("color", 0),
                    )
    return "helv", 5.0, 0


def apply_edit(doc, old_text, new_text):
    """Redact `old_text` on page 0 and, if `new_text` is given, insert it in
    the same place with a best-effort font/size match. Returns the list of
    rects that were actually edited (empty if no match found)."""
    page = doc[0]
    rects = page.search_for(old_text)
    if not rects:
        return []

    edited = []
    for rect in rects:
        font, size, color = find_span_font(page, rect)
        page.add_redact_annot(rect, fill=(1, 1, 1))
        edited.append((rect, font, size, color))
    page.apply_redactions()

    if new_text is not None:
        for rect, font, size, color in edited:
            rgb = (
                ((color >> 16) & 255) / 255,
                ((color >> 8) & 255) / 255,
                (color & 255) / 255,
            )
            page.insert_text(
                (rect.x0, rect.y1 - 1.0),
                new_text,
                fontsize=size,
                fontname="helv",
                color=rgb,
            )
    return [r for r, *_ in edited]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Rev A: byte-identical baseline copy of the source PDF.
    shutil.copyfile(SRC_PDF, PID_A)

    # Rev B: apply the known edits to a fresh copy.
    doc = fitz.open(SRC_PDF)
    applied = []
    for old_text, new_text, change_type, item_type, description in EDITS:
        rects = apply_edit(doc, old_text, new_text)
        status = "OK" if rects else "SKIPPED (no match found)"
        print(f"[{status}] {change_type}/{item_type}: {description}")
        if rects:
            applied.append(
                {
                    "change_type": change_type,
                    "item_type": item_type,
                    "page": 1,
                    "description": description,
                    "old_value": old_text,
                    "new_value": new_text,
                }
            )
    doc.save(PID_B, garbage=4, deflate=True)
    doc.close()

    dataset = {
        "pair_id": "pair_02",
        "pid_a": PID_A,
        "pid_b": PID_B,
        "dataset_notes": [
            "pair_02 is a TRUE synthetic revision pair: pid_b is generated from pid_a by "
            "applying the exact text edits below via PyMuPDF (scripts/synthesize_pair_02.py). "
            "Because the edits are made programmatically, this ground truth is exact, not "
            "estimated -- complementing pair_01, whose two PDFs are genuinely different "
            "documents (see pair_01_expected.json's dataset_notes) and therefore can only "
            "support a hand-verified sample ground truth.",
        ],
        "expected_deltas": applied,
        "qa_pairs": [
            {
                "question": "What was the interlock instrument tag changed from and to?",
                "expected_answer_contains": ["26-PDI-9015", "26-PDI-9099"],
                "expected_sources": ["delta_report"],
            },
            {
                "question": "What changed about the heat tracing setpoint?",
                "expected_answer_contains": ["25", "40"],
                "expected_sources": ["delta_report"],
            },
        ],
    }
    out_path = "eval/datasets/pair_02_expected.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nWrote {len(applied)}/{len(EDITS)} verified edits to {out_path}")
    print(f"Rev A: {PID_A}\nRev B: {PID_B}")


if __name__ == "__main__":
    main()
