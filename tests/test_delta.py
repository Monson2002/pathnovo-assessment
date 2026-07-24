import pytest
from src.canonical.model import (
    BoundingBox,
    CanonicalDocument,
    DocumentMetadata,
    Page,
    TextBlock,
)
from src.delta import (
    ChangeType,
    DeltaEngine,
    DeltaItem,
    DeltaResult,
    align_blocks,
    bbox_iou,
    generate_json_report,
    generate_markdown_report,
    text_similarity,
)


def test_bbox_iou_calculation():
    box1 = BoundingBox(x0=0.0, y0=0.0, x1=0.5, y1=0.5)
    box2 = BoundingBox(x0=0.25, y0=0.25, x1=0.75, y1=0.75)
    assert bbox_iou(box1, box2) == pytest.approx(1.0 / 7.0)

    # Non-overlapping boxes
    box3 = BoundingBox(x0=0.6, y0=0.6, x1=1.0, y1=1.0)
    assert bbox_iou(box1, box3) == 0.0


def test_text_similarity_matching():
    assert text_similarity("Export Gas Compressor", "export gas compressor") == 1.0
    assert text_similarity("Pump P-101", "Pump P-102") > 0.7
    assert text_similarity("Valve HV-001", "Compressor C-101") < 0.4


def test_align_blocks_greedy_matching():
    blocks_a = [
        TextBlock(
            content="Export Gas Compressor",
            bbox=BoundingBox(x0=0.1, y0=0.1, x1=0.4, y1=0.2),
        ),
        TextBlock(content="P-101A", bbox=BoundingBox(x0=0.5, y0=0.5, x1=0.6, y1=0.6)),
    ]
    blocks_b = [
        TextBlock(
            content="Lift Gas Compressor",
            bbox=BoundingBox(x0=0.1, y0=0.1, x1=0.4, y1=0.2),
        ),  # modified
        TextBlock(
            content="P-101A", bbox=BoundingBox(x0=0.5, y0=0.5, x1=0.6, y1=0.6)
        ),  # identical
        TextBlock(
            content="New Sensor FT-200",
            bbox=BoundingBox(x0=0.7, y0=0.7, x1=0.9, y1=0.8),
        ),  # addition
    ]

    matched, unmatched_a, unmatched_b = align_blocks(blocks_a, blocks_b)

    assert len(matched) == 2
    assert len(unmatched_a) == 0
    assert len(unmatched_b) == 1
    assert unmatched_b[0].content == "New Sensor FT-200"


def test_delta_engine_computation():
    doc_a = CanonicalDocument(
        metadata=DocumentMetadata(
            pid="PID_A", filename="pid_a.pdf", format="native_pdf", page_count=1
        ),
        pages=[
            Page(
                page_number=1,
                width=100.0,
                height=100.0,
                text_blocks=[
                    TextBlock(
                        content="Compressor C-101",
                        bbox=BoundingBox(x0=0.1, y0=0.1, x1=0.3, y1=0.2),
                    ),
                    TextBlock(
                        content="Old Valve HV-001",
                        bbox=BoundingBox(x0=0.4, y0=0.4, x1=0.6, y1=0.5),
                    ),
                    TextBlock(
                        content="Deprecated Note",
                        bbox=BoundingBox(x0=0.8, y0=0.8, x1=0.95, y1=0.9),
                    ),
                ],
            )
        ],
    )

    doc_b = CanonicalDocument(
        metadata=DocumentMetadata(
            pid="PID_B", filename="pid_b.pdf", format="native_pdf", page_count=1
        ),
        pages=[
            Page(
                page_number=1,
                width=100.0,
                height=100.0,
                text_blocks=[
                    TextBlock(
                        content="Compressor C-101",
                        bbox=BoundingBox(x0=0.1, y0=0.1, x1=0.3, y1=0.2),
                    ),  # unchanged
                    TextBlock(
                        content="New Valve HV-002",
                        bbox=BoundingBox(x0=0.4, y0=0.4, x1=0.6, y1=0.5),
                    ),  # modified
                    TextBlock(
                        content="DN100 Line",
                        bbox=BoundingBox(x0=0.7, y0=0.7, x1=0.9, y1=0.8),
                    ),  # added
                ],
            )
        ],
    )

    engine = DeltaEngine(confidence_threshold=0.0)
    result = engine.compute_delta(doc_a, doc_b)

    assert isinstance(result, DeltaResult)
    assert result.pid_a == "PID_A"
    assert result.pid_b == "PID_B"
    assert result.summary["modified"] == 1
    assert result.summary["added"] == 1
    assert result.summary["removed"] == 1
    assert result.summary["total_changes"] == 3


def test_report_generation():
    result = DeltaResult(
        pid_a="PID_A",
        pid_b="PID_B",
        items=[
            DeltaItem(
                change_type=ChangeType.MODIFIED,
                item_type="label",
                page=1,
                location=BoundingBox(x0=0.1, y0=0.1, x1=0.3, y1=0.2),
                description="Modified label: P-101 -> P-102",
                old_value="P-101",
                new_value="P-102",
                confidence=0.95,
            )
        ],
        summary={"added": 0, "removed": 0, "modified": 1, "total_changes": 1},
    )

    md = generate_markdown_report(result)
    assert "# Document Delta Report: PID_A vs PID_B" in md
    assert "[Delta Report, Item #1]" in md
    assert "Modified label: P-101 -> P-102" in md

    json_str = generate_json_report(result)
    assert '"pid_a": "PID_A"' in json_str
    assert '"change_type": "modified"' in json_str
