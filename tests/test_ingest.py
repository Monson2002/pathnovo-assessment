import os
import pytest
from src.canonical.model import BoundingBox, CanonicalDocument
from src.ingest import DWGAdapter, NativePDFAdapter, detect_and_ingest

SAMPLE_PDF_1 = "data/samples/pair_01/Export Gas Compressor-P&ID (1).pdf"
SAMPLE_PDF_2 = "data/samples/pair_01/Lift Gas compressor-P&ID.pdf"


def test_bounding_box_properties():
    bbox = BoundingBox(x0=0.1, y0=0.2, x1=0.5, y1=0.8)
    assert bbox.width == pytest.approx(0.4)
    assert bbox.height == pytest.approx(0.6)
    assert bbox.center == (pytest.approx(0.3), pytest.approx(0.5))


def test_native_pdf_adapter():
    if not os.path.exists(SAMPLE_PDF_1):
        pytest.skip("Sample PDF 1 not found")

    adapter = NativePDFAdapter()
    assert adapter.can_handle(SAMPLE_PDF_1) is True

    doc = adapter.ingest(SAMPLE_PDF_1, pid="PID_TEST_01")
    assert isinstance(doc, CanonicalDocument)
    assert doc.metadata.pid == "PID_TEST_01"
    assert doc.metadata.format == "native_pdf"
    assert doc.metadata.page_count > 0
    assert len(doc.pages) > 0

    first_page = doc.pages[0]
    assert len(first_page.text_blocks) > 0
    first_block = first_page.text_blocks[0]
    assert first_block.content != ""
    assert 0.0 <= first_block.bbox.x0 <= 1.0


def test_dwg_adapter_stub():
    adapter = DWGAdapter()
    # Should raise NotImplementedError if file exists
    with pytest.raises(NotImplementedError):
        adapter.ingest("data/samples/sample.dwg", pid="PID_DWG")


def test_detect_and_ingest():
    if not os.path.exists(SAMPLE_PDF_2):
        pytest.skip("Sample PDF 2 not found")

    doc = detect_and_ingest(SAMPLE_PDF_2, pid="PID_TEST_02")
    assert doc.metadata.pid == "PID_TEST_02"
    assert doc.metadata.page_count > 0
