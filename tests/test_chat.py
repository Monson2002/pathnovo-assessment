from src.canonical.model import (
    BoundingBox,
    CanonicalDocument,
    DocumentMetadata,
    Page,
    TextBlock,
)
from src.chat import (
    AnswerEngine,
    AnswerResult,
    DocumentIndex,
    get_embedding,
)


def test_embedding_generation():
    vec = get_embedding("Export Gas Compressor")
    assert isinstance(vec, list)
    assert len(vec) > 0


def test_document_index_and_search():
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
                        content="Export Gas Compressor C-101",
                        bbox=BoundingBox(x0=0.1, y0=0.1, x1=0.3, y1=0.2),
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
                        content="Lift Gas Compressor C-101",
                        bbox=BoundingBox(x0=0.1, y0=0.1, x1=0.3, y1=0.2),
                    ),
                ],
            )
        ],
    )

    delta_report = (
        "# Delta Report\n\n"
        "### Modified Elements\n"
        "- **Item #1** `[Delta Report, Item #1]` (Page 1, label):\n"
        "  - **Description**: Modified label from Export Gas Compressor to Lift Gas Compressor\n"
    )

    index = DocumentIndex(persist_dir=None)
    indexed_count = index.build_index(doc_a, doc_b, delta_report)
    assert indexed_count >= 3

    results = index.search("What changed in the compressor label?", top_k=2)
    assert len(results) > 0
    assert "text" in results[0]


def test_answer_engine_grounded_response():
    index = DocumentIndex(persist_dir=None)
    engine = AnswerEngine(index=index)

    res = engine.answer_question("What is the title of PID A?")
    assert isinstance(res, AnswerResult)
    assert res.question == "What is the title of PID A?"
    assert res.answer != ""
    assert isinstance(res.citations, list)
