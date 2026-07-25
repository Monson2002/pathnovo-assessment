import hashlib
from unittest.mock import MagicMock, patch
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
    Citation,
    DocumentIndex,
    get_embedding,
)


def _fake_embedding(text: str, dim: int = 1024) -> list:
    """Deterministic, offline stand-in for a real embedding vector, used to keep
    tests hermetic (no network calls, no hangs in sandboxed/offline environments).
    Matches the real embedding model's dimensionality so it's compatible with any
    already-provisioned Chroma collection (local or cloud)."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    float_vec = [(b / 255.0) * 2.0 - 1.0 for b in h]
    repeats = (dim // len(float_vec)) + 1
    return (float_vec * repeats)[:dim]


def test_embedding_generation():
    """get_embedding() should call the embedding client and return its vector."""
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value.data = [
        MagicMock(embedding=_fake_embedding("Export Gas Compressor"))
    ]

    with patch("src.chat.llm.get_embedding_client", return_value=mock_client):
        vec = get_embedding("Export Gas Compressor", client=mock_client)

    assert isinstance(vec, list)
    assert len(vec) > 0
    mock_client.embeddings.create.assert_called_once()


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

    with patch("src.chat.index.get_embedding", side_effect=_fake_embedding):
        index = DocumentIndex(persist_dir=None)
        indexed_count = index.build_index(doc_a, doc_b, delta_report)
        assert indexed_count >= 3

        results = index.search("What changed in the compressor label?", top_k=2)
    assert len(results) > 0
    assert "text" in results[0]


def test_answer_engine_grounded_response():
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
                        content="Export Gas Compressor Title",
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
                        content="Lift Gas Compressor Title",
                        bbox=BoundingBox(x0=0.1, y0=0.1, x1=0.3, y1=0.2),
                    ),
                ],
            )
        ],
    )

    with patch("src.chat.index.get_embedding", side_effect=_fake_embedding):
        index = DocumentIndex(persist_dir=None)
        index.build_index(doc_a, doc_b, "# Delta Report")

        # Mock LLM Client to return grounded answer with citation
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "The title of PID A is Export Gas Compressor. [PID A, Page 1]"
        )
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        engine = AnswerEngine(index=index, llm_client=mock_client)
        res = engine.answer_question("What is the title of PID A?")

    assert isinstance(res, AnswerResult)
    assert res.question == "What is the title of PID A?"
    assert "Export Gas Compressor" in res.answer
    assert len(res.citations) > 0
    assert isinstance(res.citations[0], Citation)
    assert res.citations[0].source == "pid_a"
    assert res.citations[0].page == 1
