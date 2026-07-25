import base64
import os
from typing import List, Optional

import fitz  # PyMuPDF
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.api.sessions import Session, sessions
from src.chat.answer import AnswerEngine, AnswerResult
from src.chat.index import DocumentIndex
from src.config import settings
from src.delta.engine import DeltaEngine, DeltaItem
from src.delta.report import generate_delta_report
from src.ingest.registry import detect_and_ingest
from src.markup.overlay import generate_delta_markup
from src.observability.logging import get_logger
from src.observability.tracing import Tracer

logger = get_logger(__name__)

app = FastAPI(
    title="Document Delta & Grounded Chat API",
    description=(
        "Ingest two P&ID document revisions, compute a structured delta, and "
        "answer grounded questions with citations over both documents plus "
        "the delta report."
    ),
    version="1.0.0",
)

_allowed_origins = (
    ["*"]
    if settings.api_cors_origins.strip() == "*"
    else [o.strip() for o in settings.api_cors_origins.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_PAIRS = {
    "pair_01": {
        "label": "pair_01 (two distinct P&ID sheets, real sample data)",
        "pid_a": "data/samples/pair_01/Export Gas Compressor-P&ID (1).pdf",
        "pid_b": "data/samples/pair_01/Lift Gas compressor-P&ID.pdf",
    },
    "pair_02": {
        "label": "pair_02 (synthetic true revision pair, exact ground truth)",
        "pid_a": "data/samples/pair_02/pid_a_rev0.pdf",
        "pid_b": "data/samples/pair_02/pid_b_rev1.pdf",
    },
}


def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    """Optional bearer-token gate. No-op if API_AUTH_TOKEN is unset (dev default)."""
    if not settings.api_auth_token:
        return
    expected = f"Bearer {settings.api_auth_token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


def _get_session(session_id: str) -> Session:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return session


# --- Schemas -------------------------------------------------------------------


class DeltaSummary(BaseModel):
    session_id: str
    pid_a_label: str
    pid_b_label: str
    summary: dict
    items: List[DeltaItem]
    has_overlay: bool


class ChatRequest(BaseModel):
    session_id: str
    question: str
    history: List[dict] = []


class TraceResponse(BaseModel):
    trace_id: str
    root_span: dict
    llm_calls: List[dict]


# --- Routes ----------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/samples")
def list_samples() -> dict:
    return {
        key: {
            "label": val["label"],
            "available": os.path.exists(val["pid_a"]) and os.path.exists(val["pid_b"]),
        }
        for key, val in SAMPLE_PAIRS.items()
    }


@app.post("/delta", response_model=DeltaSummary, dependencies=[Depends(require_auth)])
async def compute_delta(
    pid_a: Optional[UploadFile] = File(default=None),
    pid_b: Optional[UploadFile] = File(default=None),
    sample_pair: Optional[str] = Form(default=None),
) -> DeltaSummary:
    session = sessions.create()
    tracer = Tracer(trace_name="api_delta_pipeline")
    session.tracer = tracer

    if sample_pair:
        if sample_pair not in SAMPLE_PAIRS:
            raise HTTPException(
                status_code=400, detail=f"Unknown sample_pair: {sample_pair}"
            )
        pid_a_path = SAMPLE_PAIRS[sample_pair]["pid_a"]
        pid_b_path = SAMPLE_PAIRS[sample_pair]["pid_b"]
        label_a, label_b = os.path.basename(pid_a_path), os.path.basename(pid_b_path)
    elif pid_a and pid_b:
        upload_dir = session.work_dir / "uploads"
        upload_dir.mkdir(exist_ok=True)
        pid_a_path = str(upload_dir / pid_a.filename)
        pid_b_path = str(upload_dir / pid_b.filename)
        with open(pid_a_path, "wb") as f:
            f.write(await pid_a.read())
        with open(pid_b_path, "wb") as f:
            f.write(await pid_b.read())
        label_a, label_b = pid_a.filename, pid_b.filename
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either `sample_pair` or both `pid_a` and `pid_b` file uploads.",
        )

    try:
        with tracer.span("ingest_documents"):
            doc_a = detect_and_ingest(
                pid_a_path,
                pid=label_a,
                cache_dir=str(session.work_dir / ".cache" / "ingest"),
            )
            doc_b = detect_and_ingest(
                pid_b_path,
                pid=label_b,
                cache_dir=str(session.work_dir / ".cache" / "ingest"),
            )

        with tracer.span("compute_delta"):
            engine = DeltaEngine()
            delta_res = engine.compute_delta(doc_a, doc_b)

        with tracer.span("generate_report"):
            output_dir = session.work_dir / "output"
            output_dir.mkdir(exist_ok=True)
            md_report, json_report = generate_delta_report(
                delta_res, output_dir=str(output_dir)
            )
            overlay_path = generate_delta_markup(
                pid_b_path,
                delta_res,
                output_path=str(output_dir / "annotated_delta.pdf"),
            )
    except Exception as e:
        logger.exception(f"Delta pipeline failed for session {session.session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    session.doc_a = doc_a
    session.doc_b = doc_b
    session.delta_res = delta_res
    session.md_report = md_report
    session.json_report = json_report
    session.overlay_path = (
        overlay_path if overlay_path and os.path.exists(overlay_path) else None
    )

    return DeltaSummary(
        session_id=session.session_id,
        pid_a_label=label_a,
        pid_b_label=label_b,
        summary=delta_res.summary,
        items=delta_res.items,
        has_overlay=session.overlay_path is not None,
    )


@app.get("/delta/{session_id}/report.md", dependencies=[Depends(require_auth)])
def get_report_md(session_id: str):
    session = _get_session(session_id)
    if not session.md_report:
        raise HTTPException(
            status_code=404, detail="No report generated for this session yet"
        )
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(session.md_report, media_type="text/markdown")


@app.get("/delta/{session_id}/report.json", dependencies=[Depends(require_auth)])
def get_report_json(session_id: str):
    session = _get_session(session_id)
    if not session.json_report:
        raise HTTPException(
            status_code=404, detail="No report generated for this session yet"
        )
    from fastapi.responses import Response

    return Response(session.json_report, media_type="application/json")


@app.get("/delta/{session_id}/overlay.pdf", dependencies=[Depends(require_auth)])
def get_overlay_pdf(session_id: str):
    session = _get_session(session_id)
    if not session.overlay_path or not os.path.exists(session.overlay_path):
        raise HTTPException(status_code=404, detail="No overlay PDF for this session")
    return FileResponse(
        session.overlay_path,
        media_type="application/pdf",
        filename="annotated_delta.pdf",
    )


@app.get("/delta/{session_id}/overlay/pages", dependencies=[Depends(require_auth)])
def get_overlay_pages(session_id: str) -> dict:
    """Render the overlay PDF's pages as base64 PNGs, so a thin frontend can
    display them without needing a PDF-rendering library of its own."""
    session = _get_session(session_id)
    if not session.overlay_path or not os.path.exists(session.overlay_path):
        raise HTTPException(status_code=404, detail="No overlay PDF for this session")

    pages_b64 = []
    with fitz.open(session.overlay_path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            pages_b64.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
    return {"pages": pages_b64}


@app.post("/chat", response_model=AnswerResult, dependencies=[Depends(require_auth)])
def chat(req: ChatRequest) -> AnswerResult:
    session = _get_session(req.session_id)
    if session.delta_res is None or session.doc_a is None or session.doc_b is None:
        raise HTTPException(
            status_code=400,
            detail="Compute a delta for this session (POST /delta) before chatting.",
        )

    if session.index is None:
        with session.tracer.span("build_index") if session.tracer else _noop():
            index = DocumentIndex(
                persist_dir=str(session.work_dir / ".chroma"),
                collection_name=f"session_{session.session_id}",
            )
            index.build_index(session.doc_a, session.doc_b, session.md_report)
            session.index = index
            session.chat_engine = AnswerEngine(index=index)

    with session.tracer.span("chat_turn") if session.tracer else _noop():
        result = session.chat_engine.answer_question(
            req.question,
            tracer=session.tracer.trace if session.tracer else None,
            chat_history=req.history if req.history else None,
        )
    return result


@app.get(
    "/trace/{session_id}",
    response_model=TraceResponse,
    dependencies=[Depends(require_auth)],
)
def get_trace(session_id: str) -> TraceResponse:
    session = _get_session(session_id)
    if session.tracer is None:
        raise HTTPException(status_code=404, detail="No trace for this session")
    trace = session.tracer.trace
    return TraceResponse(
        trace_id=trace.trace_id,
        root_span=trace.root_span.to_dict() if trace.root_span else {},
        llm_calls=trace.llm_calls,
    )


class _noop:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
