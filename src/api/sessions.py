import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from src.canonical.model import CanonicalDocument
from src.chat.answer import AnswerEngine
from src.chat.index import DocumentIndex
from src.delta.engine import DeltaResult
from src.observability.tracing import Tracer


@dataclass
class Session:
    """In-memory state for one delta+chat session.

    Single-process, in-memory by design (matches the CLI's scope). For
    horizontal scaling across multiple API instances, this would need to move
    to a shared store (Redis/S3) keyed by session_id -- see AWS_DEPLOY.md.
    """

    session_id: str
    work_dir: Path
    created_at: float = field(default_factory=time.time)
    doc_a: Optional[CanonicalDocument] = None
    doc_b: Optional[CanonicalDocument] = None
    delta_res: Optional[DeltaResult] = None
    md_report: str = ""
    json_report: str = ""
    overlay_path: Optional[str] = None
    tracer: Optional[Tracer] = None
    index: Optional[DocumentIndex] = None
    chat_engine: Optional[AnswerEngine] = None


class SessionStore:
    def __init__(self, base_dir: Optional[str] = None, ttl_seconds: int = 3600):
        self._sessions: Dict[str, Session] = {}
        self._base_dir = Path(base_dir or tempfile.gettempdir()) / "delta_chat_api"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = ttl_seconds

    def create(self) -> Session:
        self._evict_expired()
        session_id = uuid.uuid4().hex[:16]
        work_dir = self._base_dir / session_id
        work_dir.mkdir(parents=True, exist_ok=True)
        session = Session(session_id=session_id, work_dir=work_dir)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s.created_at > self._ttl_seconds
        ]
        for sid in expired:
            session = self._sessions.pop(sid, None)
            if session:
                shutil.rmtree(session.work_dir, ignore_errors=True)


sessions = SessionStore()
