import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.config import settings
from src.observability.logging import set_correlation_id


@dataclass
class Span:
    """Represents a timed operation segment in a trace."""

    name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["Span"] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return round((self.end_time - self.start_time) * 1000, 2)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class Trace:
    """A trace recording stage timings and LLM call telemetry for a request."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    root_span: Optional[Span] = None
    llm_calls: List[Dict[str, Any]] = field(default_factory=list)

    def log_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        estimated_cost: float = 0.0,
    ) -> None:
        self.llm_calls.append(
            {
                "timestamp": time.time(),
                "model": model,
                "prompt_snippet": prompt[:200],
                "response_snippet": response[:200],
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "estimated_cost_usd": estimated_cost,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "root_span": self.root_span.to_dict() if self.root_span else None,
            "llm_calls": self.llm_calls,
        }

    def save(self, output_dir: Optional[str] = None) -> str:
        out_dir = output_dir or settings.trace_output_dir
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"trace_{self.trace_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


class Tracer:
    """Helper manager for creating and tracking Spans within a Trace."""

    def __init__(self, trace_name: str = "request_trace"):
        self.trace = Trace()
        self.trace.root_span = Span(name=trace_name)
        self._span_stack = [self.trace.root_span]
        # Correlate every structured log emitted during this trace with its trace_id.
        set_correlation_id(self.trace.trace_id)

    @contextmanager
    def span(self, name: str, **metadata):
        parent = self._span_stack[-1]
        current = Span(name=name, metadata=metadata)
        parent.children.append(current)
        self._span_stack.append(current)

        try:
            yield current
        except Exception as exc:
            current.metadata["error"] = str(exc)
            raise
        finally:
            current.end_time = time.time()
            self._span_stack.pop()

    def finish(self, output_dir: Optional[str] = None) -> str:
        if self.trace.root_span:
            self.trace.root_span.end_time = time.time()
        return self.trace.save(output_dir=output_dir)
