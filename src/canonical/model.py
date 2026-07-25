from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in normalized coordinates (0-1 range)."""

    x0: float = Field(..., ge=0.0, le=1.0)
    y0: float = Field(..., ge=0.0, le=1.0)
    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)


class TextBlock(BaseModel):
    """A block of text extracted from a document page."""

    content: str
    bbox: BoundingBox
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    font_size: Optional[float] = None
    font_name: Optional[str] = None
    block_type: Literal["text", "label", "dimension", "note", "title"] = "text"


class DrawingElement(BaseModel):
    """A geometric element extracted from a document page."""

    element_type: Literal["line", "rect", "circle", "path", "symbol"] = "path"
    bbox: BoundingBox
    properties: Dict[str, Any] = Field(default_factory=dict)


class Page(BaseModel):
    """One page/sheet of a document."""

    page_number: int = Field(..., ge=1)
    width: float = Field(..., gt=0.0)
    height: float = Field(..., gt=0.0)
    text_blocks: List[TextBlock] = Field(default_factory=list)
    drawing_elements: List[DrawingElement] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    """Metadata about the source document."""

    pid: str  # The persistent identifier
    filename: str
    format: Literal["native_pdf", "scanned_pdf", "dwg"]
    page_count: int = Field(..., ge=0)
    revision_label: Optional[str] = None
    is_cached: bool = False


class CanonicalDocument(BaseModel):
    """The format-agnostic intermediate model."""

    metadata: DocumentMetadata
    pages: List[Page] = Field(default_factory=list)
