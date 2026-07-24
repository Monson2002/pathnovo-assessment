from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in normalized coordinates (0-1 range)."""

    x0: float
    y0: float
    x1: float
    y1: float

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
    confidence: float = 1.0  # 1.0 for native PDF, OCR confidence for scanned
    font_size: Optional[float] = None
    font_name: Optional[str] = None
    block_type: str = "text"  # text | label | dimension | note | title


class DrawingElement(BaseModel):
    """A geometric element extracted from a document page."""

    element_type: str  # line | rect | circle | path | symbol
    bbox: BoundingBox
    properties: Dict[str, Any] = Field(default_factory=dict)


class Page(BaseModel):
    """One page/sheet of a document."""

    page_number: int
    width: float  # points or pixels
    height: float  # points or pixels
    text_blocks: List[TextBlock] = Field(default_factory=list)
    drawing_elements: List[DrawingElement] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    """Metadata about the source document."""

    pid: str  # The persistent identifier
    filename: str
    format: str  # native_pdf | scanned_pdf | dwg
    page_count: int
    revision_label: Optional[str] = None


class CanonicalDocument(BaseModel):
    """The format-agnostic intermediate model."""

    metadata: DocumentMetadata
    pages: List[Page]
