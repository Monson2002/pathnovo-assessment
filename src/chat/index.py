import json
import os
import re
from typing import Any, Dict, List, Optional
import chromadb
from src.canonical.model import CanonicalDocument
from src.chat.llm import get_embedding
from src.config import settings
from src.observability.logging import get_logger

logger = get_logger(__name__)


class DocumentIndex:
    """Vector index wrapping ChromaDB for document chunks and delta reports."""

    def __init__(
        self,
        collection_name: str = "pid_delta_index",
        persist_dir: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir or settings.chroma_db_dir

        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)
        else:
            self.client = chromadb.Client()

        # Delete existing collection if any to allow fresh indexing per pair
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def build_index(
        self,
        doc_a: CanonicalDocument,
        doc_b: CanonicalDocument,
        delta_report_md: str,
    ) -> int:
        """Index text blocks from PID A, PID B, and delta report markdown entries."""
        logger.info(
            "Building ChromaDB vector index over PID A, PID B, and Delta Report"
        )
        documents: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        count = 0

        # 1. Index PID A
        for page in doc_a.pages:
            for idx, block in enumerate(page.text_blocks):
                content = f"PID A ({doc_a.metadata.pid}) Page {page.page_number}: {block.content}"
                vec = get_embedding(content)
                doc_id = f"pid_a_p{page.page_number}_b{idx}"
                documents.append(content)
                embeddings.append(vec)
                metadatas.append(
                    {
                        "source": "pid_a",
                        "pid": doc_a.metadata.pid,
                        "page": page.page_number,
                        "content": block.content,
                        "bbox": json.dumps(block.bbox.model_dump()),
                    }
                )
                ids.append(doc_id)
                count += 1

        # 2. Index PID B
        for page in doc_b.pages:
            for idx, block in enumerate(page.text_blocks):
                content = f"PID B ({doc_b.metadata.pid}) Page {page.page_number}: {block.content}"
                vec = get_embedding(content)
                doc_id = f"pid_b_p{page.page_number}_b{idx}"
                documents.append(content)
                embeddings.append(vec)
                metadatas.append(
                    {
                        "source": "pid_b",
                        "pid": doc_b.metadata.pid,
                        "page": page.page_number,
                        "content": block.content,
                        "bbox": json.dumps(block.bbox.model_dump()),
                    }
                )
                ids.append(doc_id)
                count += 1

        # 3. Index Delta Report paragraphs / itemized sections
        lines = delta_report_md.split("\n")
        current_chunk = []
        chunk_idx = 0

        for line in lines:
            if (
                line.startswith("### ")
                or line.startswith("## ")
                or line.startswith("- **Item #")
            ):
                if current_chunk:
                    text_chunk = "\n".join(current_chunk).strip()
                    if text_chunk:
                        # Extract page number if present in item description (e.g., "on page 2")
                        page_match = re.search(
                            r"on page (\d+)", text_chunk, re.IGNORECASE
                        )
                        chunk_page = int(page_match.group(1)) if page_match else 1

                        vec = get_embedding(text_chunk)
                        doc_id = f"delta_report_c{chunk_idx}"
                        documents.append(text_chunk)
                        embeddings.append(vec)
                        metadatas.append(
                            {
                                "source": "delta_report",
                                "pid": f"{doc_a.metadata.pid}_vs_{doc_b.metadata.pid}",
                                "page": chunk_page,
                                "content": text_chunk,
                                "bbox": "none",
                            }
                        )
                        ids.append(doc_id)
                        chunk_idx += 1
                        count += 1
                    current_chunk = []
            current_chunk.append(line)

        if current_chunk:
            text_chunk = "\n".join(current_chunk).strip()
            if text_chunk:
                page_match = re.search(r"on page (\d+)", text_chunk, re.IGNORECASE)
                chunk_page = int(page_match.group(1)) if page_match else 1

                vec = get_embedding(text_chunk)
                doc_id = f"delta_report_c{chunk_idx}"
                documents.append(text_chunk)
                embeddings.append(vec)
                metadatas.append(
                    {
                        "source": "delta_report",
                        "pid": f"{doc_a.metadata.pid}_vs_{doc_b.metadata.pid}",
                        "page": chunk_page,
                        "content": text_chunk,
                        "bbox": "none",
                    }
                )
                ids.append(doc_id)
                count += 1

        if documents:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

        logger.info(f"Vector index built successfully with {count} chunks")
        return count

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top_k matching chunks for a user query."""
        query_vec = get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
        )

        chunks = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else []
            dists = results["distances"][0] if results.get("distances") else []

            for doc, meta, dist in zip(docs, metas, dists):
                chunks.append(
                    {
                        "text": doc,
                        "metadata": meta,
                        "distance": dist,
                    }
                )

        return chunks
