import json
import os
import re
from typing import Any, Dict, List, Optional
import chromadb
from src.canonical.model import CanonicalDocument
from src.chat.llm import get_embedding, get_embeddings_batch
from src.config import settings
from src.observability.logging import get_logger

logger = get_logger(__name__)


class DocumentIndex:
    """Vector index wrapping ChromaDB for document chunks and delta reports."""

    def __init__(
        self,
        collection_name: str = "pid_delta_index",
        persist_dir: Optional[str] = None,
        use_cloud: Optional[bool] = None,
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir or settings.chroma_db_dir
        should_use_cloud = settings.chroma_use_cloud if use_cloud is None else use_cloud

        cloud_connected = False
        if (
            should_use_cloud
            and settings.chroma_cloud_api_key
            and settings.chroma_cloud_api_key != "your-chroma-cloud-api-key-here"
        ):
            try:
                logger.info(
                    f"Connecting to Chroma Cloud (Tenant: {settings.chroma_tenant}, DB: {settings.chroma_database})"
                )
                self.client = chromadb.CloudClient(
                    tenant=settings.chroma_tenant,
                    database=settings.chroma_database,
                    api_key=settings.chroma_cloud_api_key,
                )
                cloud_connected = True
            except Exception as e:
                logger.warning(
                    f"Failed to connect to Chroma Cloud ({e}), falling back to local client."
                )

        if not cloud_connected and settings.chroma_host:
            try:
                logger.info(
                    f"Connecting to Remote Chroma Host at {settings.chroma_host}:{settings.chroma_port}"
                )
                headers = (
                    {"Authorization": f"Bearer {settings.chroma_cloud_api_key}"}
                    if settings.chroma_cloud_api_key
                    else None
                )
                self.client = chromadb.HttpClient(
                    host=settings.chroma_host,
                    port=settings.chroma_port,
                    headers=headers,
                )
                cloud_connected = True
            except Exception as e:
                logger.warning(
                    f"Failed to connect to Remote Chroma Host ({e}), falling back to local client."
                )

        if not cloud_connected:
            if self.persist_dir:
                os.makedirs(self.persist_dir, exist_ok=True)
                self.client = chromadb.PersistentClient(path=self.persist_dir)
            else:
                self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def build_index(
        self,
        doc_a: CanonicalDocument,
        doc_b: CanonicalDocument,
        delta_report_md: str,
        force_reindex: bool = False,
    ) -> int:
        """Index text blocks from PID A, PID B, and delta report markdown entries with reuse caching."""
        logger.info(
            "Checking ChromaDB vector index over PID A, PID B, and Delta Report"
        )
        items_to_process = []

        # 1. PID A
        for page in doc_a.pages:
            for idx, block in enumerate(page.text_blocks):
                content = f"PID A ({doc_a.metadata.pid}) Page {page.page_number}: {block.content}"
                doc_id = f"pid_a_p{page.page_number}_b{idx}"
                meta = {
                    "source": "pid_a",
                    "pid": doc_a.metadata.pid,
                    "page": page.page_number,
                    "content": block.content,
                    "bbox": json.dumps(block.bbox.model_dump()),
                }
                items_to_process.append((doc_id, content, meta))

        # 2. PID B
        for page in doc_b.pages:
            for idx, block in enumerate(page.text_blocks):
                content = f"PID B ({doc_b.metadata.pid}) Page {page.page_number}: {block.content}"
                doc_id = f"pid_b_p{page.page_number}_b{idx}"
                meta = {
                    "source": "pid_b",
                    "pid": doc_b.metadata.pid,
                    "page": page.page_number,
                    "content": block.content,
                    "bbox": json.dumps(block.bbox.model_dump()),
                }
                items_to_process.append((doc_id, content, meta))

        # 3. Delta Report
        lines = delta_report_md.split("\n")
        current_chunk = []
        chunk_idx = 0

        def process_delta_chunk(chunk_lines, idx):
            text_chunk = "\n".join(chunk_lines).strip()
            if not text_chunk:
                return None
            page_match = re.search(r"on page (\d+)", text_chunk, re.IGNORECASE)
            chunk_page = int(page_match.group(1)) if page_match else 1
            doc_id = f"delta_report_c{idx}"
            meta = {
                "source": "delta_report",
                "pid": f"{doc_a.metadata.pid}_vs_{doc_b.metadata.pid}",
                "page": chunk_page,
                "content": text_chunk,
                "bbox": "none",
            }
            return (doc_id, text_chunk, meta)

        for line in lines:
            if (
                line.startswith("### ")
                or line.startswith("## ")
                or line.startswith("- **Item #")
            ):
                if current_chunk:
                    item = process_delta_chunk(current_chunk, chunk_idx)
                    if item:
                        items_to_process.append(item)
                        chunk_idx += 1
                    current_chunk = []
            current_chunk.append(line)

        if current_chunk:
            item = process_delta_chunk(current_chunk, chunk_idx)
            if item:
                items_to_process.append(item)

        # Check existing IDs in ChromaDB in batches to respect payload limits
        existing_ids = set()
        all_ids = [doc_id for doc_id, _, _ in items_to_process]
        BATCH_SIZE = 200

        if not force_reindex and all_ids:
            for i in range(0, len(all_ids), BATCH_SIZE):
                batch_ids = all_ids[i : i + BATCH_SIZE]
                try:
                    res = self.collection.get(ids=batch_ids)
                    if res and "ids" in res:
                        existing_ids.update(res["ids"])
                except Exception as e:
                    logger.warning(
                        f"Failed to query existing collection IDs batch: {e}"
                    )

        new_items = [
            (doc_id, content, meta)
            for doc_id, content, meta in items_to_process
            if doc_id not in existing_ids
        ]

        if not new_items and items_to_process:
            logger.info(
                f"Vector index is up to date ({len(items_to_process)} chunks already indexed). Skipping embedding generation."
            )
            return len(items_to_process)

        documents: List[str] = [content for _, content, _ in new_items]
        metadatas: List[Dict[str, Any]] = [meta for _, _, meta in new_items]
        ids: List[str] = [doc_id for doc_id, _, _ in new_items]

        # Batched (not one-call-per-chunk) so indexing a large delta report
        # (hundreds of chunks) doesn't turn into hundreds of sequential
        # network round-trips.
        embeddings: List[List[float]] = get_embeddings_batch(documents)

        # Batch upsert in blocks of BATCH_SIZE (200) to prevent Chroma Cloud NUM_RECORDS payload error
        if documents:
            for i in range(0, len(documents), BATCH_SIZE):
                self.collection.upsert(
                    documents=documents[i : i + BATCH_SIZE],
                    embeddings=embeddings[i : i + BATCH_SIZE],
                    metadatas=metadatas[i : i + BATCH_SIZE],
                    ids=ids[i : i + BATCH_SIZE],
                )

        logger.info(
            f"Vector index updated: {len(new_items)} new/updated chunks indexed (Total: {len(items_to_process)})"
        )
        return len(items_to_process)

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
