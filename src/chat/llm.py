import hashlib
from typing import List, Optional
from openai import OpenAI
from src.config import settings
from src.observability.logging import get_logger

logger = get_logger(__name__)


def get_llm_client(api_key: Optional[str] = None) -> OpenAI:
    """Return an OpenAI client configured for NVIDIA Nemotron LLM endpoint with fast timeout."""
    key = api_key or settings.nvidia_api_key or "dummy_key_for_testing"
    return OpenAI(
        base_url=settings.nvidia_base_url,
        api_key=key,
        timeout=3.0,
        max_retries=1,
    )


def get_embedding_client(api_key: Optional[str] = None) -> OpenAI:
    """Return an OpenAI client configured for NVIDIA embeddings endpoint."""
    return get_llm_client(api_key=api_key)


def get_embedding(
    text: str, client: Optional[OpenAI] = None, dim: int = 1024
) -> List[float]:
    """Get vector embedding for text using NVIDIA embedding model or deterministic fallback."""
    if not client:
        client = get_embedding_client()

    try:
        response = client.embeddings.create(
            input=[text],
            model=settings.embedding_model,
            timeout=3.0,
        )
        return response.data[0].embedding
    except Exception as e:
        logger.debug(
            f"API embedding failed, using deterministic fallback ({dim}-dim): {e}"
        )
        # Fallback deterministic pseudo-embedding (matching embedding dimensionality)
        h = hashlib.sha256(text.encode("utf-8")).digest()
        float_vec = [(b / 255.0) * 2.0 - 1.0 for b in h]
        # Repeat to match target dimensions (default 1024)
        repeats = (dim // len(float_vec)) + 1
        return (float_vec * repeats)[:dim]
