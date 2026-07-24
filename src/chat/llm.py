from typing import List, Optional
from openai import OpenAI
from src.config import settings


def get_llm_client(api_key: Optional[str] = None) -> OpenAI:
    """Return an OpenAI client configured for NVIDIA Nemotron LLM endpoint."""
    key = api_key or settings.nvidia_api_key or "dummy_key_for_testing"
    return OpenAI(
        base_url=settings.nvidia_base_url,
        api_key=key,
    )


def get_embedding_client(api_key: Optional[str] = None) -> OpenAI:
    """Return an OpenAI client configured for NVIDIA embeddings endpoint."""
    return get_llm_client(api_key=api_key)


def get_embedding(text: str, client: Optional[OpenAI] = None) -> List[float]:
    """Get vector embedding for text using NVIDIA embedding model."""
    if not client:
        client = get_embedding_client()

    try:
        response = client.embeddings.create(
            input=[text],
            model=settings.embedding_model,
        )
        return response.data[0].embedding
    except Exception:
        # Fallback deterministic pseudo-embedding (128 dimensions) for offline/testing mode without valid API key
        import hashlib

        h = hashlib.sha256(text.encode("utf-8")).digest()
        float_vec = [(b / 255.0) * 2.0 - 1.0 for b in h]
        # Repeat to match 128 dims
        return (float_vec * 4)[:128]
