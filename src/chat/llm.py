import hashlib
import time
from typing import Any, Dict, List, Optional
import httpx
from src.config import settings
from src.observability.logging import get_logger

logger = get_logger(__name__)


# --- Minimal drop-in replacements for the `openai` SDK's response shapes ---
#
# The `openai` Python SDK's client hangs indefinitely against this project's
# NVIDIA endpoint in this environment (reproduced directly: raw `httpx` calls
# to the exact same URL/payload return in ~0.2-1.5s every time; the `openai`
# SDK's `.create()` calls to the same endpoint never return, even with an
# explicit low `timeout=`). Root cause wasn't chased further (env-specific
# httpx/SDK interaction); the fix is to talk to the OpenAI-compatible REST API
# directly with `httpx` instead of through the SDK. These small wrapper
# classes mimic the exact `.choices[0].message.content` /
# `.usage.prompt_tokens` / `.data[i].embedding` shapes the rest of the
# codebase (and its tests, which mock these same attributes) already expects,
# so this is a transport swap only -- no other file needs to change.
def _post_with_retry(
    http: httpx.Client, url: str, retries: int = 2, backoff: float = 0.5, **kwargs: Any
) -> httpx.Response:
    """POST with a short retry for transient connection/DNS blips (observed
    intermittently against this project's NVIDIA endpoint)."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return http.post(url, **kwargs)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    raise last_exc


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _ChatCompletionResponse:
    def __init__(self, content: str, prompt_tokens: int, completion_tokens: int):
        self.choices = [_Choice(content)]
        self.usage = _Usage(prompt_tokens, completion_tokens)


class _EmbeddingItem:
    def __init__(self, embedding: List[float], index: int):
        self.embedding = embedding
        self.index = index


class _EmbeddingResponse:
    def __init__(self, data: List[_EmbeddingItem]):
        self.data = data


class _ChatCompletions:
    def __init__(self, http: httpx.Client, api_key: str, base_url: str):
        self._http = http
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: Optional[float] = None,
        **_: Any,
    ) -> _ChatCompletionResponse:
        resp = _post_with_retry(
            self._http,
            f"{self._base_url}/chat/completions",
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        content = payload["choices"][0]["message"]["content"] or ""
        usage = payload.get("usage") or {}
        return _ChatCompletionResponse(
            content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        )


class _Embeddings:
    def __init__(self, http: httpx.Client, api_key: str, base_url: str):
        self._http = http
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def create(
        self,
        model: str,
        input: List[str],
        timeout: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> _EmbeddingResponse:
        body = {"model": model, "input": input}
        if extra_body:
            body.update(extra_body)
        resp = _post_with_retry(
            self._http,
            f"{self._base_url}/embeddings",
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        resp.raise_for_status()
        raw = resp.json()["data"]
        items = [
            _EmbeddingItem(d["embedding"], d.get("index", i)) for i, d in enumerate(raw)
        ]
        return _EmbeddingResponse(items)


class _Chat:
    def __init__(self, http: httpx.Client, api_key: str, base_url: str):
        self.completions = _ChatCompletions(http, api_key, base_url)


class NvidiaLLMClient:
    """httpx-backed client exposing the same `.chat.completions.create()` /
    `.embeddings.create()` surface as `openai.OpenAI`, so it's a drop-in
    replacement everywhere that surface is used."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 10.0):
        self._http = httpx.Client(timeout=timeout)
        self.chat = _Chat(self._http, api_key, base_url)
        self.embeddings = _Embeddings(self._http, api_key, base_url)


def get_llm_client(api_key: Optional[str] = None) -> NvidiaLLMClient:
    """Return an httpx-backed client configured for the NVIDIA Nemotron endpoint."""
    key = api_key or settings.nvidia_api_key
    if not key:
        logger.warning(
            "NVIDIA_API_KEY is not set. LLM calls will fail and the chat/eval "
            "pipeline will fall back to degraded/deterministic offline behavior. "
            "Set NVIDIA_API_KEY in .env for real LLM responses."
        )
        key = "dummy_key_for_testing"
    return NvidiaLLMClient(
        base_url=settings.nvidia_base_url,
        api_key=key,
        timeout=10.0,
    )


def get_embedding_client(api_key: Optional[str] = None) -> NvidiaLLMClient:
    """Return an httpx-backed client configured for the NVIDIA embeddings endpoint."""
    return get_llm_client(api_key=api_key)


def _deterministic_fallback_embedding(text: str, dim: int = 1024) -> List[float]:
    """Deterministic pseudo-embedding used when the real API is unavailable."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    float_vec = [(b / 255.0) * 2.0 - 1.0 for b in h]
    repeats = (dim // len(float_vec)) + 1
    return (float_vec * repeats)[:dim]


def get_embedding(
    text: str,
    client: Optional[NvidiaLLMClient] = None,
    dim: int = 1024,
    input_type: str = "query",
) -> List[float]:
    """Get vector embedding for text using NVIDIA embedding model or deterministic fallback.

    NVIDIA's llama-nemotron-embed model is asymmetric and requires `input_type`
    ("query" for search queries, "passage" for indexed content) or it rejects
    every call with HTTP 400.
    """
    if not client:
        client = get_embedding_client()

    try:
        response = client.embeddings.create(
            input=[text],
            model=settings.embedding_model,
            timeout=5.0,
            extra_body={"input_type": input_type, "dimensions": dim},
        )
        return response.data[0].embedding
    except Exception as e:
        logger.debug(
            f"API embedding failed, using deterministic fallback ({dim}-dim): {e}"
        )
        return _deterministic_fallback_embedding(text, dim=dim)


def get_embeddings_batch(
    texts: List[str],
    client: Optional[NvidiaLLMClient] = None,
    dim: int = 1024,
    batch_size: int = 100,
    input_type: str = "passage",
) -> List[List[float]]:
    """Embed many texts with as few API round-trips as possible.

    NVIDIA's OpenAI-compatible embeddings endpoint accepts a list of inputs per
    call, so this batches `texts` into chunks of `batch_size` instead of issuing
    one HTTP request per chunk -- for a document with N chunks this cuts an
    O(N) sequence of network round-trips down to O(N / batch_size), which is
    the difference between single-digit-second and multi-minute indexing runs
    on real P&ID-sized documents (hundreds of chunks per delta report).
    """
    if not texts:
        return []
    if not client:
        client = get_embedding_client()

    results: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = client.embeddings.create(
                input=batch,
                model=settings.embedding_model,
                timeout=20.0,
                extra_body={"input_type": input_type, "dimensions": dim},
            )
            # Preserve input order; the API returns embeddings with an `index` field.
            ordered = sorted(response.data, key=lambda d: d.index)
            results.extend(item.embedding for item in ordered)
        except Exception as e:
            logger.debug(
                f"Batch API embedding failed for {len(batch)} item(s), using "
                f"deterministic fallback ({dim}-dim): {e}"
            )
            results.extend(_deterministic_fallback_embedding(t, dim=dim) for t in batch)
    return results
