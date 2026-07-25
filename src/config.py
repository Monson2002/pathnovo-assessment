from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM & Embedding Settings
    nvidia_api_key: Optional[str] = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1"
    embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    # Nemotron Super 49B pricing (NVIDIA NIM hosted, USD per 1M tokens)
    llm_cost_per_1m_input: float = 0.60
    llm_cost_per_1m_output: float = 1.80

    # Delta Engine Parameters
    text_similarity_threshold: float = 0.4
    spatial_weight: float = 0.3
    confidence_threshold: float = 0.5
    ocr_dpi: int = 300

    # Storage & Telemetry
    trace_output_dir: str = "traces"
    chroma_db_dir: str = ".chroma"
    log_level: str = "INFO"

    # Chroma Cloud & Remote Vector DB Settings
    chroma_use_cloud: bool = False
    chroma_cloud_api_key: Optional[str] = None
    chroma_tenant: str = "default_tenant"
    chroma_database: str = "default_database"
    chroma_host: Optional[str] = None
    chroma_port: int = 8000

    # REST API Settings
    api_auth_token: Optional[str] = (
        None  # if set, required as `Authorization: Bearer <token>`
    )
    api_cors_origins: str = "*"  # comma-separated list of allowed origins, or "*"
    api_session_ttl_seconds: int = 3600

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
