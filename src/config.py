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

    # Delta Engine Parameters
    text_similarity_threshold: float = 0.4
    spatial_weight: float = 0.3
    confidence_threshold: float = 0.5
    ocr_dpi: int = 300

    # Storage & Telemetry
    trace_output_dir: str = "traces"
    chroma_db_dir: str = ".chroma"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
