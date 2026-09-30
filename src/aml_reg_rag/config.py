from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# pgvector 的 vector 型別建立 HNSW 索引的維度上限
HNSW_MAX_DIM = 2000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    llm_model: str = "google_genai:gemini-3.8-flash"
    llm_light_model: str = "google_genai:gemini-3.5-flash-lite"
    eval_judge_model: str = "google_genai:gemini-3.8-flash"
    embedding_model: str = "google_genai:gemini-embedding-001"
    embedding_dim: int = Field(default=1536, gt=0, le=HNSW_MAX_DIM)
    database_url: str = "postgresql+psycopg://reg:reg@localhost:55432/aml_reg"
    top_k: int = Field(default=6, gt=0)
    chunk_max_chars: int = Field(default=600, gt=0)
    llm_timeout: float = 120
    data_dir: Path = PROJECT_ROOT / "data"
    eval_dir: Path = PROJECT_ROOT / "eval"
    var_dir: Path = PROJECT_ROOT / "var"
