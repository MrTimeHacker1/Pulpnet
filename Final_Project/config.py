"""Central configuration for the IITK Agentic RAG platform.

Uses pydantic-settings v2 (`SettingsConfigDict`, NOT an inner `class Config`).
Values come from environment / `.env`; every field has a dev-friendly default so
the system runs out of the box with embedded/local backends (Qdrant local-file
mode, SQLite, optional Redis).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

# Native-runtime hygiene set BEFORE torch/OpenMP initialise. The real segfault
# root-cause on this host is qdrant-client sharing a process with torch (handled
# architecturally: ingestion isolates the Qdrant write into a subprocess, and
# serving reads a snapshot and never imports qdrant-client). With that isolation
# in place, torch can use multiple threads safely; 8 keeps CPU inference snappy
# without oversubscribing this shared 48-core box. Override via the real env.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Guard against duplicate OpenMP runtimes co-loaded across native deps.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = directory containing this file.
ROOT_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Data locations ---
    data_dir: Path = Field(default=ROOT_DIR / "data")
    courses_dir: Path = Field(default=ROOT_DIR / "data" / "courses")
    # NOTE: real folder is "academic_documents" (not "academic_docs").
    academic_docs_dir: Path = Field(default=ROOT_DIR / "data" / "academic_documents")
    timetable_dir: Path = Field(default=ROOT_DIR / "data" / "timetable")

    # --- Vector store (Qdrant) ---
    qdrant_path: str = Field(default=str(ROOT_DIR / "qdrant_storage"))
    qdrant_url: str | None = Field(default=None)
    qdrant_collection: str = Field(default="iitk_rag")
    # Torch-safe serving read path (written at ingestion). Serving loads this
    # instead of importing qdrant-client (which segfaults alongside torch here).
    retrieval_snapshot: str = Field(default=str(ROOT_DIR / "retrieval_snapshot.pkl"))

    # --- Relational DB ---
    db_url: str = Field(default=f"sqlite:///{ROOT_DIR / 'rag.db'}")

    # --- Cache (Redis) ---
    redis_url: str | None = Field(default="redis://localhost:6379/0")
    cache_ttl_seconds: int = Field(default=86400)
    semantic_cache_threshold: float = Field(default=0.95)

    # --- Models ---
    embed_model: str = Field(default="BAAI/bge-base-en-v1.5")
    reranker_model: str = Field(default="BAAI/bge-reranker-base")
    # Valid values: a local HF checkpoint id, "mock" (offline tests), or "hf"
    # (routes through HFRouterLLM — hosted inference, no local weights/GPU needed).
    generator_model: str = Field(default="google/gemma-4-E4B-it")
    # Model id sent to the OpenAI-compatible router when generator_model="hf".
    hf_model: str = Field(default="llama-3.3-70b-versatile")
    # Base URL for the OpenAI-compatible inference endpoint. Switch providers by
    # changing this: HF router, Groq, Together, etc.
    hf_base_url: str = Field(default="https://router.huggingface.co/v1")
    model_device: str = Field(default="auto")
    embed_dim: int = Field(default=768)

    # --- Retrieval / agent tuning ---
    retrieve_top_k: int = Field(default=20)
    rerank_top_k: int = Field(default=5)
    sufficiency_min_score: float = Field(default=0.2)
    max_retries: int = Field(default=2)
    gen_max_new_tokens: int = Field(default=512)
    gen_temperature: float = Field(default=0.3)

    # --- Misc ---
    log_level: str = Field(default="INFO")
    # HF API token — required when generator_model="hf".
    hf_token: str | None = Field(default=None)

    def resolve_device(self) -> str:
        """Resolve `model_device` to a concrete torch device string.

        "auto" → the CUDA device with the most free VRAM, else "cpu".
        Anything else is returned as-is. torch is imported lazily so importing
        config stays cheap.
        """
        dev = (self.model_device or "auto").lower()
        if dev != "auto":
            return dev
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
            best_idx, best_free = 0, -1
            for idx in range(torch.cuda.device_count()):
                free, _total = torch.cuda.mem_get_info(idx)
                if free > best_free:
                    best_free, best_idx = free, idx
            return f"cuda:{best_idx}"
        except Exception:  # torch missing or CUDA query failed
            return "cpu"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton accessor for Settings."""
    return Settings()


def configure_logging(level: str | None = None) -> None:
    """Idempotent structured-ish logging setup."""
    lvl = (level or get_settings().log_level).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# Convenience module-level singleton.
settings = get_settings()
