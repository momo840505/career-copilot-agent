from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parents[1]
load_dotenv(_REPO_ROOT / ".env")


def _optional_float(name: str, default: str) -> float | None:
    raw = os.getenv(name, default).strip()
    return float(raw) if raw else None


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    chat_model: str
    critic_model: str
    judge_model: str
    embed_model: str
    openai_timeout_seconds: float
    openai_max_retries: int
    rag_max_distance: float | None
    chroma_dir: Path
    portfolio_dir: Path
    golden_jds_dir: Path
    access_code: str | None
    history_db_path: Path


def get_settings() -> Settings:
    critic_model = os.getenv("OPENAI_CRITIC_MODEL", "gpt-4o")
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        critic_model=critic_model,
        judge_model=os.getenv("OPENAI_JUDGE_MODEL", critic_model),
        embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        openai_timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60")),
        openai_max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
        rag_max_distance=_optional_float("RAG_MAX_DISTANCE", "0.75"),
        chroma_dir=Path(os.getenv("CHROMA_DB_DIR", "./chroma_db")).resolve(),
        portfolio_dir=_PACKAGE_ROOT / "data" / "portfolio",
        golden_jds_dir=_PACKAGE_ROOT / "data" / "golden_jds",
        access_code=os.getenv("ACCESS_CODE") or None,
        history_db_path=Path(os.getenv("HISTORY_DB_PATH", "./history.db")).resolve(),
    )
