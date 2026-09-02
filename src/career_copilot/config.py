"""Central place every other module reads configuration from.

Keeping this as one small module (instead of scattering os.environ calls everywhere)
means later phases (tests, CI, Docker) only have to mock/override one thing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, from the repo root, no matter where this module is imported from.
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    chat_model: str
    critic_model: str
    embed_model: str
    chroma_dir: Path
    portfolio_dir: Path
    golden_jds_dir: Path


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        critic_model=os.getenv("OPENAI_CRITIC_MODEL", "gpt-4o-mini"),
        embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        chroma_dir=Path(os.getenv("CHROMA_DB_DIR", "./chroma_db")).resolve(),
        portfolio_dir=_REPO_ROOT / "src" / "career_copilot" / "data" / "portfolio",
        golden_jds_dir=_REPO_ROOT / "src" / "career_copilot" / "data" / "golden_jds",
    )
