"""
core/config.py — Pemuat konfigurasi global.

Cara pakai:
    from core.config import load_config, get_llm_config
    cfg = load_config()
    llm = get_llm_config()
"""
import os
import yaml
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).parent.parent


@lru_cache(maxsize=1)
def _load_yaml(path: str) -> dict:
    """Muat file YAML dan cache hasilnya (tidak reload setiap call)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(filename: str = "config.yaml") -> dict:
    """Kembalikan dict konfigurasi dari config.yaml di root repo."""
    return _load_yaml(str(ROOT / filename))


def get_llm_config() -> dict:
    """
    Ambil konfigurasi LLM dari environment variables.
    Mendukung nama baru (LLM_*) dan nama lama (OPENAI_*/AI_MODEL) sebagai fallback.
    """
    return {
        "base_url": os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        "api_key": os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        "model": os.getenv("LLM_MODEL") or os.getenv("AI_MODEL"),
    }
