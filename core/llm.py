"""
core/llm.py — Factory untuk OpenAI-compatible LLM client.

Cara pakai:
    from core.llm import get_client, chat
    client, model = get_client()

    # Atau shortcut dengan retry otomatis:
    response_text = chat(messages=[{"role": "user", "content": "..."}])
"""
import logging
import time
from typing import List, Dict, Optional, Tuple

from openai import OpenAI, APIError, APIConnectionError, RateLimitError

from core.config import get_llm_config, load_config

logger = logging.getLogger("twitter_ai.llm")


def get_client() -> Tuple[OpenAI, str]:
    """
    Kembalikan (client, model_name) dari env vars.
    Mendukung LLM_* (baru) dan OPENAI_*/AI_MODEL (lama).
    """
    cfg = get_llm_config()
    if not cfg["api_key"]:
        raise ValueError(
            "LLM API key tidak ditemukan. "
            "Set LLM_API_KEY atau OPENAI_API_KEY di .env"
        )
    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        timeout=load_config().get("llm", {}).get("timeout", 60),
    )
    model = cfg["model"] or "gpt-4o-mini"
    return client, model


def chat(
    messages: List[Dict[str, str]],
    max_tokens: int = 500,
    temperature: float = 0.8,
    json_mode: bool = False,
) -> str:
    """
    Kirim pesan ke LLM dengan retry exponential backoff.
    Kembalikan teks respons (string).

    Args:
        messages: list pesan format OpenAI [{"role": ..., "content": ...}]
        max_tokens: batas token output
        temperature: kreativitas (0.0–1.0)
        json_mode: aktifkan response_format JSON jika provider mendukung
    """
    llm_cfg = load_config().get("llm", {})
    max_retries = llm_cfg.get("max_retries", 3)
    base_delay = llm_cfg.get("retry_base_delay", 2)

    client, model = get_client()

    kwargs: Dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(f"Rate limit LLM (attempt {attempt}/{max_retries}), retry in {delay}s")
            last_error = e
            time.sleep(delay)
        except (APIConnectionError, APIError) as e:
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(f"LLM API error (attempt {attempt}/{max_retries}): {e}, retry in {delay}s")
            last_error = e
            time.sleep(delay)

    raise RuntimeError(f"LLM gagal setelah {max_retries} percobaan: {last_error}")
