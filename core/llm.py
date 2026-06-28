"""
core/llm.py — Factory untuk OpenAI-compatible LLM client.

Cara pakai:
    from core.llm import get_client, chat
    client, model = get_client()

    # Atau shortcut dengan retry otomatis:
    response_text = chat(messages=[{"role": "user", "content": "..."}])
"""
import base64
import logging
import os
import time
from typing import List, Dict, Optional, Tuple

from openai import OpenAI, APIError, APIConnectionError, RateLimitError

from core.config import get_llm_config, load_config

logger = logging.getLogger("twitter_ai.llm")


def get_vision_model() -> str:
    """
    Model multimodal untuk membaca screenshot. Diambil dari env VISION_MODEL
    (atau LLM_VISION_MODEL). Default: model vision umum di OpenRouter.
    """
    return (
        os.getenv("VISION_MODEL")
        or os.getenv("LLM_VISION_MODEL")
        or "openai/gpt-4o-mini"
    )


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


def vision_chat(
    prompt: str,
    image_bytes: bytes,
    mime: str = "image/jpeg",
    max_tokens: int = 700,
    temperature: float = 0.4,
) -> str:
    """
    Kirim 1 prompt + 1 gambar ke model vision (OpenAI-compatible / OpenRouter).
    Kembalikan teks respons. Retry exponential backoff seperti chat().
    """
    llm_cfg = load_config().get("llm", {})
    max_retries = llm_cfg.get("max_retries", 3)
    base_delay = llm_cfg.get("retry_base_delay", 2)

    client, _ = get_client()
    model = get_vision_model()

    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }]

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError as e:
            last_error = e
            time.sleep(base_delay * (2 ** (attempt - 1)))
        except (APIConnectionError, APIError) as e:
            last_error = e
            logger.warning("Vision LLM error (attempt %d/%d): %s", attempt, max_retries, e)
            time.sleep(base_delay * (2 ** (attempt - 1)))

    raise RuntimeError(f"Vision LLM gagal setelah {max_retries} percobaan: {last_error}")
