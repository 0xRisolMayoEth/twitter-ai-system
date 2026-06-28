"""
agents/image_agent.py — IMAGE AGENT (BARU).

Input : tweet_text, trending_topic, search_query_for_image
Output: ImageRec { image_description, google_search_queries[3], image_style, reason }

Agent ini TIDAK membuat gambar — ia merekomendasikan kata kunci pencarian
Google Images (2 dalam bahasa Inggris, 1 dalam bahasa Jepang) yang cocok untuk
menemani tweet salaryman. Rekomendasi dikirim ke Telegram bersama draft.

Aturan: gambar harus SFW, tanpa wajah orang nyata, dan sesuai mood salaryman.
"""
import json
from typing import List

from core.llm import chat
from core.logger import get_logger
from core.models import ImageRec

logger = get_logger("image_agent")

_SYSTEM = (
    "Kamu IMAGE AGENT. Kamu menyarankan gambar pendukung untuk tweet salaryman "
    "Jepang. Gambar HARUS SFW, tanpa wajah orang nyata (boleh ilustrasi/objek/"
    "pemandangan), dan cocok dengan mood salaryman."
)


def recommend_image(tweet_text: str, trending_topic: str,
                    search_query_for_image: str = "") -> ImageRec:
    """Hasilkan rekomendasi gambar (3 query Google: 2 EN + 1 JP)."""
    prompt = _build_prompt(tweet_text, trending_topic, search_query_for_image)
    try:
        raw = chat(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.6,
        )
        rec = _parse(raw, search_query_for_image or trending_topic)
        logger.info("Image Agent: %d query | style=%s",
                    len(rec.google_search_queries), rec.image_style)
        return rec
    except Exception as e:
        logger.error("Image Agent LLM error: %s — pakai fallback", e)
        return _fallback(search_query_for_image or trending_topic)


# ------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------

def _build_prompt(tweet_text: str, trending_topic: str, seed_query: str) -> str:
    return f"""Tweet salaryman:
{tweet_text}

Trending topic : {trending_topic}
Kata kunci awal : {seed_query or '(tidak ada)'}

Sarankan gambar pendukung. Berikan TEPAT 3 query Google Images:
- 2 query dalam bahasa Inggris
- 1 query dalam bahasa Jepang
Semua harus SFW, TANPA wajah orang nyata, dan sesuai mood salaryman
(mis. kereta penuh, meja kantor, ramen tengah malam, langit Tokyo).

Balas HANYA JSON valid:
{{
  "image_description": "deskripsi singkat gambar ideal",
  "google_search_queries": ["english query 1", "english query 2", "日本語クエリ"],
  "image_style": "mis. 'foto realistis', 'ilustrasi flat', 'aesthetic moody'",
  "reason": "kenapa gambar ini cocok dengan tweet"
}}"""


def _parse(raw: str, seed_query: str) -> ImageRec:
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("JSON tidak ditemukan dalam respons LLM")
    data = json.loads(raw[start:end])

    queries = data.get("google_search_queries") or []
    if not isinstance(queries, list):
        queries = [str(queries)]
    queries = [str(q).strip() for q in queries if str(q).strip()][:3]
    if not queries:
        raise ValueError("google_search_queries kosong")

    return ImageRec(
        image_description=(data.get("image_description") or "").strip(),
        google_search_queries=queries,
        image_style=(data.get("image_style") or "").strip(),
        reason=(data.get("reason") or "").strip(),
    )


def _fallback(seed_query: str) -> ImageRec:
    """Rekomendasi darurat dari kata kunci awal (tanpa LLM)."""
    seed = seed_query or "salaryman Tokyo"
    return ImageRec(
        image_description=f"Gambar bertema '{seed}' bernuansa salaryman",
        google_search_queries=[
            f"{seed} aesthetic no people",
            "tired salaryman commute train illustration",
            "サラリーマン 日常 イラスト",
        ],
        image_style="aesthetic moody",
        reason="fallback — LLM tidak tersedia",
        is_fallback=True,
    )
