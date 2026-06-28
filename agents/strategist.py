"""
agents/strategist.py — TREND AGENT (salaryman).

Input : List[TrendCandidate]  (kandidat mentah dari TrendScout/RSS)
Output: List[StrategistOutput] (terurut: paling relatable untuk salaryman dulu)

Tugas: dari banyak kandidat trending, pilih topik yang RELATABLE untuk
salaryman Jepang (commute, lembur, makan siang, gaji, rapat, bos, dll) dan
HINDARI politik, bencana, dan kontroversi. Untuk tiap pilihan, hasilkan:
  { trending_topic, topic_category, why_relatable, search_query_for_image }
"""
import json
from typing import List

from core.config import load_config
from core.llm import chat
from core.logger import get_logger
from core.models import TrendCandidate, StrategistOutput

logger = get_logger("trend_agent")

# Kategori relatable untuk salaryman (panduan, bukan pembatas keras)
SALARY_CATEGORIES = (
    "commute", "overtime", "lunch", "paycheck", "meeting",
    "boss", "coworker", "weekend", "morning", "tired", "money",
)

# Topik yang HARUS dihindari
BANNED = ("politik", "politics", "選挙", "政治", "bencana", "disaster",
          "震災", "地震", "事故", "death", "tewas", "kontroversi", "controversy")

_SYSTEM = (
    "Kamu TREND AGENT untuk akun Twitter/X salaryman Jepang. Kamu memilih "
    "trending topic yang paling 'ini gue banget' untuk salaryman biasa, dan "
    "menolak topik politik, bencana, atau kontroversi."
)


def select_trends(candidates: List[TrendCandidate]) -> List[StrategistOutput]:
    """
    Pilih & ranking topik salaryman-relatable dari kandidat.
    Mengembalikan list (boleh kosong jika tidak ada yang cocok).
    """
    if not candidates:
        return []

    # Saring kasar topik terlarang sebelum ke LLM
    safe = [c for c in candidates if not _looks_banned(c.topic)]
    if not safe:
        logger.warning("Semua kandidat kena filter terlarang")
        return []

    prompt = _build_prompt(safe)
    try:
        raw = chat(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.6,
        )
        picks = _parse(raw, safe)
        if picks:
            logger.info("TREND AGENT pilih %d topik relatable (top: %s)",
                        len(picks), picks[0].topic[:50])
            return picks
    except Exception as e:
        logger.error("TREND AGENT LLM error: %s — pakai fallback", e)

    return _fallback(safe)


# ------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------

def _build_prompt(candidates: List[TrendCandidate]) -> str:
    niche = load_config().get("niche", "")
    listing = "\n".join(
        f"{i+1}. {c.topic}" + (f"  ({c.raw_summary[:80]})" if c.raw_summary else "")
        for i, c in enumerate(candidates[:15])
    )
    cats = ", ".join(SALARY_CATEGORIES)
    return f"""Niche akun: {niche}

Berikut daftar trending topic. Pilih SAMPAI 3 yang paling RELATABLE untuk
salaryman Jepang biasa (urut dari paling relatable). Untuk tiap pilihan,
hubungkan ke kehidupan salaryman.

KATEGORI relatable (pilih salah satu yang paling pas): {cats}

HINDARI total: politik, pemilu, bencana, kecelakaan, kematian, kontroversi.
Kalau tidak ada yang benar-benar relatable, kembalikan list kosong.

DAFTAR TOPIK:
{listing}

Balas HANYA JSON valid:
{{
  "picks": [
    {{
      "trending_topic": "<judul topik yang dipilih dari daftar>",
      "topic_category": "<salah satu kategori relatable>",
      "why_relatable": "<1 kalimat: kenapa ini ngena untuk salaryman>",
      "search_query_for_image": "<kata kunci singkat untuk cari gambar pendukung>"
    }}
  ]
}}"""


def _parse(raw: str, candidates: List[TrendCandidate]) -> List[StrategistOutput]:
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("JSON tidak ditemukan dalam respons LLM")
    data = json.loads(raw[start:end])
    picks = data.get("picks") or []
    if not isinstance(picks, list):
        raise ValueError("'picks' bukan list")

    # Map topik → url sumber agar source_url tetap terisi
    url_by_topic = {c.topic: c.url for c in candidates}

    out: List[StrategistOutput] = []
    for p in picks:
        if not isinstance(p, dict):
            continue
        topic = (p.get("trending_topic") or "").strip()
        if not topic or _looks_banned(topic):
            continue
        category = (p.get("topic_category") or "relatable").strip()
        why = (p.get("why_relatable") or "").strip()
        out.append(StrategistOutput(
            topic=topic,
            source_url=_best_url(topic, url_by_topic),
            topic_category=category,
            why_relatable=why,
            search_query_for_image=(p.get("search_query_for_image") or topic).strip(),
            angle_type=category,            # disimpan ke DB
            angle_description=why,
            reasoning="dipilih TREND AGENT (salaryman-relatable)",
        ))
    return out[:3]


def _fallback(candidates: List[TrendCandidate]) -> List[StrategistOutput]:
    """Tanpa LLM: ambil beberapa kandidat teratas apa adanya."""
    out: List[StrategistOutput] = []
    for c in candidates[:3]:
        out.append(StrategistOutput(
            topic=c.topic,
            source_url=c.url,
            topic_category="relatable",
            why_relatable="(fallback) topik umum salaryman",
            search_query_for_image=c.topic,
            angle_type="relatable",
            angle_description="fallback — LLM tidak tersedia",
            reasoning="fallback",
        ))
    return out


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _looks_banned(text: str) -> bool:
    low = text.lower()
    return any(b in low for b in BANNED)


def _best_url(topic: str, url_by_topic: dict) -> str:
    if topic in url_by_topic:
        return url_by_topic[topic]
    # cocokkan longgar bila LLM sedikit mengubah judul
    for t, u in url_by_topic.items():
        if topic[:20] and topic[:20] in t:
            return u
    return ""
