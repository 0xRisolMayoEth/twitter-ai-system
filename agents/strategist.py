"""
agents/strategist.py — Strategist agent.

Input : TrendCandidate
Output: StrategistOutput

Memilih satu dari 5 pola angle via LLM dengan mempertimbangkan
variasi angle yang sudah dipakai (diambil dari DB) agar 20 konten/hari
tidak monoton.
"""
import json
import logging
from collections import Counter
from typing import List

from core.config import load_config
from core.llm import chat
from core.logger import get_logger
from core.models import TrendCandidate, StrategistOutput
from database.db_manager import get_recent_angle_types

logger = get_logger("strategist")

# 5 pola angle yang tersedia — sesuai spec Bagian 5.2
ANGLE_TYPES: dict[str, str] = {
    "curiosity_gap": "Hook pertanyaan / misteri — bikin penasaran sebelum reveal jawaban",
    "contrarian":    "Opini yang melawan arus — unexpected take yang bikin orang berpikir ulang",
    "relatable":     "Momen atau perasaan yang langsung beresonansi: 'ini gue banget'",
    "news_insight":  "Fakta utama berita + satu insight yang tidak ada di headline biasa",
    "listicle":      "Mini list atau fakta mengejutkan (2–3 poin) yang mudah dibaca & di-share",
}

_SYSTEM = (
    "Kamu adalah Strategist untuk akun Twitter/X berbahasa Jepang. "
    "Tugasmu memilih angle konten yang akan menghasilkan engagement tertinggi "
    "berdasarkan topik, niche, dan audiens target."
)


def pick_angle(trend: TrendCandidate) -> StrategistOutput:
    """
    Pilih angle terbaik untuk TrendCandidate yang diberikan via LLM.
    Rotasi otomatis berdasarkan history angle 10 run terakhir.
    """
    cfg = load_config()
    niche = cfg.get("niche", "")

    recent = get_recent_angle_types(limit=10)
    overused = _detect_overused(recent)

    prompt = _build_prompt(trend, niche, recent, overused)

    try:
        raw = chat(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=350,
            temperature=0.7,
        )
        result = _parse(raw, trend)
        logger.info(
            "Strategist → angle='%s': %s",
            result.angle_type, result.angle_description[:70],
        )
        return result

    except Exception as e:
        logger.error("Strategist LLM error: %s — pakai fallback angle", e)
        return _fallback(trend, recent)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _detect_overused(recent: List[str]) -> List[str]:
    """Tandai angle yang muncul ≥2 kali dalam 10 run terakhir sebagai 'overused'."""
    counts = Counter(recent)
    return [a for a, c in counts.items() if c >= 2]


def _build_prompt(
    trend: TrendCandidate,
    niche: str,
    recent: List[str],
    overused: List[str],
) -> str:
    angles_str = "\n".join(f'  "{k}": {v}' for k, v in ANGLE_TYPES.items())
    recent_str  = ", ".join(recent[-5:]) if recent else "belum ada"
    avoid_str   = ", ".join(overused)    if overused else "tidak ada"

    return f"""Topik  : {trend.topic}
Sumber : {trend.source}
Summary: {trend.raw_summary or '(kosong)'}
Niche  : {niche}
Audiens: pengguna Twitter/X Jepang

Angle yang tersedia:
{angles_str}

5 angle terakhir yang dipakai: {recent_str}
Angle yang SEBAIKNYA DIHINDARI (overused): {avoid_str}

Pilih SATU angle paling cocok. Pertimbangkan:
1. Kesesuaian topik dengan angle
2. Potensi engagement & virality
3. Variasi (hindari yang overused)

Balas HANYA JSON valid (tidak ada teks lain):
{{
  "angle_type": "salah_satu_dari_{list(ANGLE_TYPES.keys())}",
  "angle_description": "deskripsi singkat angle (1 kalimat, bahasa Indonesia)",
  "reasoning": "alasan singkat kenapa angle ini paling cocok"
}}"""


def _parse(raw: str, trend: TrendCandidate) -> StrategistOutput:
    """Parse JSON dari LLM; fallback jika format salah."""
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("JSON tidak ditemukan dalam respons LLM")
        data = json.loads(raw[start:end])

        angle_type = data.get("angle_type", "news_insight")
        if angle_type not in ANGLE_TYPES:
            logger.warning("angle_type '%s' tidak dikenal, pakai 'news_insight'", angle_type)
            angle_type = "news_insight"

        return StrategistOutput(
            topic=trend.topic,
            source_url=trend.url,
            angle_type=angle_type,
            angle_description=data.get("angle_description", ANGLE_TYPES[angle_type]),
            reasoning=data.get("reasoning", ""),
        )
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Strategist JSON parse gagal: %s", e)
        return _fallback(trend, [])


def _fallback(trend: TrendCandidate, recent: List[str]) -> StrategistOutput:
    """Pilih angle fallback dengan rotasi sederhana (tanpa LLM)."""
    counts = Counter(recent)
    angle = min(ANGLE_TYPES.keys(), key=lambda a: counts.get(a, 0))
    return StrategistOutput(
        topic=trend.topic,
        source_url=trend.url,
        angle_type=angle,
        angle_description=ANGLE_TYPES[angle],
        reasoning="fallback — LLM tidak tersedia",
    )
