"""
agents/reviser.py — Reviser agent.

Input : TweetDraft + CriticResult
Output: TweetDraft (versi yang diperbaiki)

Menulis ulang tweet JP+ID berdasarkan umpan balik Critic yang spesifik.
Jika LLM gagal atau versi baru lebih buruk (JP kosong), kembalikan draft asli.
"""
import json
import logging
from typing import Optional

from core.config import load_config
from core.llm import chat
from core.logger import get_logger
from core.models import CriticResult, TweetDraft

logger = get_logger("reviser")

JP_MAX = 140
ID_MAX = 280


def revise(draft: TweetDraft, critic: CriticResult) -> TweetDraft:
    """
    Tulis ulang draft berdasarkan umpan balik Critic.
    Kembalikan draft asli jika revisi gagal (fail-safe).
    """
    cfg = load_config()
    persona = cfg.get("persona", {})
    tweet_register = cfg.get("tweet_register", "casual")

    prompt = _build_prompt(draft, critic, persona, tweet_register)

    try:
        raw = chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=450,
            temperature=0.75,
        )
        revised = _parse(raw, draft)

        logger.info(
            "Reviser: JP=%d chars | skor sebelum=%d | issues_fixed=%d",
            len(revised.japanese), critic.score, len(critic.issues),
        )
        return revised

    except Exception as e:
        logger.error("Reviser LLM error: %s — pertahankan draft asli", e)
        return draft


# ------------------------------------------------------------------
# Prompt builder
# ------------------------------------------------------------------

def _build_prompt(
    draft: TweetDraft,
    critic: CriticResult,
    persona: dict,
    tweet_register: str,
) -> str:
    cfg = load_config()
    niche = cfg.get("niche", "")

    issues_str = "\n".join(f"  - {i}" for i in critic.issues) if critic.issues else "  (tidak ada)"
    suggestions_str = "\n".join(f"  - {s}" for s in critic.suggestions) if critic.suggestions else "  (tidak ada)"

    # Aspek dengan skor rendah (< 70% dari maks)
    from agents.critic_v2 import _DEFAULT_WEIGHTS
    weights = cfg.get("scoring", {}).get("weights", _DEFAULT_WEIGHTS)
    weak_aspects = [
        f"{k} ({v}/{weights.get(k, 1)} poin)"
        for k, v in critic.breakdown.items()
        if weights.get(k, 1) > 0 and v / weights.get(k, 1) < 0.70
    ]
    weak_str = ", ".join(weak_aspects) if weak_aspects else "tidak ada"

    jp_register = (
        "futsū-form (bahasa sehari-hari netizen Jepang). "
        "Boleh pakai partikel akhir ね/よ/けど, sesekali w/笑. Hindari です/ます berlebihan."
        if tweet_register == "casual"
        else "keigo yang sopan tapi mudah dipahami, hindari terlalu formal."
    )

    persona_name = persona.get("name", "JP Content Creator")

    return f"""Kamu adalah {persona_name}, editor konten Twitter/X berbahasa Jepang.

Tweet berikut mendapat skor {critic.score}/100 dari reviewer.
Tugasmu: tulis ulang tweet ini dengan memperbaiki masalah yang ditemukan.

=== TWEET ASLI ===
🇯🇵 JP : {draft.japanese}
🇮🇩 ID : {draft.indonesian}

=== MASALAH YANG HARUS DIPERBAIKI ===
{issues_str}

=== SARAN SPESIFIK ===
{suggestions_str}

=== ASPEK YANG LEMAH ===
{weak_str}

=== KONTEKS KONTEN ===
Topik  : {draft.topic}
Angle  : {draft.angle_type}
Niche  : {niche}

=== ATURAN WAJIB ===
• JP register : {jp_register}
• JP panjang  : WAJIB ≤ {JP_MAX} karakter
• ID gaya     : Santai-cerdas, natural, bukan terjemahan kata-per-kata
• Pertahankan INTI pesan dan angle, hanya perbaiki aspek yang lemah
• JANGAN ubah topik atau angle menjadi sesuatu yang berbeda

Balas HANYA JSON valid:
{{
  "japanese": "versi JP yang sudah diperbaiki (≤{JP_MAX} char)",
  "indonesian": "versi ID yang sudah diperbaiki"
}}"""


# ------------------------------------------------------------------
# Parsing & validation
# ------------------------------------------------------------------

def _parse(raw: str, original: TweetDraft) -> TweetDraft:
    """Parse JSON revisi dari LLM. Kembalikan original jika parsing gagal."""
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("JSON tidak ditemukan dalam output LLM")

        data = json.loads(raw[start:end])
        jp   = (data.get("japanese")   or "").strip()
        indo = (data.get("indonesian") or "").strip()

        if not jp:
            logger.warning("Reviser: field 'japanese' kosong, pakai draft asli")
            return original

        # Safety truncation setelah revisi
        if len(jp) > JP_MAX:
            logger.warning("Reviser JP masih %d chars, dipotong ke %d", len(jp), JP_MAX)
            jp = jp[:JP_MAX]
        if len(indo) > ID_MAX:
            indo = indo[:ID_MAX]

        return TweetDraft(
            japanese=jp,
            indonesian=indo or original.indonesian,
            topic=original.topic,
            angle_type=original.angle_type,
            source_url=original.source_url,
        )

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Reviser JSON parse gagal (%s), pakai draft asli", e)
        return original
