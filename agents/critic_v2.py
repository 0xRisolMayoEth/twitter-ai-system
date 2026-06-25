"""
agents/critic_v2.py — Critic agent (rubrik 0-100).

Input : TweetDraft
Output: CriticResult

Mengevaluasi tweet JP pada 6 aspek dengan bobot sesuai config:
  hook (25) | engagement (20) | naturalness (20)
  japanese_quality (15) | relevance (10) | format (10)

Skor per aspek dijumlahkan menjadi total 0-100.
"""
import json
import logging
from typing import Dict, List

from core.config import load_config
from core.llm import chat
from core.logger import get_logger
from core.models import CriticResult, TweetDraft

logger = get_logger("critic_v2")

# Rubrik default jika config tidak tersedia
_DEFAULT_WEIGHTS: Dict[str, int] = {
    "hook":             25,
    "engagement":       20,
    "naturalness":      20,
    "japanese_quality": 15,
    "relevance":        10,
    "format":           10,
}

_ASPECT_DESC: Dict[str, str] = {
    "hook":             "Kalimat pertama menghentikan scroll, langsung menarik perhatian",
    "engagement":       "Potensi mendapat RT/like/reply; memancing interaksi",
    "naturalness":      "Terdengar seperti ditulis manusia, bukan template AI",
    "japanese_quality": "Grammar, kosakata, dan register bahasa Jepang benar & natural",
    "relevance":        "Relevan dengan niche dan angle yang dipilih",
    "format":           "Panjang ≤140 char, emoji 0-1, hashtag 0-2, tidak ada karakter aneh",
}


def _verdict(score: int) -> str:
    """Petakan skor 0-100 ke verdict. Dipakai untuk tag Telegram & alur revisi."""
    if score >= 90:
        return "APPROVED"   # ✅ luar biasa
    if score >= 75:
        return "GOOD"       # ⚡ bagus, layak kirim
    if score >= 60:
        return "REVISE"     # ✏️ perlu perbaikan
    return "REJECT"         # ✗ tolak, ganti topik


def review(draft: TweetDraft) -> CriticResult:
    """Evaluasi TweetDraft dan kembalikan CriticResult (skor 0-100)."""
    cfg = load_config()
    weights = cfg.get("scoring", {}).get("weights", _DEFAULT_WEIGHTS)
    niche = cfg.get("niche", "")

    prompt = _build_prompt(draft, weights, niche)

    try:
        raw = chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,  # evaluasi harus konsisten, bukan kreatif
        )
        result = _parse(raw, weights)
        logger.info(
            "Critic: total=%d | hook=%d eng=%d nat=%d jpq=%d rel=%d fmt=%d",
            result.score,
            result.breakdown.get("hook", 0),
            result.breakdown.get("engagement", 0),
            result.breakdown.get("naturalness", 0),
            result.breakdown.get("japanese_quality", 0),
            result.breakdown.get("relevance", 0),
            result.breakdown.get("format", 0),
        )
        return result

    except Exception as e:
        logger.error("Critic LLM error: %s — pakai fallback score", e)
        return _fallback(weights)


# ------------------------------------------------------------------
# Prompt builder
# ------------------------------------------------------------------

def _build_prompt(draft: TweetDraft, weights: Dict[str, int], niche: str) -> str:
    aspects_str = "\n".join(
        f'  "{k}" (maks {weights.get(k, _DEFAULT_WEIGHTS.get(k, 0))} poin): {_ASPECT_DESC[k]}'
        for k in _ASPECT_DESC
    )
    return f"""Kamu adalah editor konten Twitter/X yang ahli dalam konten berbahasa Jepang.

Evaluasi tweet Jepang di bawah ini secara objektif dan kritis.

=== TWEET ===
{draft.japanese}

=== KONTEKS ===
Topik  : {draft.topic}
Angle  : {draft.angle_type} — {draft.angle_description if hasattr(draft, 'angle_description') else ''}
Niche  : {niche}
Panjang: {len(draft.japanese)} karakter (maks 140)

=== ASPEK PENILAIAN ===
{aspects_str}

=== INSTRUKSI PENILAIAN (BERSIKAP KETAT) ===
Beri skor per aspek sesuai maksimum masing-masing (BUKAN skala 0-10).
"hook" maks 25, "engagement" maks 20, dst.

Patokan ketat — JANGAN menggerombol di skor tengah:
• Jika hook hanya menyalin/menerjemahkan judul berita → hook MAKS 8 dari 25.
• Jika tweet tidak punya opini/reaksi/insight (cuma menyampaikan fakta)
  → naturalness & engagement MASING-MASING maks setengah.
• Jika nada terdengar seperti koran/korporat (「発表された」「非常に」)
  → japanese_quality & naturalness dipotong drastis.
• Tweet yang benar-benar berkarakter & memancing reaksi BOLEH dapat 90+.
Gunakan SELURUH rentang 0-100. Bedakan tweet biasa (60-an) dari yang bagus (80+).

Identifikasi 1-3 masalah konkret dan 1-3 saran SPESIFIK & actionable
(sebut bagian mana & harus diubah jadi apa — bukan "buat lebih menarik").

Balas HANYA JSON valid:
{{
  "breakdown": {{
    "hook": <0-25>,
    "engagement": <0-20>,
    "naturalness": <0-20>,
    "japanese_quality": <0-15>,
    "relevance": <0-10>,
    "format": <0-10>
  }},
  "issues": ["masalah konkret 1", "masalah konkret 2"],
  "suggestions": ["saran spesifik 1", "saran spesifik 2"]
}}"""


# ------------------------------------------------------------------
# Parsing & validation
# ------------------------------------------------------------------

def _parse(raw: str, weights: Dict[str, int]) -> CriticResult:
    """Parse JSON dari LLM, hitung total, validasi tiap aspek."""
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("JSON tidak ditemukan dalam respons LLM")

        data = json.loads(raw[start:end])
        breakdown_raw: dict = data.get("breakdown", {})

        # Validasi dan clamp tiap aspek ke [0, max_weight]
        breakdown: Dict[str, int] = {}
        for aspect, max_w in weights.items():
            raw_val = breakdown_raw.get(aspect, 0)
            try:
                val = int(round(float(raw_val)))
            except (TypeError, ValueError):
                val = 0
            breakdown[aspect] = max(0, min(val, max_w))

        total = sum(breakdown.values())

        issues: List[str] = data.get("issues") or []
        suggestions: List[str] = data.get("suggestions") or []

        if not isinstance(issues, list):
            issues = [str(issues)]
        if not isinstance(suggestions, list):
            suggestions = [str(suggestions)]

        return CriticResult(
            score=total,
            verdict=_verdict(total),
            breakdown=breakdown,
            issues=issues[:5],
            suggestions=suggestions[:5],
        )

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Critic JSON parse gagal: %s", e)
        return _fallback(weights)


# ------------------------------------------------------------------
# Fallback
# ------------------------------------------------------------------

def _fallback(weights: Dict[str, int]) -> CriticResult:
    """
    Hasil fallback saat LLM gagal. Skor 0 + is_fallback=True agar
    orchestrator memperlakukannya sebagai kegagalan (skip), BUKAN konten
    layak kirim. (Dulu mengembalikan 73 sehingga sampah ikut terkirim.)
    """
    logger.error("Critic FALLBACK aktif — LLM tidak menghasilkan penilaian. Konten akan di-skip.")
    return CriticResult(
        score=0,
        verdict="REJECT",
        breakdown={k: 0 for k in weights},
        issues=["Tidak dapat mengevaluasi (LLM tidak tersedia)"],
        suggestions=["Periksa LLM_API_KEY / LLM_BASE_URL / LLM_MODEL di .env"],
        is_fallback=True,
    )
