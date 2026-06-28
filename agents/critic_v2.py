"""
agents/critic_v2.py — CRITIC AGENT (salaryman).

Input : TweetDraft
Output: CriticResult

Menilai tweet salaryman pada 4 dimensi (skala 1–10):
  relatability | naturalness | engagement | topic_fit

Aturan: jika SALAH SATU dimensi < min_dimension_score (default 6) →
verdict REJECT + improved_tweet (rewrite otomatis). total_score = rata-rata
keempat dimensi (dibulatkan, skala /10).
"""
import json
from typing import Dict, List

from core.config import load_config
from core.llm import chat
from core.logger import get_logger
from core.models import CriticResult, TweetDraft

logger = get_logger("critic")

DIMENSIONS = ("relatability", "naturalness", "engagement", "topic_fit")

_DIM_DESC = {
    "relatability": "Seberapa 'ini gue banget' untuk salaryman Jepang",
    "naturalness":  "Terdengar seperti manusia ngetik santai, bukan berita/AI",
    "engagement":   "Potensi like/RT/reply; bikin orang pengen bales 'わかる'",
    "topic_fit":    "Nyambung dengan trending topic & persona salaryman",
}


def _min_score() -> int:
    return load_config().get("scoring", {}).get("min_dimension_score", 6)


def review(draft: TweetDraft) -> CriticResult:
    """Nilai TweetDraft pada 4 dimensi; rewrite otomatis bila ada dimensi < min."""
    prompt = _build_prompt(draft, _min_score())
    try:
        raw = chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
        )
        result = _parse(raw, _min_score())
        logger.info("Critic: total=%d/10 | %s | verdict=%s",
                    result.score,
                    " ".join(f"{k}={result.breakdown.get(k,0)}" for k in DIMENSIONS),
                    result.verdict)
        return result
    except Exception as e:
        logger.error("Critic LLM error: %s — pakai fallback", e)
        return _fallback()


# ------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------

def _build_prompt(draft: TweetDraft, min_score: int) -> str:
    dims = "\n".join(f'  "{k}" (1–10): {_DIM_DESC[k]}' for k in DIMENSIONS)
    return f"""Kamu editor konten Twitter/X untuk akun salaryman Jepang.
Nilai tweet di bawah secara kritis pada 4 dimensi (skala 1–10).

=== TWEET ===
{draft.japanese}

=== KONTEKS ===
Topik   : {draft.topic}
Kategori: {draft.angle_type}
Panjang : {len(draft.japanese)} karakter

=== DIMENSI ===
{dims}

=== ATURAN ===
• Beri skor 1–10 tiap dimensi (jangan menggerombol di tengah).
• Jika SALAH SATU dimensi < {min_score}: verdict = "REJECT" dan WAJIB tulis
  "improved_tweet" — versi perbaikan (tetap salaryman, ≤140 char, ada hashtag
  #サラリーマン atau #あるある).
• Jika SEMUA dimensi ≥ {min_score}: verdict = "APPROVE", "improved_tweet" boleh "".

Balas HANYA JSON valid:
{{
  "scores": {{
    "relatability": <1-10>,
    "naturalness": <1-10>,
    "engagement": <1-10>,
    "topic_fit": <1-10>
  }},
  "verdict": "APPROVE" atau "REJECT",
  "improved_tweet": "versi perbaikan jika REJECT, selain itu kosong",
  "feedback": "1-2 kalimat masukan utama"
}}"""


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------

def _parse(raw: str, min_score: int) -> CriticResult:
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("JSON tidak ditemukan dalam respons LLM")

        data = json.loads(raw[start:end])
        scores_raw = data.get("scores", {})

        breakdown: Dict[str, int] = {}
        for dim in DIMENSIONS:
            try:
                val = int(round(float(scores_raw.get(dim, 0))))
            except (TypeError, ValueError):
                val = 0
            breakdown[dim] = max(0, min(val, 10))

        total = round(sum(breakdown.values()) / len(DIMENSIONS))
        all_pass = all(v >= min_score for v in breakdown.values())

        # Verdict dari LLM dihormati, tapi aturan ambang yang menentukan akhir
        verdict = "APPROVE" if all_pass else "REJECT"

        feedback = (data.get("feedback") or "").strip()
        improved = (data.get("improved_tweet") or "").strip()
        if len(improved) > 140:
            improved = improved[:140]

        return CriticResult(
            score=total,
            verdict=verdict,
            breakdown=breakdown,
            improved_tweet=improved,
            feedback=feedback,
            issues=[feedback] if feedback else [],
            suggestions=[improved] if improved else [],
        )

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Critic JSON parse gagal: %s", e)
        return _fallback()


# ------------------------------------------------------------------
# Fallback
# ------------------------------------------------------------------

def _fallback() -> CriticResult:
    """Skor 0 + is_fallback=True → orchestrator men-skip (bukan kirim sampah)."""
    logger.error("Critic FALLBACK aktif — LLM tidak menilai. Konten akan di-skip.")
    return CriticResult(
        score=0,
        verdict="REJECT",
        breakdown={k: 0 for k in DIMENSIONS},
        improved_tweet="",
        feedback="Tidak dapat mengevaluasi (LLM tidak tersedia)",
        issues=["Tidak dapat mengevaluasi (LLM tidak tersedia)"],
        suggestions=["Periksa LLM_API_KEY / LLM_BASE_URL / LLM_MODEL di .env"],
        is_fallback=True,
    )
