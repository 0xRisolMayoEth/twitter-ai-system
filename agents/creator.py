"""
agents/creator.py — WRITER AGENT (田中サトル, salaryman Tokyo).

Input : StrategistOutput (trending topic + why_relatable)
Output: TweetDraft (tweet_text JP 80–140 char + terjemahan ID)

Menulis tweet Jepang gaya salaryman sehari-hari (口語体) yang menghubungkan
trending topic ke pengalaman pribadi salaryman, diakhiri hashtag
#サラリーマン atau #あるある. Bukan berita, bukan promosi.
"""
import json
from typing import Optional

from core.config import load_config
from core.llm import chat
from core.logger import get_logger
from core.models import StrategistOutput, TweetDraft

logger = get_logger("writer")

JP_MAX = 140   # batas atas karakter tweet Jepang
JP_MIN = 80    # batas bawah ideal
ID_MAX = 280


def create_tweet(strategy: StrategistOutput) -> TweetDraft:
    """Tulis tweet salaryman JP + terjemahan ID. Retry 1x bila JP > 140."""
    persona = load_config().get("persona", {})

    prompt = _build_prompt(strategy, persona)
    try:
        raw = chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=450,
            temperature=0.9,
        )
        draft = _parse(raw, strategy, truncate=False)

        if len(draft.japanese) > JP_MAX:
            logger.info("JP %d chars > %d, retry lebih ketat", len(draft.japanese), JP_MAX)
            raw2 = chat(
                messages=[{"role": "user", "content": _build_retry_prompt(strategy, persona, draft.japanese)}],
                max_tokens=350,
                temperature=0.7,
            )
            draft = _parse(raw2, strategy, original=draft, truncate=True)

        logger.info("Writer: JP=%d chars | ID=%d chars | tone=%s",
                    len(draft.japanese), len(draft.indonesian), draft.tone)
        return draft

    except Exception as e:
        logger.error("Writer LLM error: %s — pakai fallback draft", e)
        return _fallback(strategy)


# ------------------------------------------------------------------
# Prompt builders
# ------------------------------------------------------------------

def _build_prompt(strategy: StrategistOutput, persona: dict) -> str:
    name = persona.get("name", "田中サトル")
    desc = persona.get("description", "salaryman Tokyo biasa")

    return f"""Kamu adalah {name}.
{desc}

Kamu baru lihat trending topic ini dan ingin nge-tweet pengalaman/perasaanmu
sebagai salaryman — seolah ngobrol sama rekan kerja, bukan baca berita.

TRENDING TOPIC : {strategy.topic}
KATEGORI       : {strategy.topic_category}
KENAPA NGENA   : {strategy.why_relatable}

=== ATURAN TWEET JEPANG (tweet_text) ===
• Persona  : 田中サトル, 32th, salaryman Tokyo, commute 1 jam, sering lembur
• Gaya     : 口語体 (bahasa sehari-hari), personal story, JUJUR & santai
• WAJIB    : hubungkan trending topic ke pengalaman pribadi salaryman
• Panjang  : 80–140 karakter Jepang (tiap karakter = 1)
• Emoji    : pakai 1–3 emoji yang pas (😮‍💨😴🍜💸🚃 dll)
• Hashtag  : akhiri dengan #サラリーマン ATAU #あるある (pilih satu)
• DILARANG : terdengar seperti berita, pengumuman, atau promosi
• DILARANG : politik, bencana, kontroversi

=== CONTOH NADA ===
✗ 「○○が話題になっています」(seperti berita)
✓ 「満員電車で○○のニュース見て、思わず吐きそうになった😮‍💨 #サラリーマン」
✓ 「今日も残業。○○とか言われても、こっちは定時で帰りたいだけなんだ😴 #あるある」

=== TERJEMAHAN INDONESIA (indonesian) ===
• Santai, mengalir, semangat yang SAMA dengan versi JP (bukan terjemahan kaku)
• Maks {ID_MAX} karakter

=== OUTPUT ===
Balas HANYA JSON valid:
{{
  "tweet_text": "日本語のツイート (80〜140文字、ハッシュタグで終わる)",
  "indonesian": "terjemahan/parafrase santai versi Indonesia",
  "tone": "deskripsi singkat nada (mis. 'capek tapi lucu')",
  "best_posting_time": "saran waktu posting (mis. '通勤中', '昼休み', '退勤後')"
}}"""


def _build_retry_prompt(strategy: StrategistOutput, persona: dict, too_long: str) -> str:
    return (
        _build_prompt(strategy, persona)
        + f"\n\n⚠️ Versi sebelumnya {len(too_long)} karakter — TERLALU PANJANG (maks {JP_MAX}).\n"
        f"Sebelumnya: {too_long}\nTulis ulang ≤{JP_MAX} karakter, pertahankan inti & hashtag."
    )


# ------------------------------------------------------------------
# Parsing & validation
# ------------------------------------------------------------------

def _parse(raw: str, strategy: StrategistOutput,
           original: Optional[TweetDraft] = None, truncate: bool = True) -> TweetDraft:
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("JSON tidak ditemukan dalam output LLM")

        data = json.loads(raw[start:end])
        jp = (data.get("tweet_text") or data.get("japanese") or "").strip()
        indo = (data.get("indonesian") or "").strip()
        if not jp:
            raise ValueError("field 'tweet_text' kosong")

        if truncate:
            if len(jp) > JP_MAX:
                logger.warning("JP masih %d chars setelah retry, dipotong", len(jp))
                jp = jp[:JP_MAX]
            if len(indo) > ID_MAX:
                indo = indo[:ID_MAX]

        return TweetDraft(
            japanese=jp,
            indonesian=indo or jp,
            topic=strategy.topic,
            angle_type=strategy.angle_type,
            source_url=strategy.source_url,
            tone=(data.get("tone") or "").strip(),
            best_posting_time=(data.get("best_posting_time") or "").strip(),
        )

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Writer JSON parse gagal (%s), pakai %s",
                       e, "original" if original else "fallback")
        return original if original else _fallback(strategy)


# ------------------------------------------------------------------
# Fallback
# ------------------------------------------------------------------

def _fallback(strategy: StrategistOutput) -> TweetDraft:
    """Draft darurat (is_fallback=True) agar orchestrator men-skip, bukan kirim."""
    jp = f"【{strategy.topic}】 #サラリーマン"
    if len(jp) > JP_MAX:
        jp = jp[:JP_MAX]
    return TweetDraft(
        japanese=jp,
        indonesian=f"(fallback) {strategy.topic}",
        topic=strategy.topic,
        angle_type=strategy.angle_type,
        source_url=strategy.source_url,
        is_fallback=True,
    )
