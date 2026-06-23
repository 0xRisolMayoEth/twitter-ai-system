"""
agents/creator.py — Creator agent.

Input : StrategistOutput
Output: TweetDraft (bilingual JP + ID)

Menulis tweet JP (futsū-form, ≤140 char) dan ID (santai-cerdas)
berdasarkan angle dari Strategist. Memvalidasi panjang dan retry
sekali jika JP terlalu panjang.
"""
import json
import logging
from typing import Optional

from core.config import load_config
from core.llm import chat
from core.logger import get_logger
from core.models import StrategistOutput, TweetDraft

logger = get_logger("creator")

JP_MAX = 140   # batas karakter tweet Jepang
ID_MAX = 280   # batas karakter tweet Indonesia


def create_tweet(strategy: StrategistOutput) -> TweetDraft:
    """
    Tulis tweet JP + ID berdasarkan strategy dari Strategist.
    Retry 1x otomatis jika JP terlalu panjang.
    """
    cfg = load_config()
    persona       = cfg.get("persona", {})
    tweet_register = cfg.get("tweet_register", "casual")

    prompt = _build_prompt(strategy, persona, tweet_register)

    try:
        raw = chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=450,
            temperature=0.85,
        )
        # truncate=False: preserve overlong JP so the retry check below can fire
        draft = _parse(raw, strategy, truncate=False)

        # Retry sekali jika JP terlalu panjang
        if len(draft.japanese) > JP_MAX:
            logger.info(
                "JP terlalu panjang (%d chars), retry dengan constraint lebih ketat",
                len(draft.japanese),
            )
            retry_prompt = _build_retry_prompt(strategy, persona, tweet_register, draft.japanese)
            raw2 = chat(
                messages=[{"role": "user", "content": retry_prompt}],
                max_tokens=350,
                temperature=0.6,
            )
            # truncate=True: safety net after retry
            draft = _parse(raw2, strategy, original=draft, truncate=True)

        logger.info(
            "Creator: JP=%d chars | ID=%d chars | angle=%s",
            len(draft.japanese), len(draft.indonesian), strategy.angle_type,
        )
        return draft

    except Exception as e:
        logger.error("Creator LLM error: %s — pakai fallback draft", e)
        return _fallback(strategy)


# ------------------------------------------------------------------
# Prompt builders
# ------------------------------------------------------------------

def _build_prompt(
    strategy: StrategistOutput,
    persona: dict,
    tweet_register: str,
) -> str:
    cfg  = load_config()
    niche = cfg.get("niche", "")

    jp_register = (
        "futsū-form (bahasa sehari-hari). Boleh pakai partikel akhir ね/よ/けど, "
        "sesekali w/笑, kosakata netizen yang wajar. Hindari です/ます berlebihan."
        if tweet_register == "casual"
        else "keigo yang sopan tapi tetap mudah dipahami, hindari terlalu formal."
    )

    persona_name = persona.get("name", "JP Content Creator")
    persona_desc = persona.get("description", "Content creator Jepang yang informatif dan akrab.")

    return f"""Kamu adalah {persona_name}.
{persona_desc}

Buat tweet tentang:

TOPIK  : {strategy.topic}
ANGLE  : {strategy.angle_type} — {strategy.angle_description}
NICHE  : {niche}
SUMBER : {strategy.source_url or '(tidak ada)'}

=== ATURAN TWEET JEPANG ===
• Register  : {jp_register}
• Panjang   : WAJIB ≤ {JP_MAX} karakter (hitung cermat, tiap karakter JP = 1)
• Hook      : Kalimat pertama HARUS menarik dan menghentikan scroll
• Emoji     : 0–1 (jangan berlebihan, hanya jika benar-benar pas)
• Hashtag   : 0–2 yang relevan (atau tidak ada sama sekali)
• HINDARI   : kalimat terlalu rapi/korporat, hype kosong ("game changer!"),
              pola template berulang, pembuka klise yang sama tiap tweet

=== ATURAN TWEET INDONESIA ===
• Panjang   : ≤ {ID_MAX} karakter
• Gaya      : Santai-cerdas, mengalir, seperti ngobrol di Twitter
• JANGAN    : terjemahkan kata-per-kata dari JP — buat versi sendiri dengan nada natural

=== CONTOH GAYA JEPANG YANG BAIK ===
❌ 非常に興味深い最新情報が公開されました
✅ これ、知らなかった人多そう...

❌ 公式から正式に発表されました
✅ ついに来た、みんな見て

=== OUTPUT ===
Balas HANYA JSON valid, tanpa teks lain:
{{
  "japanese": "ここに日本語ツイートを書く (≤{JP_MAX}文字)",
  "indonesian": "Tulis tweet Indonesia di sini"
}}"""


def _build_retry_prompt(
    strategy: StrategistOutput,
    persona: dict,
    tweet_register: str,
    too_long: str,
) -> str:
    base = _build_prompt(strategy, persona, tweet_register)
    return (
        base
        + f"\n\n⚠️ PERINGATAN: Versi sebelumnya {len(too_long)} karakter — TERLALU PANJANG!\n"
        f"Versi sebelumnya: {too_long}\n"
        f"Tulis ulang versi JP maks {JP_MAX} karakter. "
        f"Pertahankan inti pesan, potong bagian tidak esensial."
    )


# ------------------------------------------------------------------
# Parsing & validation
# ------------------------------------------------------------------

def _parse(
    raw: str,
    strategy: StrategistOutput,
    original: Optional[TweetDraft] = None,
    truncate: bool = True,
) -> TweetDraft:
    """Parse JSON dari LLM, validasi, potong jika perlu.

    truncate=False: biarkan JP melewati JP_MAX agar caller bisa cek dan retry.
    truncate=True : potong sebagai safety net setelah semua upaya selesai.
    """
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("JSON tidak ditemukan dalam output LLM")

        data = json.loads(raw[start:end])
        jp   = (data.get("japanese")   or "").strip()
        indo = (data.get("indonesian") or "").strip()

        if not jp:
            raise ValueError("field 'japanese' kosong")

        if truncate:
            if len(jp) > JP_MAX:
                logger.warning("JP masih %d chars setelah retry, dipotong ke %d", len(jp), JP_MAX)
                jp = jp[:JP_MAX]
            if len(indo) > ID_MAX:
                indo = indo[:ID_MAX]

        return TweetDraft(
            japanese=jp,
            indonesian=indo or jp,  # fallback ke JP jika ID kosong
            topic=strategy.topic,
            angle_type=strategy.angle_type,
            source_url=strategy.source_url,
        )

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Creator JSON parse gagal (%s), pakai %s", e, "original" if original else "fallback")
        return original if original else _fallback(strategy)


# ------------------------------------------------------------------
# Fallback
# ------------------------------------------------------------------

def _fallback(strategy: StrategistOutput) -> TweetDraft:
    """Draft minimal jika LLM gagal total (tidak pernah None)."""
    jp = f"【{strategy.topic}】"
    if len(jp) > JP_MAX:
        jp = jp[:JP_MAX]
    return TweetDraft(
        japanese=jp,
        indonesian=f"Info terbaru: {strategy.topic}",
        topic=strategy.topic,
        angle_type=strategy.angle_type,
        source_url=strategy.source_url,
    )
