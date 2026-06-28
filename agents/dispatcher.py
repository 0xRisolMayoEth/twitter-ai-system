"""
agents/dispatcher.py — Dispatcher agent.

Input : ContentPackage
Output: int (tweet_id di DB)

Pipeline:
  1. Simpan konten ke DB (save_tweet_full)
  2. Simpan embedding untuk dedup berikutnya
  3. Catat topik ke memory
  4. Format pesan Telegram bilingual
  5. Kirim langsung ke Telegram (tanpa approve/reject)
  6. Update status DB jika berhasil kirim

Telegram dianggap opsional — jika token tidak di-set, konten tetap tersimpan di DB.
"""
import html
import logging
import os
import time
from datetime import datetime
from typing import Optional

import pytz
import requests

from core.config import load_config
from core.logger import get_logger
from core.models import ContentPackage
from database.db_manager import save_tweet_full, save_topic_used, update_tweet_sent
from database.dedup import check_and_save

logger = get_logger("dispatcher")

_TG_API = "https://api.telegram.org/bot{token}/sendMessage"
_MAX_RETRIES = 3
_RETRY_BASE  = 2  # detik
_JST = pytz.timezone("Asia/Tokyo")


def _slot_label(hour: int) -> str:
    """Label slot posting JST: pagi/siang/sore/malam."""
    if 5 <= hour <= 10:
        return "pagi"
    if 11 <= hour <= 14:
        return "siang"
    if 15 <= hour <= 18:
        return "sore"
    return "malam"


def dispatch(package: ContentPackage) -> int:
    """
    Simpan ke DB, kirim ke Telegram, kembalikan tweet_id.
    Telegram gagal tidak menghentikan pipeline — tweet_id tetap dikembalikan.
    """
    # 1. Simpan ke DB
    tweet_id = save_tweet_full(
        topic=package.topic,
        content_jp=package.japanese,
        content_indo=package.indonesian,
        draft_jp=package.japanese,
        score=package.score,
        score_breakdown=package.score_breakdown,
        angle_type=package.angle_type,
    )

    # 2. Simpan embedding untuk dedup konten berikutnya
    check_and_save(tweet_id, package.japanese)

    # 3. Catat topik + angle ke memory
    save_topic_used(package.topic, angle_type=package.angle_type)

    # 4. Format & kirim ke Telegram
    message = format_message(package)
    sent = send_telegram(message)

    # 5. Update status
    if sent:
        update_tweet_sent(tweet_id)
        logger.info("Konten #%d berhasil dikirim ke Telegram | skor=%d", tweet_id, package.score)
    else:
        logger.warning("Konten #%d tersimpan di DB tapi TIDAK terkirim ke Telegram", tweet_id)

    return tweet_id


# ------------------------------------------------------------------
# Format pesan Telegram
# ------------------------------------------------------------------

def format_message(package: ContentPackage) -> str:
    """
    Format pesan Telegram untuk review salaryman (format-only, tanpa posting).

    Contoh output:
    🕐 06:00 — Jadwal posting pagi
    📝 DRAFT TWEET:
    <code>満員電車で...😮‍💨 #サラリーマン</code>

    🇮🇩 Di kereta penuh sesak...

    📊 SCORE: 8/10
    🖼️ REKOMENDASI GAMBAR:
    - crowded tokyo train morning
    - tired office worker desk
    - 満員電車 イラスト

    ✅ Ketik /approve untuk post
    ❌ Ketik /reject untuk skip
    """
    now = datetime.now(_JST)
    jam = now.strftime("%H:%M")
    label = _slot_label(now.hour)

    jp_safe   = html.escape(package.japanese)
    indo_safe = html.escape(package.indonesian)

    parts = [
        f"🕐 {jam} — Jadwal posting {label}",
        "📝 DRAFT TWEET:",
        f"<code>{jp_safe}</code>",
        "",
        f"🇮🇩 {indo_safe}",
        "",
        f"📊 SCORE: <b>{package.score}/10</b>",
    ]

    # Rekomendasi gambar dari Image Agent
    if package.image and package.image.google_search_queries:
        parts.append("🖼️ REKOMENDASI GAMBAR:")
        for q in package.image.google_search_queries[:3]:
            parts.append(f"- {html.escape(q)}")

    parts += [
        "",
        "✅ Ketik /approve untuk post",
        "❌ Ketik /reject untuk skip",
    ]

    return "\n".join(parts)


# ------------------------------------------------------------------
# Kirim ke Telegram via Bot API (requests, bukan library)
# ------------------------------------------------------------------

def send_telegram(text: str, token: Optional[str] = None, chat_id: Optional[str] = None) -> bool:
    """
    Kirim pesan teks ke Telegram menggunakan requests.
    Retry otomatis hingga _MAX_RETRIES kali dengan exponential backoff.
    Kembalikan False (tidak raise) jika semua percobaan gagal.
    """
    token   = token   or os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        logger.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID tidak di-set — skip Telegram")
        return False

    url = _TG_API.format(token=token)
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    cfg = load_config()
    rate_limit = cfg.get("telegram", {}).get("rate_limit_seconds", 0)

    for attempt in range(_MAX_RETRIES):
        try:
            if rate_limit and attempt == 0:
                time.sleep(rate_limit)

            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if not data.get("ok"):
                logger.error("Telegram API error: %s", data.get("description", "unknown"))
                return False

            return True

        except requests.exceptions.Timeout:
            logger.warning("Telegram timeout (percobaan %d/%d)", attempt + 1, _MAX_RETRIES)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else 0
            if status == 429:  # Too Many Requests
                retry_after = int(e.response.headers.get("Retry-After", 30))
                logger.warning("Telegram rate-limited, tunggu %ds", retry_after)
                time.sleep(retry_after)
            else:
                logger.error("Telegram HTTP %d: %s", status, e)
                return False  # tidak perlu retry untuk 4xx
        except requests.exceptions.RequestException as e:
            logger.warning("Telegram request error (percobaan %d/%d): %s", attempt + 1, _MAX_RETRIES, e)

        if attempt < _MAX_RETRIES - 1:
            time.sleep(_RETRY_BASE ** attempt)

    logger.error("Telegram gagal setelah %d percobaan", _MAX_RETRIES)
    return False
