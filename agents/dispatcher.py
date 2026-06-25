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
from typing import Optional

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

# Tag header per verdict (hanya konten lolos kualitas yang dikirim)
_VERDICT_TAG = {
    "APPROVED": "✅ <b>APPROVED</b>",
    "GOOD":     "⚡ <b>GOOD</b>",
}


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
    Format pesan Telegram bilingual.

    Contoh output:
    ━━━━━━━━━━━━━━━━━━━━
    🇯🇵 AIのニュース、知らなかった人多そう...

    🇮🇩 Pada belum tau nih soal AI terbaru...
    ━━━━━━━━━━━━━━━━━━━━
    📊 Skor: <b>82/100</b>  |  Angle: news_insight
    🔗 <a href="https://...">Sumber</a>
    ⚠️ <i>Skor di bawah threshold — konten tetap dikirim</i>
    """
    sep = "━" * 20

    jp_safe   = html.escape(package.japanese)
    indo_safe = html.escape(package.indonesian)

    # Tag verdict — hanya konten lolos kualitas yang sampai ke sini
    tag = _VERDICT_TAG.get(package.verdict, "")

    parts = [
        f"{tag}".strip() or sep,
        sep if tag else None,
        f"🇯🇵 <code>{jp_safe}</code>",
        "",
        f"🇮🇩 {indo_safe}",
        sep,
        f'📊 Skor: <b>{package.score}/100</b>  |  Angle: {package.angle_type}',
    ]

    # URL sumber jika ada
    if package.source_url:
        url_safe = html.escape(package.source_url)
        parts.append(f'🔗 <a href="{url_safe}">Sumber berita</a>')

    return "\n".join(p for p in parts if p is not None)


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
