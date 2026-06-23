"""
tg_bot/bot.py — Telegram bot utilities.

Semua pengiriman ke Telegram dilakukan via requests (Telegram Bot API),
bukan library python-telegram-bot, agar tidak ada dependency yang berat.

Pipeline baru (Tahap 7+) menggunakan agents/dispatcher.py yang memanggil
send_direct() di sini. Fungsi lama (approval flow) dipertahankan sebagai
referensi tapi tidak dipakai dari pipeline utama.
"""
import html
import os

import requests
from dotenv import load_dotenv

load_dotenv()

_TG_API = "https://api.telegram.org/bot{token}/{method}"


def _post(method: str, payload: dict) -> dict:
    """HTTP helper — panggil Telegram Bot API, kembalikan JSON respons."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN tidak di-set")
    url = _TG_API.format(token=token, method=method)
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def send_direct(text: str, parse_mode: str = "HTML") -> bool:
    """
    Kirim teks langsung ke chat tanpa tombol approve/reject.
    Digunakan oleh dispatcher baru.
    """
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not chat_id:
        print("⚠️  TELEGRAM_CHAT_ID tidak di-set, pesan tidak dikirim")
        return False
    try:
        result = _post("sendMessage", {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        })
        return result.get("ok", False)
    except Exception as e:
        print(f"❌ Telegram send gagal: {e}")
        return False


def notify_simple(message: str) -> bool:
    """Kirim notifikasi teks sederhana (untuk error/info sistem)."""
    return send_direct(message, parse_mode="HTML")


# ------------------------------------------------------------------
# Legacy — dipertahankan untuk referensi, tidak dipakai dari pipeline baru
# ------------------------------------------------------------------

def send_tweet_for_approval(tweet_db_id: int, tweet_content: str, topic: str) -> None:
    """
    [LEGACY] Kirim tweet dengan tombol Approve/Reject.
    Pipeline baru (Tahap 7) tidak memanggil fungsi ini — gunakan dispatcher.dispatch().
    """
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not chat_id:
        return

    content_safe = html.escape(tweet_content)
    topic_safe   = html.escape(topic)

    message = (
        f"🐦 <b>Tweet Baru untuk Review</b>\n\n"
        f"📌 <b>Topik:</b> {topic_safe}\n\n"
        f"📝 <b>Draft:</b>\n<code>{content_safe}</code>\n\n"
        f"📊 Panjang: {len(tweet_content)}/280 karakter"
    )
    keyboard = {"inline_keyboard": [[
        {"text": "✅ Approve", "callback_data": f"approve_{tweet_db_id}"},
        {"text": "❌ Reject",  "callback_data": f"reject_{tweet_db_id}"},
    ]]}
    try:
        _post("sendMessage", {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "reply_markup": keyboard,
        })
        print(f"📨 Tweet #{tweet_db_id} dikirim ke Telegram!")
    except Exception as e:
        print(f"❌ Telegram error: {e}")
