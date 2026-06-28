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


# ==================================================================
# Comment Agent bot — terima SCREENSHOT, balas 3 saran komentar
# ==================================================================

_FILE_API = "https://api.telegram.org/file/bot{token}/{path}"
_HELP = (
    "👋 Kirim <b>screenshot</b> postingan Jepang, dan aku balas "
    "<b>3 saran komentar</b> (JP + ID) untuk engagement tinggi — aman, tidak menyinggung."
)


def _send(chat_id, text: str) -> bool:
    """Kirim pesan ke chat tertentu (membalas pengirim)."""
    try:
        r = _post("sendMessage", {
            "chat_id": chat_id, "text": text,
            "parse_mode": "HTML", "disable_web_page_preview": True,
        })
        return r.get("ok", False)
    except Exception as e:
        print(f"❌ Telegram send gagal: {e}")
        return False


def _largest_photo_id(message: dict) -> str:
    """Ambil file_id foto resolusi tertinggi dari sebuah message (atau '')."""
    photos = message.get("photo") or []
    if not photos:
        return ""
    # Telegram mengirim beberapa ukuran; ambil yang terbesar (terakhir)
    return photos[-1].get("file_id", "")


def _download_file(file_id: str) -> bytes:
    """Unduh file Telegram berdasarkan file_id. b'' jika gagal."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return b""
    try:
        info = _post("getFile", {"file_id": file_id})
        path = info.get("result", {}).get("file_path", "")
        if not path:
            return b""
        url = _FILE_API.format(token=token, path=path)
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"❌ Gagal unduh file: {e}")
        return b""


def process_photo(file_id: str, chat_id) -> bool:
    """Unduh foto → Comment Agent → balas 3 saran komentar."""
    from agents.comment_agent import comments_from_image, format_reply

    image = _download_file(file_id)
    if not image:
        return _send(chat_id, "⚠️ Gagal mengunduh gambar. Coba kirim ulang ya.")

    _send(chat_id, "🔎 Lagi baca screenshot-nya...")
    cs = comments_from_image(image, mime="image/jpeg")
    return _send(chat_id, format_reply(cs))


def handle_update(update: dict) -> None:
    """Proses satu update Telegram (foto → komentar; teks → bantuan)."""
    message = update.get("message") or update.get("channel_post") or {}
    chat_id = (message.get("chat") or {}).get("id")
    if chat_id is None:
        return

    file_id = _largest_photo_id(message)
    if file_id:
        process_photo(file_id, chat_id)
        return

    # Bukan foto → kirim petunjuk
    _send(chat_id, _HELP)


def _get_updates(offset: int, timeout: int = 30) -> list:
    """Long-polling getUpdates. Kembalikan list update (boleh kosong)."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    url = _TG_API.format(token=token, method="getUpdates")
    try:
        resp = requests.get(
            url, params={"offset": offset, "timeout": timeout},
            timeout=timeout + 10,
        )
        resp.raise_for_status()
        return resp.json().get("result", [])
    except requests.exceptions.RequestException as e:
        print(f"⚠️ getUpdates error: {e}")
        return []


def run_bot() -> None:
    """
    Jalankan bot Comment Agent (long-polling).
    Kirim screenshot → balas 3 saran komentar JP+ID.
    """
    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        print("❌ TELEGRAM_BOT_TOKEN tidak di-set — bot tidak bisa jalan")
        return

    print("🤖 Comment Agent bot berjalan (kirim screenshot ke bot)...")
    offset = 0
    while True:
        updates = _get_updates(offset)
        for upd in updates:
            offset = max(offset, upd.get("update_id", 0) + 1)
            try:
                handle_update(upd)
            except Exception as e:
                print(f"❌ Error memproses update: {e}")


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
