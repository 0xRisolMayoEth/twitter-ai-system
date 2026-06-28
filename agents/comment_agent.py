"""
agents/comment_agent.py — COMMENT AGENT (on-demand dari screenshot).

Alur:
  1. User kirim SCREENSHOT konten Jepang ke bot Telegram.
  2. extract_japanese_text(): model vision membaca teks Jepang dari gambar.
  3. generate_comments(): model teks membuat 3 saran komentar (JP + ID) yang
     dirancang untuk engagement & view tinggi, dan TIDAK menyinggung.

comments_from_image() menggabungkan keduanya. format_reply() merangkai pesan
Telegram. Semua punya fallback agar bot tidak pernah crash.
"""
import json
from typing import List

from core.llm import chat, vision_chat
from core.logger import get_logger
from core.models import Comment, CommentSet

logger = get_logger("comment_agent")

N_COMMENTS = 3

_VISION_PROMPT = (
    "Baca screenshot ini. Ini adalah sebuah postingan media sosial berbahasa "
    "Jepang. Tuliskan KEMBALI teks utama postingan itu dalam bahasa Jepang "
    "apa adanya (tanpa terjemahan, tanpa komentar tambahan). Jika ada beberapa "
    "bagian, ambil teks postingan utamanya saja."
)


def comments_from_image(image_bytes: bytes, mime: str = "image/jpeg") -> CommentSet:
    """Pipeline lengkap: gambar → teks Jepang → 3 komentar JP+ID."""
    text = extract_japanese_text(image_bytes, mime)
    if not text:
        logger.warning("Tidak ada teks terbaca dari gambar")
        return CommentSet(comments=[], source_text="", is_fallback=True)
    return generate_comments(text)


# ------------------------------------------------------------------
# Langkah 1 — baca teks dari screenshot (vision)
# ------------------------------------------------------------------

def extract_japanese_text(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    """Baca teks Jepang dari screenshot via model vision. '' jika gagal."""
    try:
        text = vision_chat(_VISION_PROMPT, image_bytes, mime=mime,
                           max_tokens=500, temperature=0.0).strip()
        logger.info("Vision membaca %d karakter", len(text))
        return text
    except Exception as e:
        logger.error("Vision gagal membaca gambar: %s", e)
        return ""


# ------------------------------------------------------------------
# Langkah 2 — buat 3 komentar (teks)
# ------------------------------------------------------------------

def generate_comments(jp_text: str) -> CommentSet:
    """Buat 3 saran komentar (JP + ID) untuk postingan jp_text."""
    prompt = _build_prompt(jp_text)
    try:
        raw = chat(messages=[{"role": "user", "content": prompt}],
                   max_tokens=700, temperature=0.85)
        cs = _parse(raw, jp_text)
        logger.info("Comment Agent: %d komentar dibuat", len(cs.comments))
        return cs
    except Exception as e:
        logger.error("Comment Agent LLM error: %s — pakai fallback", e)
        return _fallback(jp_text)


def _build_prompt(jp_text: str) -> str:
    return f"""Kamu pengguna Twitter/X Jepang yang pandai membuat komentar yang
"nyangkut" — bikin orang ingin like, reply, dan postingannya makin terlihat.

Berikut postingan orang lain (bahasa Jepang) yang ingin kamu komentari:
\"\"\"
{jp_text}
\"\"\"

LANGKAH 1 — PAHAMI dulu isinya: apa inti postingan ini, apa emosi/maksudnya,
dan apa detail spesifik yang bisa kamu tanggapi. JANGAN balas template umum.

LANGKAH 2 — Buat {N_COMMENTS} SARAN komentar berbeda. Tujuan: ENGAGEMENT &
VIEW tinggi. Tiap komentar:
• RELEVAN dengan isi spesifik postingan (sebut/singgung hal konkret di dalamnya,
  bukan komentar yang bisa ditempel ke postingan apa pun)
• Bahasa Jepang SANTAI seperti ngobrol sama teman (口語体/タメ口 wajar,
  boleh ね/よ/じゃん/わ/笑). Bukan kaku, bukan formal.
• JANGAN terlalu pendek — buat yang berisi, kira-kira 1–2 kalimat
  (sekitar 40–120 karakter), terasa seperti orang sungguhan menanggapi
• Tiap komentar punya sudut berbeda: empati/setuju, pertanyaan yang memancing
  balasan, atau humor ringan yang relatable
• Boleh 0–2 emoji bila pas

ATURAN PENTING:
• DILARANG menyinggung, menghina, sarkas menyerang, SARA, politik, body-shaming
• Jangan menggurui, jangan spam, jangan promosi
• Santai & ramah — aman untuk semua orang

Untuk tiap komentar Jepang, sertakan terjemahan Indonesianya (santai, natural).

Balas HANYA JSON valid:
{{
  "comments": [
    {{"japanese": "コメント1", "indonesian": "terjemahan 1", "angle": "empati|pertanyaan|humor"}},
    {{"japanese": "コメント2", "indonesian": "terjemahan 2", "angle": "..."}},
    {{"japanese": "コメント3", "indonesian": "terjemahan 3", "angle": "..."}}
  ]
}}"""


def _parse(raw: str, jp_text: str) -> CommentSet:
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("JSON tidak ditemukan dalam respons LLM")
    data = json.loads(raw[start:end])
    items = data.get("comments") or []
    if not isinstance(items, list):
        raise ValueError("'comments' bukan list")

    comments: List[Comment] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        jp = (it.get("japanese") or "").strip()
        indo = (it.get("indonesian") or "").strip()
        if not jp:
            continue
        comments.append(Comment(japanese=jp, indonesian=indo or jp,
                                angle=(it.get("angle") or "").strip()))
    if not comments:
        raise ValueError("tidak ada komentar valid")
    return CommentSet(comments=comments[:N_COMMENTS], source_text=jp_text)


def _fallback(jp_text: str) -> CommentSet:
    """Saran komentar darurat (aman & netral) bila LLM gagal."""
    return CommentSet(
        comments=[
            Comment(japanese="いやこれめっちゃ分かるわ…自分も同じこと思ってたとこ！",
                    indonesian="Ini relatable banget sih… gue juga lagi mikirin hal yang sama!",
                    angle="empati"),
            Comment(japanese="ちなみにこういうの、みんなはどうやって乗り切ってるんだろ？気になる〜",
                    indonesian="Btw kalau yang kayak gini, pada ngadepinnya gimana ya? Penasaran~",
                    angle="pertanyaan"),
            Comment(japanese="朝からこれはなかなか効くやつだわ…笑 でもなんか元気もらった！",
                    indonesian="Dari pagi udah kena yang kayak gini wkwk tapi malah jadi semangat!",
                    angle="humor"),
        ],
        source_text=jp_text,
        is_fallback=True,
    )


# ------------------------------------------------------------------
# Format balasan Telegram
# ------------------------------------------------------------------

def format_reply(cs: CommentSet) -> str:
    """Rangkai CommentSet jadi pesan Telegram HTML (komentar JP copyable)."""
    import html

    if not cs.comments:
        return "⚠️ Maaf, tidak ada teks Jepang yang bisa kubaca dari gambar itu."

    parts = ["💬 <b>3 Saran Komentar</b>"]
    if cs.source_text:
        src = html.escape(cs.source_text[:120])
        parts.append(f"<i>Postingan:</i> {src}")
    if cs.is_fallback:
        parts.append("⚠️ <i>(saran umum — LLM sedang tidak tersedia)</i>")
    parts.append("")

    for i, c in enumerate(cs.comments, 1):
        jp = html.escape(c.japanese)
        indo = html.escape(c.indonesian)
        tag = f" <i>({html.escape(c.angle)})</i>" if c.angle else ""
        parts.append(f"<b>{i}.</b>{tag}")
        parts.append(f"🇯🇵 <code>{jp}</code>")
        parts.append(f"🇮🇩 {indo}")
        parts.append("")

    return "\n".join(parts).rstrip()
