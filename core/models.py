"""
core/models.py — Dataclass (Pydantic) untuk state antar-agent.

Setiap agent menerima dan mengembalikan salah satu model ini
sehingga kontrak input/output jelas dan mudah dites.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# 1. Output TrendScout
# ------------------------------------------------------------------
class TrendCandidate(BaseModel):
    """Satu kandidat topik dari TrendScout."""
    topic: str
    source: str                          # nama sumber (e.g. "NHK News")
    url: str = ""                        # URL artikel asli
    freshness: datetime = Field(         # waktu publikasi
        default_factory=lambda: datetime.now(timezone.utc)
    )
    raw_summary: str = ""                # ringkasan mentah dari feed
    category: str = ""                  # kategori (ja/en, niche, dsb.)


# ------------------------------------------------------------------
# 2. Output Strategist
# ------------------------------------------------------------------
ANGLE_TYPES = (
    "curiosity_gap",   # hook pertanyaan, bikin penasaran
    "contrarian",      # opini/take yang melawan arus
    "relatable",       # "ini gue banget"
    "news_insight",    # berita + insight singkat
    "listicle",        # fakta mengejutkan / mini list
)

class StrategistOutput(BaseModel):
    """
    Hasil pemilihan trend salaryman (TREND AGENT).
    Field lama (angle_type/angle_description/reasoning) dipertahankan untuk
    kompatibilitas DB & test; field salaryman baru ada di bawahnya.
    """
    topic: str                           # = trending_topic
    source_url: str = ""
    topic_category: str = ""             # commute / overtime / lunch / paycheck / ...
    why_relatable: str = ""              # kenapa relevan untuk salaryman
    search_query_for_image: str = ""     # kata kunci awal untuk Image Agent
    angle_type: str = "relatable"        # disimpan ke DB (diisi = topic_category)
    angle_description: str = ""          # diisi = why_relatable
    reasoning: str = ""


# ------------------------------------------------------------------
# 3. Output Creator (Writer — 田中サトル)
# ------------------------------------------------------------------
class TweetDraft(BaseModel):
    """Draft tweet salaryman (JP) + terjemahan ID."""
    japanese: str                        # = tweet_text (80–140 char, 口語体)
    indonesian: str                      # terjemahan/parafrase ID
    topic: str
    angle_type: str
    source_url: str = ""
    tone: str = ""                       # mis. "lelah tapi lucu", "self-deprecating"
    best_posting_time: str = ""          # saran waktu posting dari writer
    is_fallback: bool = False            # True jika LLM gagal & ini draft darurat (jangan dikirim)


# ------------------------------------------------------------------
# 3b. Output Image Agent (BARU)
# ------------------------------------------------------------------
class ImageRec(BaseModel):
    """Rekomendasi gambar untuk menemani tweet (bukan generate gambar)."""
    image_description: str = ""
    google_search_queries: List[str] = Field(default_factory=list)  # 3 query (2 EN + 1 JP)
    image_style: str = ""
    reason: str = ""
    is_fallback: bool = False


# ------------------------------------------------------------------
# 4. Output Critic
# ------------------------------------------------------------------
class CriticResult(BaseModel):
    """
    Hasil penilaian Critic salaryman.
    4 dimensi (1–10): relatability, naturalness, engagement, topic_fit.
    Jika SALAH SATU dimensi < 6 → verdict REJECT + improved_tweet (rewrite).
    """
    score: int                           # total_score 0–10 (rata-rata dibulatkan)
    verdict: str = "REJECT"              # APPROVE | REJECT
    breakdown: Dict[str, int] = Field(  # skor per dimensi (1–10)
        default_factory=dict
        # contoh: {"relatability": 8, "naturalness": 7, "engagement": 6, "topic_fit": 9}
    )
    improved_tweet: str = ""             # hasil rewrite otomatis jika REJECT
    feedback: str = ""                   # ringkasan masukan
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    is_fallback: bool = False            # True jika ini hasil fallback (LLM gagal)


# ------------------------------------------------------------------
# 5. Paket konten final (setelah loop Critic⇄Reviser)
# ------------------------------------------------------------------
class ContentPackage(BaseModel):
    """Paket konten final siap kirim ke Telegram."""
    topic: str
    source_url: str = ""
    angle_type: str
    japanese: str
    indonesian: str
    score: int                           # total_score 0–10
    verdict: str = "REJECT"             # APPROVE | REJECT
    score_breakdown: Dict[str, int] = Field(default_factory=dict)
    tone: str = ""
    best_posting_time: str = ""
    image: Optional[ImageRec] = None     # rekomendasi gambar dari Image Agent
    below_threshold: bool = False        # True jika ada dimensi < 6 setelah N revisi
    revision_count: int = 0
    is_fallback: bool = False            # True jika konten berasal dari fallback (LLM gagal)


# ------------------------------------------------------------------
# 5b. Comment Agent (on-demand dari screenshot)
# ------------------------------------------------------------------
class Comment(BaseModel):
    """Satu saran komentar dua bahasa."""
    japanese: str
    indonesian: str
    angle: str = ""                      # tipe komentar: empati / pertanyaan / humor


class CommentSet(BaseModel):
    """Kumpulan saran komentar untuk satu postingan (dari screenshot)."""
    comments: List[Comment] = Field(default_factory=list)
    source_text: str = ""                # teks Jepang yang terbaca dari gambar
    summary: str = ""                    # ringkasan isi konten (bahasa Indonesia)
    is_fallback: bool = False


# ------------------------------------------------------------------
# 6. Hasil satu siklus orchestrator
# ------------------------------------------------------------------
class RunResult(BaseModel):
    """Status hasil satu siklus produksi."""
    success: bool
    content_id: Optional[int] = None    # ID baris di tabel tweets
    reason: Optional[str] = None        # "no_trend" | "duplicate" | "below_threshold"
    error: Optional[str] = None         # pesan error jika ada
