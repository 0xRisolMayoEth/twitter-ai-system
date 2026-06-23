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
    """Hasil pemilihan angle oleh Strategist."""
    topic: str
    source_url: str = ""
    angle_type: str                      # salah satu dari ANGLE_TYPES
    angle_description: str              # deskripsi singkat angle yang dipilih
    reasoning: str                      # alasan LLM memilih angle ini


# ------------------------------------------------------------------
# 3. Output Creator (Writer)
# ------------------------------------------------------------------
class TweetDraft(BaseModel):
    """Draft tweet dua bahasa dari Creator."""
    japanese: str                        # versi Jepang (≤140 char)
    indonesian: str                      # versi Indonesia
    topic: str
    angle_type: str
    source_url: str = ""


# ------------------------------------------------------------------
# 4. Output Critic
# ------------------------------------------------------------------
class CriticResult(BaseModel):
    """Hasil penilaian dari Critic (skor 0-100 dengan rubrik per aspek)."""
    score: int                           # total 0-100
    breakdown: Dict[str, int] = Field(  # skor per aspek
        default_factory=dict
        # contoh: {"hook": 22, "engagement": 16, "naturalness": 18, ...}
    )
    issues: List[str] = Field(default_factory=list)       # masalah yang ditemukan
    suggestions: List[str] = Field(default_factory=list)  # saran perbaikan spesifik


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
    score: int
    score_breakdown: Dict[str, int] = Field(default_factory=dict)
    below_threshold: bool = False        # True jika skor < threshold setelah 5 revisi
    revision_count: int = 0


# ------------------------------------------------------------------
# 6. Hasil satu siklus orchestrator
# ------------------------------------------------------------------
class RunResult(BaseModel):
    """Status hasil satu siklus produksi."""
    success: bool
    content_id: Optional[int] = None    # ID baris di tabel tweets
    reason: Optional[str] = None        # "no_trend" | "duplicate" | "below_threshold"
    error: Optional[str] = None         # pesan error jika ada
