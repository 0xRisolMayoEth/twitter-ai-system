"""
orchestrator.py — Pipeline utama multi-agent.

Menjalankan state machine per siklus produksi:
  TrendScout → Strategist → Creator → Critic ⇄ Reviser (loop) → Dedup → Dispatcher

Setiap tahap merupakan method terpisah yang akan diisi implementasi
penuh di tahap upgrade masing-masing. Saat ini sebagian masih stub
yang memanggil modul lama agar sistem tetap bisa jalan.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from core.config import load_config
from core.logger import get_logger
from core.models import (
    TrendCandidate,
    StrategistOutput,
    TweetDraft,
    ContentPackage,
    RunResult,
)

logger = get_logger("orchestrator")


class Orchestrator:
    """Menjalankan satu siklus produksi: hasilkan 1 paket konten JP+ID."""

    def __init__(self):
        self.config = load_config()
        self.threshold = self.config["scoring"]["threshold"]
        self.max_revisions = self.config["scoring"]["max_revisions"]

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run_cycle(self) -> RunResult:
        """
        Jalankan satu siklus penuh.
        Satu konten gagal tidak menghentikan siklus lain.
        """
        logger.info("=== Mulai siklus produksi ===")
        try:
            # Tahap 1: Cari trend
            trend = self._scout_trend()
            if not trend:
                logger.warning("Tidak ada trend segar, siklus dibatalkan")
                return RunResult(success=False, reason="no_trend")
            logger.info(f"Trend: '{trend.topic}' [{trend.source}]")

            # Tahap 2: Pilih angle
            strategy = self._strategize(trend)
            logger.info(f"Angle: {strategy.angle_type} — {strategy.angle_description}")

            # Tahap 3: Buat draft konten (JP + ID)
            draft = self._create(strategy)
            logger.info(f"Draft JP ({len(draft.japanese)} chars): {draft.japanese[:60]}…")

            # Tahap 4-5: Loop Critic ⇄ Reviser maks N kali
            package = self._critic_revise_loop(draft)
            logger.info(
                f"Skor final: {package.score}/100 | revisi: {package.revision_count}x"
                + (" [BELOW-THRESHOLD]" if package.below_threshold else "")
            )

            # Tahap 6: Cek duplikat semantik
            if not self._dedup_check(package):
                logger.warning("Konten terlalu mirip dengan yang sudah ada, dibuang")
                return RunResult(success=False, reason="duplicate")

            # Tahap 7: Kirim ke Telegram & simpan ke DB
            content_id = self._dispatch(package)
            logger.info(f"Konten #{content_id} berhasil dikirim ke Telegram")
            return RunResult(success=True, content_id=content_id)

        except Exception as e:
            logger.error(f"Error dalam siklus produksi: {e}", exc_info=True)
            return RunResult(success=False, error=str(e))

    # ------------------------------------------------------------------
    # Tahap 1 — Trend Scout
    # [Implementasi penuh: Tahap 3 upgrade]
    # ------------------------------------------------------------------
    def _scout_trend(self) -> Optional[TrendCandidate]:
        """Cari topik segar dari RSS + Google Trends."""
        # Stub: pakai trend.py lama
        from agents.trend import get_trending_topics
        topics = get_trending_topics()
        if not topics:
            return None
        t = topics[0]
        return TrendCandidate(
            topic=t["title"],
            source=t.get("source", "rss"),
            url=t.get("url", ""),
            freshness=datetime.now(timezone.utc),
            raw_summary=t.get("title", ""),
            category=t.get("category", ""),
        )

    # ------------------------------------------------------------------
    # Tahap 2 — Strategist
    # [Implementasi penuh: Tahap 4 upgrade]
    # ------------------------------------------------------------------
    def _strategize(self, trend: TrendCandidate) -> StrategistOutput:
        """Pilih angle berpotensi engagement tinggi via LLM."""
        # Stub: angle tetap, akan diganti LLM call di Tahap 4
        niche = self.config.get("niche", "")
        return StrategistOutput(
            topic=trend.topic,
            source_url=trend.url,
            angle_type="news_insight",
            angle_description=f"Insight singkat tentang {trend.topic} untuk audiens Jepang",
            reasoning="[stub — Strategist akan diimplementasi di Tahap 4]",
        )

    # ------------------------------------------------------------------
    # Tahap 3 — Creator (Writer)
    # [Implementasi penuh: Tahap 5 upgrade]
    # ------------------------------------------------------------------
    def _create(self, strategy: StrategistOutput) -> TweetDraft:
        """Tulis tweet JP + ID berdasarkan angle dari Strategist."""
        # Stub: pakai writer.py lama
        from agents.writer import write_tweet
        content = write_tweet(
            strategy.topic,
            strategy.angle_description,
            [],
        )
        # writer.py lama belum bilingual — keduanya pakai hasil yang sama
        return TweetDraft(
            japanese=content,
            indonesian=content,
            topic=strategy.topic,
            angle_type=strategy.angle_type,
            source_url=strategy.source_url,
        )

    # ------------------------------------------------------------------
    # Tahap 4 — Critic ⇄ Reviser loop
    # [Implementasi penuh: Tahap 6 upgrade]
    # ------------------------------------------------------------------
    def _critic_revise_loop(self, draft: TweetDraft) -> ContentPackage:
        """
        Loop penilaian & revisi, maks self.max_revisions kali.
        Saat ini stub: satu round dengan critic.py lama (skor 1-10 → 0-100).
        """
        from agents.critic import review_tweet
        review = review_tweet(draft.japanese)

        # Konversi skor lama (1-10) ke skala baru (0-100)
        raw_score = review.get("skor", 7)
        score = min(100, int(raw_score * 10))
        below = score < self.threshold

        return ContentPackage(
            topic=draft.topic,
            source_url=draft.source_url,
            angle_type=draft.angle_type,
            japanese=review.get("tweet_revisi", draft.japanese),
            indonesian=draft.indonesian,
            score=score,
            score_breakdown={"legacy_skor": raw_score},
            below_threshold=below,
            revision_count=0,
        )

    # ------------------------------------------------------------------
    # Tahap 5 — Dedup semantik
    # [Implementasi penuh: Tahap 2 upgrade]
    # ------------------------------------------------------------------
    def _dedup_check(self, package: ContentPackage) -> bool:
        """
        Cek kesamaan semantik vs konten 30 hari terakhir.
        Stub: selalu lolos — embedding belum diimplementasi.
        """
        logger.debug("[STUB] Dedup check dilewati (akan aktif di Tahap 2)")
        return True

    # ------------------------------------------------------------------
    # Tahap 6 — Dispatcher
    # [Implementasi penuh: Tahap 7 upgrade]
    # ------------------------------------------------------------------
    def _dispatch(self, package: ContentPackage) -> int:
        """
        Format & kirim ke Telegram, catat ke DB.
        Stub: simpan ke DB + log, belum kirim Telegram dengan format baru.
        """
        from database.db_manager import save_tweet, save_topic_used
        tweet_db_id = save_tweet(package.japanese, package.topic)
        save_topic_used(package.topic)

        logger.info(
            f"[STUB] Konten #{tweet_db_id} disimpan "
            f"(Telegram dispatch aktif di Tahap 7)"
        )
        logger.info(f"  [JP] {package.japanese}")
        logger.info(f"  [ID] {package.indonesian}")
        logger.info(f"  Skor: {package.score}/100")
        return tweet_db_id
