"""
orchestrator.py — Pipeline utama multi-agent.

Menjalankan state machine per siklus produksi:
  TrendScout → Strategist → Creator → Critic ⇄ Reviser (loop) → Dedup → Dispatcher

Setiap tahap merupakan method terpisah yang akan diisi implementasi
penuh di tahap upgrade masing-masing. Saat ini sebagian masih stub
yang memanggil modul lama agar sistem tetap bisa jalan.
"""
import logging
import random
from datetime import datetime, timezone
from typing import List, Optional

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
        from database.db_manager import start_run, finish_run
        run_id = start_run()
        produced = 0
        errors = 0

        logger.info("=== Mulai siklus produksi (run #%d) ===", run_id)
        try:
            # Tahap 1: Cari trend
            trend = self._scout_trend()
            if not trend:
                logger.warning("Tidak ada trend segar, siklus dibatalkan")
                finish_run(run_id, produced=0, errors=0)
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
                finish_run(run_id, produced=0, errors=0)
                return RunResult(success=False, reason="duplicate")

            # Tahap 7: Kirim ke Telegram & simpan ke DB
            content_id = self._dispatch(package)
            produced = 1
            logger.info(f"Konten #{content_id} berhasil dikirim ke Telegram")
            finish_run(run_id, produced=produced, errors=errors)
            return RunResult(success=True, content_id=content_id)

        except Exception as e:
            errors = 1
            logger.error(f"Error dalam siklus produksi: {e}", exc_info=True)
            finish_run(run_id, produced=produced, errors=errors)
            return RunResult(success=False, error=str(e))

    # ------------------------------------------------------------------
    # Tahap 1 — Trend Scout
    # ------------------------------------------------------------------
    def _scout_trend(self) -> Optional[TrendCandidate]:
        """
        Cari topik segar dari RSS feeds + Google Trends JP.
        Pilih secara acak dari top-5 untuk variasi tiap run.
        """
        from agents.trend_scout import scout_trends
        candidates = scout_trends()
        if not candidates:
            return None
        # Acak dari top-5 agar tidak selalu pakai kandidat pertama
        top = candidates[:min(5, len(candidates))]
        return random.choice(top)

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
    # ------------------------------------------------------------------
    def _dedup_check(self, package: ContentPackage) -> bool:
        """
        Cek kesamaan semantik konten JP vs konten 30 hari terakhir.
        Menggunakan sentence-transformers + cosine similarity.
        """
        from database.dedup import is_duplicate
        dedup_cfg = self.config["dedup"]
        # Cek berdasarkan konten JP (representasi utama)
        return not is_duplicate(
            package.japanese,
            lookback_days=dedup_cfg["lookback_days"],
            threshold=dedup_cfg["similarity_threshold"],
        )

    # ------------------------------------------------------------------
    # Tahap 6 — Dispatcher
    # [Implementasi penuh: Tahap 7 upgrade]
    # ------------------------------------------------------------------
    def _dispatch(self, package: ContentPackage) -> int:
        """
        [Tahap 7] Format & kirim ke Telegram, simpan ke DB + embedding.
        Stub sekarang: simpan ke DB lengkap + simpan embedding.
        Telegram dispatch format baru aktif di Tahap 7.
        """
        from database.db_manager import save_tweet_full, save_topic_used, update_tweet_sent
        from database.dedup import check_and_save

        # Simpan ke DB dengan semua metadata
        tweet_id = save_tweet_full(
            topic=package.topic,
            content_jp=package.japanese,
            content_indo=package.indonesian,
            draft_jp=package.japanese,
            score=package.score,
            score_breakdown=package.score_breakdown,
            angle_type=package.angle_type,
        )

        # Simpan embedding untuk dedup konten berikutnya
        check_and_save(tweet_id, package.japanese)

        # Catat ke memory
        save_topic_used(package.topic, angle_type=package.angle_type)

        logger.info("[STUB] Konten #%d tersimpan (Telegram dispatch aktif di Tahap 7)", tweet_id)
        logger.info("  [JP] %s", package.japanese)
        logger.info("  [ID] %s", package.indonesian)
        logger.info("  Skor: %d/100 | Angle: %s", package.score, package.angle_type)
        return tweet_id
