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
    # ------------------------------------------------------------------
    def _strategize(self, trend: TrendCandidate) -> StrategistOutput:
        """Pilih angle berpotensi engagement tinggi via LLM."""
        from agents.strategist import pick_angle
        return pick_angle(trend)

    # ------------------------------------------------------------------
    # Tahap 3 — Creator (Writer)
    # ------------------------------------------------------------------
    def _create(self, strategy: StrategistOutput) -> TweetDraft:
        """Tulis tweet JP + ID berdasarkan angle dari Strategist."""
        from agents.creator import create_tweet
        return create_tweet(strategy)

    # ------------------------------------------------------------------
    # Tahap 4 — Critic ⇄ Reviser loop
    # ------------------------------------------------------------------
    def _critic_revise_loop(self, draft: TweetDraft) -> ContentPackage:
        """
        Loop Critic ⇄ Reviser maks self.max_revisions kali.
        Lacak draft terbaik (skor tertinggi) — bukan hanya yang terakhir.
        Berhenti lebih awal jika skor ≥ threshold.
        """
        from agents.critic_v2 import review
        from agents.reviser import revise

        current_draft = draft
        best_draft = draft
        best_score = -1
        best_breakdown: dict = {}
        revision_count = 0

        for iteration in range(self.max_revisions + 1):
            critic = review(current_draft)

            if critic.score > best_score:
                best_score = critic.score
                best_breakdown = critic.breakdown
                best_draft = current_draft

            if critic.score >= self.threshold:
                logger.info("Critic loop berhenti di iterasi %d (skor %d ≥ threshold %d)",
                            iteration, critic.score, self.threshold)
                break

            if iteration == self.max_revisions:
                break  # habis jatah, pakai best sejauh ini

            current_draft = revise(current_draft, critic)
            revision_count += 1

        return ContentPackage(
            topic=best_draft.topic,
            source_url=best_draft.source_url,
            angle_type=best_draft.angle_type,
            japanese=best_draft.japanese,
            indonesian=best_draft.indonesian,
            score=best_score,
            score_breakdown=best_breakdown,
            below_threshold=best_score < self.threshold,
            revision_count=revision_count,
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
    # ------------------------------------------------------------------
    def _dispatch(self, package: ContentPackage) -> int:
        """Format & kirim ke Telegram, simpan ke DB + embedding."""
        from agents.dispatcher import dispatch
        return dispatch(package)
