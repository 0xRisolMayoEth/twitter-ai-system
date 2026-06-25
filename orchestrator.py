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
        # Berapa banyak topik berbeda yang dicoba per siklus sebelum menyerah
        self.max_topic_attempts = self.config["scoring"].get("max_topic_attempts", 3)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run_cycle(self) -> RunResult:
        """
        Jalankan satu siklus penuh.

        Mencoba beberapa topik (max_topic_attempts) sampai mendapat SATU konten
        yang lolos ambang kualitas. Konten fallback (LLM gagal) atau di bawah
        threshold TIDAK PERNAH dikirim — siklus mencoba topik lain, atau gagal
        bersih tanpa mengirim sampah.
        """
        from database.db_manager import start_run, finish_run
        run_id = start_run()

        logger.info("=== Mulai siklus produksi (run #%d) ===", run_id)
        try:
            # Tahap 1: Kumpulkan beberapa kandidat trend
            candidates = self._scout_trends()
            if not candidates:
                logger.warning("Tidak ada trend segar, siklus dibatalkan")
                finish_run(run_id, produced=0, errors=0)
                return RunResult(success=False, reason="no_trend")

            last_reason = "no_quality_content"
            for attempt, trend in enumerate(candidates[: self.max_topic_attempts], start=1):
                logger.info("[Percobaan %d/%d] Trend: '%s' [%s]",
                            attempt, self.max_topic_attempts, trend.topic, trend.source)

                package, reason = self._produce_for_trend(trend)
                if package is None:
                    last_reason = reason
                    logger.info("  → topik di-skip (%s), coba topik lain", reason)
                    continue

                # Cek duplikat semantik
                if not self._dedup_check(package):
                    logger.warning("  → konten terlalu mirip dengan yang sudah ada, coba topik lain")
                    last_reason = "duplicate"
                    continue

                # Lolos semua gate → kirim
                content_id = self._dispatch(package)
                logger.info("Konten #%s dikirim | skor=%d verdict=%s",
                            content_id, package.score, package.verdict)
                finish_run(run_id, produced=1, errors=0)
                return RunResult(success=True, content_id=content_id)

            logger.warning("Tidak ada konten lolos kualitas dari %d topik (%s)",
                           min(len(candidates), self.max_topic_attempts), last_reason)
            finish_run(run_id, produced=0, errors=0)
            return RunResult(success=False, reason=last_reason)

        except Exception as e:
            logger.error(f"Error dalam siklus produksi: {e}", exc_info=True)
            finish_run(run_id, produced=0, errors=1)
            return RunResult(success=False, error=str(e))

    def _produce_for_trend(self, trend: TrendCandidate):
        """
        Hasilkan ContentPackage layak-kirim untuk satu trend.
        Kembalikan (package, "ok") jika lolos, atau (None, alasan) jika harus di-skip.
        """
        # Tahap 2: Pilih angle
        strategy = self._strategize(trend)
        logger.info("  Angle: %s — %s", strategy.angle_type, strategy.angle_description)

        # Tahap 3: Buat draft konten (JP + ID)
        draft = self._create(strategy)
        if draft.is_fallback:
            return None, "creator_fallback"   # LLM gagal — jangan kirim
        logger.info("  Draft JP (%d chars): %s…", len(draft.japanese), draft.japanese[:60])

        # Tahap 4-5: Loop Critic ⇄ Reviser
        package = self._critic_revise_loop(draft)
        logger.info("  Skor: %d/100 | verdict: %s | revisi: %dx",
                    package.score, package.verdict, package.revision_count)

        if package.is_fallback:
            return None, "critic_fallback"    # LLM gagal menilai — jangan kirim
        if package.below_threshold:
            return None, "below_threshold"    # kualitas kurang — jangan kirim
        return package, "ok"

    # ------------------------------------------------------------------
    # Tahap 1 — Trend Scout
    # ------------------------------------------------------------------
    def _scout_trends(self) -> List[TrendCandidate]:
        """
        Cari topik segar dari RSS feeds + Google Trends JP.
        Kembalikan list teracak agar run_cycle bisa mencoba beberapa topik.
        """
        from agents.trend_scout import scout_trends
        candidates = scout_trends()
        if not candidates:
            return []
        candidates = list(candidates)
        random.shuffle(candidates)
        return candidates

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
        best_verdict = "REJECT"
        best_breakdown: dict = {}
        revision_count = 0
        any_fallback = False

        for iteration in range(self.max_revisions + 1):
            critic = review(current_draft)
            if critic.is_fallback:
                any_fallback = True  # LLM gagal menilai

            if critic.score > best_score:
                best_score = critic.score
                best_verdict = critic.verdict
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

        # Konten dianggap fallback jika ada penilaian fallback DAN tak ada skor valid
        is_fallback = any_fallback and best_score <= 0

        return ContentPackage(
            topic=best_draft.topic,
            source_url=best_draft.source_url,
            angle_type=best_draft.angle_type,
            japanese=best_draft.japanese,
            indonesian=best_draft.indonesian,
            score=max(best_score, 0),
            verdict=best_verdict,
            score_breakdown=best_breakdown,
            below_threshold=best_score < self.threshold,
            revision_count=revision_count,
            is_fallback=is_fallback or best_draft.is_fallback,
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
