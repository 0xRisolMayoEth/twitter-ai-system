"""
orchestrator.py — Pipeline utama multi-agent.

Menjalankan state machine per siklus produksi:
  TrendScout → Strategist → Creator → Critic ⇄ Reviser (loop) → Dedup → Dispatcher

Setiap tahap merupakan method terpisah yang akan diisi implementasi
penuh di tahap upgrade masing-masing. Saat ini sebagian masih stub
yang memanggil modul lama agar sistem tetap bisa jalan.
"""
import logging
import os
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
        scoring = self.config.get("scoring", {})
        self.min_dimension_score = scoring.get("min_dimension_score", 6)
        self.max_revisions = scoring.get("max_revisions", 2)
        # Berapa banyak topik berbeda yang dicoba per siklus sebelum menyerah
        self.max_topic_attempts = scoring.get("max_topic_attempts", 3)

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
            # Tahap 1: Kumpulkan kandidat trend mentah
            candidates = self._scout_trends()
            if not candidates:
                logger.warning("Tidak ada trend segar, siklus dibatalkan")
                finish_run(run_id, produced=0, errors=0)
                return RunResult(success=False, reason="no_trend")

            # Tahap 2: TREND AGENT pilih topik salaryman-relatable (ranked)
            selections = self._select_trends(candidates)
            if not selections:
                logger.warning("Tidak ada topik relatable untuk salaryman")
                finish_run(run_id, produced=0, errors=0)
                return RunResult(success=False, reason="no_relatable_topic")

            last_reason = "no_quality_content"
            for attempt, selection in enumerate(selections[: self.max_topic_attempts], start=1):
                logger.info("[Percobaan %d/%d] Topik: '%s' [%s]",
                            attempt, self.max_topic_attempts,
                            selection.topic, selection.topic_category)

                package, reason = self._produce_for_trend(selection)
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

    def _produce_for_trend(self, selection: StrategistOutput):
        """
        Hasilkan ContentPackage layak-kirim untuk satu pilihan trend.
        Kembalikan (package, "ok") jika lolos, atau (None, alasan) jika di-skip.
        """
        # Tahap 3: WRITER — tulis tweet salaryman (JP + ID)
        draft = self._create(selection)
        if draft.is_fallback:
            return None, "creator_fallback"   # LLM gagal — jangan kirim
        logger.info("  Draft JP (%d chars): %s…", len(draft.japanese), draft.japanese[:60])

        # Tahap 4: CRITIC ⇄ rewrite otomatis
        package = self._critic_revise_loop(draft)
        logger.info("  Skor: %d/10 | verdict: %s | revisi: %dx",
                    package.score, package.verdict, package.revision_count)

        if package.is_fallback:
            return None, "critic_fallback"    # LLM gagal menilai — jangan kirim
        if package.below_threshold:
            return None, "below_threshold"    # ada dimensi < min — jangan kirim

        # Tahap 5: IMAGE AGENT — rekomendasi gambar (tidak memblok pengiriman)
        package.image = self._recommend_image(package, selection)
        return package, "ok"

    # ------------------------------------------------------------------
    # Tahap 1 — Trend Scout
    # ------------------------------------------------------------------
    def _scout_trends(self) -> List[TrendCandidate]:
        """
        Cari topik segar dari sumber trend.

        Feature flag USE_AGENT_REACH:
          - "true"  → agents.trend_reach (Twitter/X + Web + Reddit + RSS fallback)
          - lainnya → agents.trend_scout (RSS + Google Trends, perilaku lama)
        Keduanya mengekspos scout_trends() -> List[TrendCandidate] yang identik,
        jadi rollback cukup dengan mengubah satu env var.
        """
        if os.getenv("USE_AGENT_REACH", "false").lower() == "true":
            from agents.trend_reach import scout_trends
        else:
            from agents.trend_scout import scout_trends
        candidates = scout_trends()
        if not candidates:
            return []
        candidates = list(candidates)
        random.shuffle(candidates)
        return candidates

    # ------------------------------------------------------------------
    # Tahap 2 — TREND AGENT (pilih topik salaryman-relatable)
    # ------------------------------------------------------------------
    def _select_trends(self, candidates: List[TrendCandidate]) -> List[StrategistOutput]:
        """Pilih topik relatable untuk salaryman (terurut)."""
        from agents.strategist import select_trends
        return select_trends(candidates)

    # ------------------------------------------------------------------
    # Tahap 3 — WRITER (田中サトル)
    # ------------------------------------------------------------------
    def _create(self, selection: StrategistOutput) -> TweetDraft:
        """Tulis tweet salaryman JP + ID."""
        from agents.creator import create_tweet
        return create_tweet(selection)

    # ------------------------------------------------------------------
    # Tahap 5 — IMAGE AGENT
    # ------------------------------------------------------------------
    def _recommend_image(self, package: ContentPackage, selection: StrategistOutput):
        """Rekomendasi gambar pendukung (gagal → None, tidak memblok)."""
        try:
            from agents.image_agent import recommend_image
            return recommend_image(
                package.japanese, package.topic, selection.search_query_for_image
            )
        except Exception as e:
            logger.warning("Image Agent gagal: %s — lanjut tanpa gambar", e)
            return None

    # ------------------------------------------------------------------
    # Tahap 4 — Critic ⇄ Reviser loop
    # ------------------------------------------------------------------
    def _critic_revise_loop(self, draft: TweetDraft) -> ContentPackage:
        """
        Loop Critic dengan rewrite otomatis (improved_tweet) maks max_revisions.
        Lolos jika SEMUA dimensi ≥ min_dimension_score. Lacak hasil terbaik.
        """
        from agents.critic_v2 import review

        min_score = self.config["scoring"].get("min_dimension_score", 6)

        current = draft
        best = draft
        best_score = -1
        best_verdict = "REJECT"
        best_breakdown: dict = {}
        revision_count = 0
        any_fallback = False

        for iteration in range(self.max_revisions + 1):
            critic = review(current)
            if critic.is_fallback:
                any_fallback = True

            if critic.score > best_score:
                best_score = critic.score
                best_verdict = critic.verdict
                best_breakdown = critic.breakdown
                best = current

            # Lolos: semua dimensi ≥ min
            if critic.breakdown and all(v >= min_score for v in critic.breakdown.values()):
                logger.info("Critic loop lolos di iterasi %d (semua dimensi ≥ %d)",
                            iteration, min_score)
                break

            if iteration == self.max_revisions:
                break

            # Rewrite otomatis pakai improved_tweet dari Critic
            if not critic.improved_tweet:
                break  # tidak ada perbaikan yang ditawarkan
            current = current.model_copy(update={"japanese": critic.improved_tweet})
            revision_count += 1

        all_pass = bool(best_breakdown) and all(v >= min_score for v in best_breakdown.values())
        is_fallback = (any_fallback and best_score <= 0) or best.is_fallback

        return ContentPackage(
            topic=best.topic,
            source_url=best.source_url,
            angle_type=best.angle_type,
            japanese=best.japanese,
            indonesian=best.indonesian,
            score=max(best_score, 0),
            verdict=best_verdict,
            score_breakdown=best_breakdown,
            tone=best.tone,
            best_posting_time=best.best_posting_time,
            below_threshold=not all_pass,
            revision_count=revision_count,
            is_fallback=is_fallback,
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
