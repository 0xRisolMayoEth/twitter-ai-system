"""
tests/test_orchestrator.py — Test alur kualitas run_cycle.

Fokus: konten fallback / di bawah threshold TIDAK PERNAH dikirim, dan
run_cycle mencoba beberapa topik sebelum menyerah.
"""
import unittest
from unittest import mock

from core.models import (
    TrendCandidate, StrategistOutput, TweetDraft, ContentPackage, RunResult,
)


def _orch(threshold=75, max_revisions=2, max_topic_attempts=3):
    """Bangun Orchestrator tanpa __init__ (hindari load_config nyata)."""
    from orchestrator import Orchestrator
    o = Orchestrator.__new__(Orchestrator)
    o.config = {"scoring": {"threshold": threshold, "max_revisions": max_revisions,
                            "max_topic_attempts": max_topic_attempts},
                "dedup": {"lookback_days": 30, "similarity_threshold": 0.85},
                "niche": "test"}
    o.threshold = threshold
    o.max_revisions = max_revisions
    o.max_topic_attempts = max_topic_attempts
    return o


def _trend(topic="トピック"):
    return TrendCandidate(topic=topic, source="test", url="http://x.co")


def _draft(jp="良いツイート", is_fallback=False):
    return TweetDraft(japanese=jp, indonesian="bagus", topic="t",
                      angle_type="news_insight", is_fallback=is_fallback)


def _package(score=85, verdict="GOOD", below=False, is_fallback=False):
    return ContentPackage(
        topic="t", source_url="http://x.co", angle_type="news_insight",
        japanese="JP", indonesian="ID", score=score, verdict=verdict,
        below_threshold=below, is_fallback=is_fallback,
    )


class TestQualityGating(unittest.TestCase):

    def _patch_db(self):
        return mock.patch.multiple(
            "database.db_manager",
            start_run=mock.DEFAULT, finish_run=mock.DEFAULT,
        )

    def test_good_content_dispatched(self):
        o = _orch()
        with mock.patch("database.db_manager.start_run", return_value=1), \
             mock.patch("database.db_manager.finish_run"), \
             mock.patch.object(o, "_scout_trends", return_value=[_trend("A")]), \
             mock.patch.object(o, "_strategize", return_value=mock.MagicMock()), \
             mock.patch.object(o, "_create", return_value=_draft()), \
             mock.patch.object(o, "_critic_revise_loop", return_value=_package(score=88, verdict="GOOD")), \
             mock.patch.object(o, "_dedup_check", return_value=True), \
             mock.patch.object(o, "_dispatch", return_value=42) as mock_dispatch:
            result = o.run_cycle()

        self.assertTrue(result.success)
        self.assertEqual(result.content_id, 42)
        mock_dispatch.assert_called_once()

    def test_creator_fallback_not_dispatched(self):
        """Draft fallback (LLM gagal) tidak boleh dikirim."""
        o = _orch(max_topic_attempts=1)
        with mock.patch("database.db_manager.start_run", return_value=1), \
             mock.patch("database.db_manager.finish_run"), \
             mock.patch.object(o, "_scout_trends", return_value=[_trend("A")]), \
             mock.patch.object(o, "_strategize", return_value=mock.MagicMock()), \
             mock.patch.object(o, "_create", return_value=_draft(is_fallback=True)), \
             mock.patch.object(o, "_dispatch") as mock_dispatch:
            result = o.run_cycle()

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "creator_fallback")
        mock_dispatch.assert_not_called()

    def test_below_threshold_not_dispatched(self):
        o = _orch(max_topic_attempts=1)
        with mock.patch("database.db_manager.start_run", return_value=1), \
             mock.patch("database.db_manager.finish_run"), \
             mock.patch.object(o, "_scout_trends", return_value=[_trend("A")]), \
             mock.patch.object(o, "_strategize", return_value=mock.MagicMock()), \
             mock.patch.object(o, "_create", return_value=_draft()), \
             mock.patch.object(o, "_critic_revise_loop",
                               return_value=_package(score=60, verdict="REVISE", below=True)), \
             mock.patch.object(o, "_dispatch") as mock_dispatch:
            result = o.run_cycle()

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "below_threshold")
        mock_dispatch.assert_not_called()

    def test_critic_fallback_not_dispatched(self):
        o = _orch(max_topic_attempts=1)
        with mock.patch("database.db_manager.start_run", return_value=1), \
             mock.patch("database.db_manager.finish_run"), \
             mock.patch.object(o, "_scout_trends", return_value=[_trend("A")]), \
             mock.patch.object(o, "_strategize", return_value=mock.MagicMock()), \
             mock.patch.object(o, "_create", return_value=_draft()), \
             mock.patch.object(o, "_critic_revise_loop",
                               return_value=_package(score=0, verdict="REJECT", below=True, is_fallback=True)), \
             mock.patch.object(o, "_dispatch") as mock_dispatch:
            result = o.run_cycle()

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "critic_fallback")
        mock_dispatch.assert_not_called()

    def test_tries_next_topic_after_bad_one(self):
        """Topik pertama buruk → coba topik kedua yang bagus → dispatch."""
        o = _orch(max_topic_attempts=3)

        # Topik 1 below threshold, topik 2 bagus
        packages = [
            _package(score=50, verdict="REJECT", below=True),
            _package(score=90, verdict="APPROVED"),
        ]
        call = [0]
        def fake_loop(draft):
            p = packages[call[0]]
            call[0] += 1
            return p

        with mock.patch("database.db_manager.start_run", return_value=1), \
             mock.patch("database.db_manager.finish_run"), \
             mock.patch.object(o, "_scout_trends", return_value=[_trend("A"), _trend("B")]), \
             mock.patch.object(o, "_strategize", return_value=mock.MagicMock()), \
             mock.patch.object(o, "_create", return_value=_draft()), \
             mock.patch.object(o, "_critic_revise_loop", side_effect=fake_loop), \
             mock.patch.object(o, "_dedup_check", return_value=True), \
             mock.patch.object(o, "_dispatch", return_value=7) as mock_dispatch:
            result = o.run_cycle()

        self.assertTrue(result.success)
        self.assertEqual(result.content_id, 7)
        self.assertEqual(call[0], 2)  # dua topik dievaluasi
        mock_dispatch.assert_called_once()

    def test_all_topics_fail_no_dispatch(self):
        o = _orch(max_topic_attempts=3)
        with mock.patch("database.db_manager.start_run", return_value=1), \
             mock.patch("database.db_manager.finish_run"), \
             mock.patch.object(o, "_scout_trends", return_value=[_trend("A"), _trend("B"), _trend("C")]), \
             mock.patch.object(o, "_strategize", return_value=mock.MagicMock()), \
             mock.patch.object(o, "_create", return_value=_draft()), \
             mock.patch.object(o, "_critic_revise_loop",
                               return_value=_package(score=40, verdict="REJECT", below=True)), \
             mock.patch.object(o, "_dispatch") as mock_dispatch:
            result = o.run_cycle()

        self.assertFalse(result.success)
        mock_dispatch.assert_not_called()

    def test_no_trend_returns_failure(self):
        o = _orch()
        with mock.patch("database.db_manager.start_run", return_value=1), \
             mock.patch("database.db_manager.finish_run"), \
             mock.patch.object(o, "_scout_trends", return_value=[]), \
             mock.patch.object(o, "_dispatch") as mock_dispatch:
            result = o.run_cycle()

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "no_trend")
        mock_dispatch.assert_not_called()

    def test_duplicate_skipped(self):
        """Konten bagus tapi duplikat → coba topik lain."""
        o = _orch(max_topic_attempts=1)
        with mock.patch("database.db_manager.start_run", return_value=1), \
             mock.patch("database.db_manager.finish_run"), \
             mock.patch.object(o, "_scout_trends", return_value=[_trend("A")]), \
             mock.patch.object(o, "_strategize", return_value=mock.MagicMock()), \
             mock.patch.object(o, "_create", return_value=_draft()), \
             mock.patch.object(o, "_critic_revise_loop", return_value=_package(score=88)), \
             mock.patch.object(o, "_dedup_check", return_value=False), \
             mock.patch.object(o, "_dispatch") as mock_dispatch:
            result = o.run_cycle()

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "duplicate")
        mock_dispatch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
