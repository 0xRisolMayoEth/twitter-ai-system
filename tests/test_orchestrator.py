"""
tests/test_orchestrator.py — Test alur kualitas run_cycle (pipeline salaryman).

Fokus: konten fallback / di bawah ambang TIDAK PERNAH dikirim, run_cycle
mencoba beberapa topik, dan Image Agent dipanggil sebelum dispatch.
"""
import unittest
from unittest import mock

from core.models import (
    TrendCandidate, StrategistOutput, TweetDraft, ContentPackage, ImageRec,
)


def _orch(min_score=6, max_revisions=2, max_topic_attempts=3):
    from orchestrator import Orchestrator
    o = Orchestrator.__new__(Orchestrator)
    o.config = {"scoring": {"min_dimension_score": min_score, "max_revisions": max_revisions,
                            "max_topic_attempts": max_topic_attempts},
                "dedup": {"lookback_days": 30, "similarity_threshold": 0.85},
                "niche": "salaryman"}
    o.min_dimension_score = min_score
    o.max_revisions = max_revisions
    o.max_topic_attempts = max_topic_attempts
    return o


def _trend(topic="満員電車"):
    return TrendCandidate(topic=topic, source="test", url="http://x")


def _sel(topic="満員電車"):
    return StrategistOutput(topic=topic, topic_category="commute",
                            why_relatable="毎朝", search_query_for_image="train",
                            angle_type="commute")


def _draft(is_fallback=False):
    return TweetDraft(japanese="満員電車つらい #サラリーマン", indonesian="kereta",
                      topic="満員電車", angle_type="commute", is_fallback=is_fallback)


def _package(score=8, verdict="APPROVE", below=False, is_fallback=False):
    return ContentPackage(topic="満員電車", source_url="http://x", angle_type="commute",
                          japanese="満員電車つらい #サラリーマン", indonesian="kereta",
                          score=score, verdict=verdict, below_threshold=below,
                          is_fallback=is_fallback)


def _img():
    return ImageRec(google_search_queries=["a", "b", "日本"])


class TestQualityGating(unittest.TestCase):

    def test_good_content_dispatched(self):
        o = _orch()
        with mock.patch("database.db_manager.start_run", return_value=1), \
             mock.patch("database.db_manager.finish_run"), \
             mock.patch.object(o, "_scout_trends", return_value=[_trend()]), \
             mock.patch.object(o, "_select_trends", return_value=[_sel("A")]), \
             mock.patch.object(o, "_create", return_value=_draft()), \
             mock.patch.object(o, "_critic_revise_loop", return_value=_package()), \
             mock.patch.object(o, "_recommend_image", return_value=_img()), \
             mock.patch.object(o, "_dedup_check", return_value=True), \
             mock.patch.object(o, "_dispatch", return_value=42) as md:
            result = o.run_cycle()
        self.assertTrue(result.success)
        self.assertEqual(result.content_id, 42)
        md.assert_called_once()

    def test_image_attached_before_dispatch(self):
        o = _orch()
        captured = {}
        with mock.patch("database.db_manager.start_run", return_value=1), \
             mock.patch("database.db_manager.finish_run"), \
             mock.patch.object(o, "_scout_trends", return_value=[_trend()]), \
             mock.patch.object(o, "_select_trends", return_value=[_sel("A")]), \
             mock.patch.object(o, "_create", return_value=_draft()), \
             mock.patch.object(o, "_critic_revise_loop", return_value=_package()), \
             mock.patch.object(o, "_recommend_image", return_value=_img()), \
             mock.patch.object(o, "_dedup_check", return_value=True), \
             mock.patch.object(o, "_dispatch", side_effect=lambda p: captured.setdefault("img", p.image) or 1):
            o.run_cycle()
        self.assertIsNotNone(captured["img"])
        self.assertEqual(len(captured["img"].google_search_queries), 3)

    def test_creator_fallback_not_dispatched(self):
        o = _orch(max_topic_attempts=1)
        with mock.patch("database.db_manager.start_run", return_value=1), \
             mock.patch("database.db_manager.finish_run"), \
             mock.patch.object(o, "_scout_trends", return_value=[_trend()]), \
             mock.patch.object(o, "_select_trends", return_value=[_sel("A")]), \
             mock.patch.object(o, "_create", return_value=_draft(is_fallback=True)), \
             mock.patch.object(o, "_dispatch") as md:
            result = o.run_cycle()
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "creator_fallback")
        md.assert_not_called()

    def test_below_threshold_not_dispatched(self):
        o = _orch(max_topic_attempts=1)
        with mock.patch("database.db_manager.start_run", return_value=1), \
             mock.patch("database.db_manager.finish_run"), \
             mock.patch.object(o, "_scout_trends", return_value=[_trend()]), \
             mock.patch.object(o, "_select_trends", return_value=[_sel("A")]), \
             mock.patch.object(o, "_create", return_value=_draft()), \
             mock.patch.object(o, "_critic_revise_loop",
                               return_value=_package(score=4, verdict="REJECT", below=True)), \
             mock.patch.object(o, "_dispatch") as md:
            result = o.run_cycle()
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "below_threshold")
        md.assert_not_called()

    def test_critic_fallback_not_dispatched(self):
        o = _orch(max_topic_attempts=1)
        with mock.patch("database.db_manager.start_run", return_value=1), \
             mock.patch("database.db_manager.finish_run"), \
             mock.patch.object(o, "_scout_trends", return_value=[_trend()]), \
             mock.patch.object(o, "_select_trends", return_value=[_sel("A")]), \
             mock.patch.object(o, "_create", return_value=_draft()), \
             mock.patch.object(o, "_critic_revise_loop",
                               return_value=_package(score=0, verdict="REJECT", below=True, is_fallback=True)), \
             mock.patch.object(o, "_dispatch") as md:
            result = o.run_cycle()
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "critic_fallback")
        md.assert_not_called()

    def test_tries_next_topic_after_bad(self):
        o = _orch(max_topic_attempts=3)
        packages = [_package(score=3, verdict="REJECT", below=True), _package(score=9)]
        call = [0]

        def fake_loop(draft):
            p = packages[call[0]]
            call[0] += 1
            return p

        with mock.patch("database.db_manager.start_run", return_value=1), \
             mock.patch("database.db_manager.finish_run"), \
             mock.patch.object(o, "_scout_trends", return_value=[_trend()]), \
             mock.patch.object(o, "_select_trends", return_value=[_sel("A"), _sel("B")]), \
             mock.patch.object(o, "_create", return_value=_draft()), \
             mock.patch.object(o, "_critic_revise_loop", side_effect=fake_loop), \
             mock.patch.object(o, "_recommend_image", return_value=_img()), \
             mock.patch.object(o, "_dedup_check", return_value=True), \
             mock.patch.object(o, "_dispatch", return_value=7) as md:
            result = o.run_cycle()
        self.assertTrue(result.success)
        self.assertEqual(call[0], 2)
        md.assert_called_once()

    def test_no_trend(self):
        o = _orch()
        with mock.patch("database.db_manager.start_run", return_value=1), \
             mock.patch("database.db_manager.finish_run"), \
             mock.patch.object(o, "_scout_trends", return_value=[]), \
             mock.patch.object(o, "_dispatch") as md:
            result = o.run_cycle()
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "no_trend")
        md.assert_not_called()

    def test_no_relatable_topic(self):
        o = _orch()
        with mock.patch("database.db_manager.start_run", return_value=1), \
             mock.patch("database.db_manager.finish_run"), \
             mock.patch.object(o, "_scout_trends", return_value=[_trend()]), \
             mock.patch.object(o, "_select_trends", return_value=[]), \
             mock.patch.object(o, "_dispatch") as md:
            result = o.run_cycle()
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "no_relatable_topic")
        md.assert_not_called()

    def test_duplicate_skipped(self):
        o = _orch(max_topic_attempts=1)
        with mock.patch("database.db_manager.start_run", return_value=1), \
             mock.patch("database.db_manager.finish_run"), \
             mock.patch.object(o, "_scout_trends", return_value=[_trend()]), \
             mock.patch.object(o, "_select_trends", return_value=[_sel("A")]), \
             mock.patch.object(o, "_create", return_value=_draft()), \
             mock.patch.object(o, "_critic_revise_loop", return_value=_package()), \
             mock.patch.object(o, "_recommend_image", return_value=_img()), \
             mock.patch.object(o, "_dedup_check", return_value=False), \
             mock.patch.object(o, "_dispatch") as md:
            result = o.run_cycle()
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "duplicate")
        md.assert_not_called()


class TestCriticReviseLoop(unittest.TestCase):
    """Loop Critic + rewrite otomatis (improved_tweet)."""

    def _draft(self):
        return TweetDraft(japanese="元ツイート #サラリーマン", indonesian="asli",
                          topic="満員電車", angle_type="commute")

    def _critic(self, dims, improved=""):
        from core.models import CriticResult
        verdict = "APPROVE" if all(v >= 6 for v in dims.values()) else "REJECT"
        return CriticResult(score=round(sum(dims.values()) / len(dims)),
                            verdict=verdict, breakdown=dims, improved_tweet=improved)

    def test_all_pass_no_revision(self):
        o = _orch()
        good = self._critic({"relatability": 8, "naturalness": 8, "engagement": 7, "topic_fit": 9})
        with mock.patch("agents.critic_v2.review", return_value=good) as mr:
            pkg = o._critic_revise_loop(self._draft())
        self.assertFalse(pkg.below_threshold)
        self.assertEqual(pkg.revision_count, 0)
        self.assertEqual(mr.call_count, 1)

    def test_auto_rewrite_uses_improved_tweet(self):
        o = _orch(max_revisions=2)
        bad = self._critic({"relatability": 8, "naturalness": 3, "engagement": 7, "topic_fit": 8},
                           improved="改善版 #あるある")
        good = self._critic({"relatability": 8, "naturalness": 8, "engagement": 7, "topic_fit": 8})
        seq = [bad, good]
        calls = []

        def fake_review(draft):
            calls.append(draft.japanese)
            return seq[len(calls) - 1]

        with mock.patch("agents.critic_v2.review", side_effect=fake_review):
            pkg = o._critic_revise_loop(self._draft())
        self.assertFalse(pkg.below_threshold)
        self.assertEqual(pkg.revision_count, 1)
        self.assertEqual(calls[1], "改善版 #あるある")  # iterasi ke-2 pakai improved_tweet

    def test_below_threshold_when_never_passes(self):
        o = _orch(max_revisions=1)
        bad = self._critic({"relatability": 4, "naturalness": 4, "engagement": 4, "topic_fit": 4},
                           improved="まだダメ #あるある")
        with mock.patch("agents.critic_v2.review", return_value=bad):
            pkg = o._critic_revise_loop(self._draft())
        self.assertTrue(pkg.below_threshold)

    def test_critic_fallback_flagged(self):
        o = _orch()
        from core.models import CriticResult
        fb = CriticResult(score=0, verdict="REJECT",
                          breakdown={"relatability": 0, "naturalness": 0,
                                     "engagement": 0, "topic_fit": 0}, is_fallback=True)
        with mock.patch("agents.critic_v2.review", return_value=fb):
            pkg = o._critic_revise_loop(self._draft())
        self.assertTrue(pkg.is_fallback)


if __name__ == "__main__":
    unittest.main()
