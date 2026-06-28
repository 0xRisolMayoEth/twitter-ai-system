"""
tests/test_critic.py — Unit test untuk CRITIC AGENT (agents/critic_v2.py).

LLM di-mock. Fokus: scoring 4 dimensi (1–10), aturan REJECT bila ada
dimensi < min, rewrite otomatis (improved_tweet), dan fallback.
"""
import json
import unittest
from unittest import mock

from core.models import CriticResult, TweetDraft
from agents import critic_v2 as cv

_CFG = {"scoring": {"min_dimension_score": 6}}


def _draft(jp="満員電車つらい😮‍💨 #サラリーマン"):
    return TweetDraft(japanese=jp, indonesian="kereta penuh",
                      topic="満員電車", angle_type="commute", source_url="http://x")


def _scores(rel=8, nat=7, eng=6, fit=9):
    return {"relatability": rel, "naturalness": nat, "engagement": eng, "topic_fit": fit}


def _resp(scores=None, verdict="APPROVE", improved="", feedback="bagus"):
    return json.dumps({"scores": scores or _scores(), "verdict": verdict,
                       "improved_tweet": improved, "feedback": feedback})


class TestParse(unittest.TestCase):
    def test_all_pass_is_approve(self):
        r = cv._parse(_resp(_scores(8, 8, 7, 9)), 6)
        self.assertEqual(r.verdict, "APPROVE")
        self.assertEqual(r.score, 8)  # round(mean(8,8,7,9)=8.0)

    def test_one_dim_below_min_is_reject(self):
        r = cv._parse(_resp(_scores(8, 3, 7, 9), verdict="APPROVE", improved="better #あるある"), 6)
        self.assertEqual(r.verdict, "REJECT")  # aturan ambang menimpa verdict LLM
        self.assertEqual(r.improved_tweet, "better #あるある")

    def test_score_is_rounded_mean(self):
        r = cv._parse(_resp(_scores(10, 9, 8, 9)), 6)
        self.assertEqual(r.score, 9)  # mean=9.0

    def test_dims_clamped_to_10(self):
        r = cv._parse(_resp(_scores(99, 99, 99, 99)), 6)
        self.assertTrue(all(v == 10 for v in r.breakdown.values()))

    def test_missing_dim_defaults_zero(self):
        raw = json.dumps({"scores": {"relatability": 8}, "verdict": "APPROVE"})
        r = cv._parse(raw, 6)
        self.assertEqual(r.breakdown["naturalness"], 0)
        self.assertEqual(r.verdict, "REJECT")

    def test_improved_tweet_capped(self):
        r = cv._parse(_resp(_scores(1, 1, 1, 1), improved="あ" * 200), 6)
        self.assertLessEqual(len(r.improved_tweet), 140)

    def test_invalid_json_falls_back(self):
        r = cv._parse("bukan json", 6)
        self.assertEqual(r.score, 0)
        self.assertTrue(r.is_fallback)


class TestFallback(unittest.TestCase):
    def test_fallback_score_zero_rejected(self):
        r = cv._fallback()
        self.assertEqual(r.score, 0)
        self.assertEqual(r.verdict, "REJECT")
        self.assertTrue(r.is_fallback)

    def test_fallback_has_all_dimensions(self):
        r = cv._fallback()
        for d in cv.DIMENSIONS:
            self.assertIn(d, r.breakdown)


class TestReview(unittest.TestCase):
    def test_successful_review(self):
        with mock.patch("agents.critic_v2.load_config", return_value=_CFG), \
             mock.patch("agents.critic_v2.chat", return_value=_resp()):
            r = cv.review(_draft())
        self.assertEqual(r.verdict, "APPROVE")
        self.assertGreater(r.score, 0)

    def test_llm_error_returns_fallback(self):
        with mock.patch("agents.critic_v2.load_config", return_value=_CFG), \
             mock.patch("agents.critic_v2.chat", side_effect=Exception("API down")):
            r = cv.review(_draft())
        self.assertTrue(r.is_fallback)
        self.assertEqual(r.score, 0)

    def test_prompt_contains_tweet_and_dimensions(self):
        captured = []

        def fake_chat(messages, **kw):
            captured.extend(messages)
            return _resp()

        with mock.patch("agents.critic_v2.load_config", return_value=_CFG), \
             mock.patch("agents.critic_v2.chat", side_effect=fake_chat):
            cv.review(_draft(jp="残業なう😴 #あるある"))
        content = " ".join(m.get("content", "") for m in captured)
        self.assertIn("残業なう", content)
        for d in cv.DIMENSIONS:
            self.assertIn(d, content)


if __name__ == "__main__":
    unittest.main()
