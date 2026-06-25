"""
tests/test_critic.py — Unit test untuk agents/critic_v2.py

LLM di-mock agar test cepat dan deterministik.
"""
import json
import unittest
from unittest import mock

from core.models import CriticResult, TweetDraft

_WEIGHTS = {
    "hook": 25,
    "engagement": 20,
    "naturalness": 20,
    "japanese_quality": 15,
    "relevance": 10,
    "format": 10,
}

_CFG = {
    "niche": "AI & tech Japan",
    "scoring": {"weights": _WEIGHTS},
}


def _make_draft(jp="AIの話題ツイートです。", indo="Tweet Indonesia."):
    return TweetDraft(
        japanese=jp,
        indonesian=indo,
        topic="AI最新情報",
        angle_type="news_insight",
        source_url="http://example.com",
    )


def _full_breakdown(hook=20, eng=16, nat=17, jpq=12, rel=8, fmt=9):
    return {
        "hook": hook, "engagement": eng, "naturalness": nat,
        "japanese_quality": jpq, "relevance": rel, "format": fmt,
    }


def _llm_response(breakdown=None, issues=None, suggestions=None):
    bd = breakdown or _full_breakdown()
    return json.dumps({
        "breakdown": bd,
        "issues": issues or ["Hook kurang kuat"],
        "suggestions": suggestions or ["Buat kalimat pertama lebih mengejutkan"],
    })


class TestFallback(unittest.TestCase):
    def test_fallback_score_is_zero(self):
        """Fallback sekarang skor 0 (bukan 73) agar konten tidak ikut terkirim."""
        from agents.critic_v2 import _fallback
        result = _fallback(_WEIGHTS)
        self.assertEqual(result.score, 0)

    def test_fallback_marked_and_rejected(self):
        from agents.critic_v2 import _fallback
        result = _fallback(_WEIGHTS)
        self.assertTrue(result.is_fallback)
        self.assertEqual(result.verdict, "REJECT")

    def test_fallback_has_all_aspects(self):
        from agents.critic_v2 import _fallback
        result = _fallback(_WEIGHTS)
        for k in _WEIGHTS:
            self.assertIn(k, result.breakdown)

    def test_fallback_non_empty_issues(self):
        from agents.critic_v2 import _fallback
        result = _fallback(_WEIGHTS)
        self.assertGreater(len(result.issues), 0)


class TestParse(unittest.TestCase):
    def test_valid_response_parsed(self):
        from agents.critic_v2 import _parse
        raw = _llm_response(issues=["Masalah A"], suggestions=["Saran B"])
        result = _parse(raw, _WEIGHTS)
        self.assertIsInstance(result, CriticResult)
        self.assertEqual(result.score, sum(_full_breakdown().values()))
        self.assertEqual(result.issues, ["Masalah A"])
        self.assertEqual(result.suggestions, ["Saran B"])

    def test_score_is_sum_of_breakdown(self):
        from agents.critic_v2 import _parse
        bd = {"hook": 20, "engagement": 15, "naturalness": 18, "japanese_quality": 12, "relevance": 9, "format": 8}
        raw = json.dumps({"breakdown": bd, "issues": [], "suggestions": []})
        result = _parse(raw, _WEIGHTS)
        self.assertEqual(result.score, sum(bd.values()))

    def test_aspect_clamped_to_max_weight(self):
        """Skor melebihi batas → diclamped ke maks."""
        from agents.critic_v2 import _parse
        bd = {"hook": 999, "engagement": 999, "naturalness": 999,
              "japanese_quality": 999, "relevance": 999, "format": 999}
        raw = json.dumps({"breakdown": bd, "issues": [], "suggestions": []})
        result = _parse(raw, _WEIGHTS)
        self.assertEqual(result.score, 100)  # sum of all maxes = 100
        self.assertEqual(result.breakdown["hook"], 25)

    def test_negative_score_clamped_to_zero(self):
        from agents.critic_v2 import _parse
        bd = {"hook": -5, "engagement": 0, "naturalness": 0,
              "japanese_quality": 0, "relevance": 0, "format": 0}
        raw = json.dumps({"breakdown": bd, "issues": [], "suggestions": []})
        result = _parse(raw, _WEIGHTS)
        self.assertEqual(result.breakdown["hook"], 0)

    def test_json_embedded_in_text(self):
        from agents.critic_v2 import _parse
        raw = 'Hasil review:\n' + _llm_response() + '\nSekian.'
        result = _parse(raw, _WEIGHTS)
        self.assertIsInstance(result, CriticResult)

    def test_invalid_json_falls_back(self):
        from agents.critic_v2 import _parse
        result = _parse("ini bukan json", _WEIGHTS)
        self.assertIsInstance(result, CriticResult)
        self.assertEqual(result.score, 0)
        self.assertTrue(result.is_fallback)

    def test_missing_aspects_default_to_zero(self):
        """Jika LLM hanya kasih sebagian aspek, sisanya 0."""
        from agents.critic_v2 import _parse
        raw = json.dumps({"breakdown": {"hook": 20}, "issues": [], "suggestions": []})
        result = _parse(raw, _WEIGHTS)
        self.assertEqual(result.breakdown["hook"], 20)
        self.assertEqual(result.breakdown.get("engagement", 0), 0)

    def test_issues_capped_at_five(self):
        from agents.critic_v2 import _parse
        issues = [f"Masalah {i}" for i in range(10)]
        raw = json.dumps({"breakdown": _full_breakdown(), "issues": issues, "suggestions": []})
        result = _parse(raw, _WEIGHTS)
        self.assertLessEqual(len(result.issues), 5)


class TestReview(unittest.TestCase):
    """Test review() end-to-end dengan LLM mock."""

    def test_successful_review(self):
        from agents.critic_v2 import review
        draft = _make_draft()

        with mock.patch("agents.critic_v2.load_config", return_value=_CFG), \
             mock.patch("agents.critic_v2.chat", return_value=_llm_response()):
            result = review(draft)

        self.assertIsInstance(result, CriticResult)
        self.assertGreater(result.score, 0)
        self.assertIn("hook", result.breakdown)

    def test_llm_error_returns_fallback(self):
        from agents.critic_v2 import review
        draft = _make_draft()

        with mock.patch("agents.critic_v2.load_config", return_value=_CFG), \
             mock.patch("agents.critic_v2.chat", side_effect=Exception("API error")):
            result = review(draft)

        self.assertIsInstance(result, CriticResult)
        self.assertEqual(result.score, 0)
        self.assertTrue(result.is_fallback)

    def test_high_quality_tweet_gets_high_score(self):
        from agents.critic_v2 import review
        draft = _make_draft()
        perfect_breakdown = {
            "hook": 25, "engagement": 20, "naturalness": 20,
            "japanese_quality": 15, "relevance": 10, "format": 10,
        }

        with mock.patch("agents.critic_v2.load_config", return_value=_CFG), \
             mock.patch("agents.critic_v2.chat",
                        return_value=json.dumps({"breakdown": perfect_breakdown,
                                                 "issues": [], "suggestions": []})):
            result = review(draft)

        self.assertEqual(result.score, 100)

    def test_prompt_contains_tweet_text(self):
        from agents.critic_v2 import review
        draft = _make_draft(jp="これはテスト用ツイート")
        captured = []

        def fake_chat(messages, **kwargs):
            captured.extend(messages)
            return _llm_response()

        with mock.patch("agents.critic_v2.load_config", return_value=_CFG), \
             mock.patch("agents.critic_v2.chat", side_effect=fake_chat):
            review(draft)

        all_content = " ".join(m.get("content", "") for m in captured)
        self.assertIn("これはテスト用ツイート", all_content)

    def test_prompt_contains_all_aspects(self):
        from agents.critic_v2 import review
        draft = _make_draft()
        captured = []

        def fake_chat(messages, **kwargs):
            captured.extend(messages)
            return _llm_response()

        with mock.patch("agents.critic_v2.load_config", return_value=_CFG), \
             mock.patch("agents.critic_v2.chat", side_effect=fake_chat):
            review(draft)

        all_content = " ".join(m.get("content", "") for m in captured)
        for aspect in _WEIGHTS:
            self.assertIn(aspect, all_content)


class TestBuildPrompt(unittest.TestCase):
    def test_prompt_contains_tweet(self):
        from agents.critic_v2 import _build_prompt
        draft = _make_draft(jp="テストツイート本文")
        prompt = _build_prompt(draft, _WEIGHTS, "AI")
        self.assertIn("テストツイート本文", prompt)

    def test_prompt_contains_char_count(self):
        from agents.critic_v2 import _build_prompt
        jp = "あいう"
        draft = _make_draft(jp=jp)
        prompt = _build_prompt(draft, _WEIGHTS, "")
        self.assertIn(str(len(jp)), prompt)

    def test_prompt_requests_json_output(self):
        from agents.critic_v2 import _build_prompt
        draft = _make_draft()
        prompt = _build_prompt(draft, _WEIGHTS, "")
        self.assertIn("breakdown", prompt)
        self.assertIn("issues", prompt)
        self.assertIn("suggestions", prompt)


if __name__ == "__main__":
    unittest.main()
