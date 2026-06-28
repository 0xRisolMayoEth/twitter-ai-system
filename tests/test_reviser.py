"""
tests/test_reviser.py — Unit test untuk agents/reviser.py

LLM di-mock agar test cepat dan tidak butuh API key.
"""
import json
import unittest
from unittest import mock

from core.models import CriticResult, TweetDraft

_CFG = {
    "niche": "AI & tech",
    "persona": {"name": "TestBot", "description": "Bot untuk testing"},
    "tweet_register": "casual",
    "scoring": {"weights": {
        "hook": 25, "engagement": 20, "naturalness": 20,
        "japanese_quality": 15, "relevance": 10, "format": 10,
    }},
}


def _make_draft(jp="元のツイートです。", indo="Tweet asli Indonesia."):
    return TweetDraft(
        japanese=jp,
        indonesian=indo,
        topic="AIニュース",
        angle_type="curiosity_gap",
        source_url="http://example.com",
    )


def _make_critic(score=65, issues=None, suggestions=None):
    return CriticResult(
        score=score,
        breakdown={"hook": 12, "engagement": 13, "naturalness": 14,
                   "japanese_quality": 10, "relevance": 8, "format": 8},
        issues=issues or ["Hook kurang menarik", "Terlalu formal"],
        suggestions=suggestions or ["Mulai dengan pertanyaan", "Pakai kata netizen"],
    )


def _llm_response(jp="改善したツイートです！", indo="Tweet yang sudah diperbaiki!"):
    return json.dumps({"japanese": jp, "indonesian": indo})


class TestParse(unittest.TestCase):
    def test_valid_response_parsed(self):
        from agents.reviser import _parse
        original = _make_draft()
        raw = _llm_response("新しいツイート", "Tweet baru")
        result = _parse(raw, original)
        self.assertEqual(result.japanese, "新しいツイート")
        self.assertEqual(result.indonesian, "Tweet baru")

    def test_json_embedded_in_text(self):
        from agents.reviser import _parse
        original = _make_draft()
        raw = 'Hasil revisi:\n{"japanese": "改善", "indonesian": "Revisi"}\nSelesai.'
        result = _parse(raw, original)
        self.assertEqual(result.japanese, "改善")

    def test_empty_japanese_returns_original(self):
        from agents.reviser import _parse
        original = _make_draft(jp="元のツイート")
        raw = json.dumps({"japanese": "", "indonesian": "Indo"})
        result = _parse(raw, original)
        self.assertEqual(result.japanese, "元のツイート")

    def test_jp_truncated_at_140(self):
        from agents.reviser import _parse, JP_MAX
        original = _make_draft()
        long_jp = "あ" * 200
        raw = _llm_response(jp=long_jp)
        result = _parse(raw, original)
        self.assertLessEqual(len(result.japanese), JP_MAX)

    def test_indo_truncated_at_280(self):
        from agents.reviser import _parse, ID_MAX
        original = _make_draft()
        long_indo = "a" * 400
        raw = _llm_response(indo=long_indo)
        result = _parse(raw, original)
        self.assertLessEqual(len(result.indonesian), ID_MAX)

    def test_empty_indonesian_falls_back_to_original_indo(self):
        from agents.reviser import _parse
        original = _make_draft(indo="Fallback Indonesia")
        raw = json.dumps({"japanese": "日本語", "indonesian": ""})
        result = _parse(raw, original)
        self.assertEqual(result.indonesian, "Fallback Indonesia")

    def test_invalid_json_returns_original(self):
        from agents.reviser import _parse
        original = _make_draft(jp="オリジナル")
        result = _parse("bukan json", original)
        self.assertEqual(result.japanese, "オリジナル")

    def test_topic_and_angle_preserved(self):
        from agents.reviser import _parse
        original = _make_draft()
        raw = _llm_response("改善版", "Perbaikan")
        result = _parse(raw, original)
        self.assertEqual(result.topic, original.topic)
        self.assertEqual(result.angle_type, original.angle_type)
        self.assertEqual(result.source_url, original.source_url)


class TestRevise(unittest.TestCase):
    """Test revise() end-to-end dengan LLM mock."""

    def test_successful_revision(self):
        from agents.reviser import revise
        draft = _make_draft()
        critic = _make_critic()

        with mock.patch("agents.reviser.load_config", return_value=_CFG), \
             mock.patch("agents.reviser.chat", return_value=_llm_response("改善されたツイート！", "Sudah diperbaiki!")):
            result = revise(draft, critic)

        self.assertEqual(result.japanese, "改善されたツイート！")
        self.assertEqual(result.indonesian, "Sudah diperbaiki!")

    def test_llm_error_returns_original(self):
        from agents.reviser import revise
        draft = _make_draft(jp="元のツイート変わらず")
        critic = _make_critic()

        with mock.patch("agents.reviser.load_config", return_value=_CFG), \
             mock.patch("agents.reviser.chat", side_effect=Exception("timeout")):
            result = revise(draft, critic)

        self.assertEqual(result.japanese, "元のツイート変わらず")

    def test_prompt_contains_issues(self):
        from agents.reviser import revise
        draft = _make_draft()
        critic = _make_critic(issues=["Hook sangat lemah", "Terlalu panjang"])
        captured = []

        def fake_chat(messages, **kwargs):
            captured.extend(messages)
            return _llm_response()

        with mock.patch("agents.reviser.load_config", return_value=_CFG), \
             mock.patch("agents.reviser.chat", side_effect=fake_chat):
            revise(draft, critic)

        all_content = " ".join(m.get("content", "") for m in captured)
        self.assertIn("Hook sangat lemah", all_content)
        self.assertIn("Terlalu panjang", all_content)

    def test_prompt_contains_suggestions(self):
        from agents.reviser import revise
        draft = _make_draft()
        critic = _make_critic(suggestions=["Mulai dengan angka", "Tambahkan emoji satu"])
        captured = []

        def fake_chat(messages, **kwargs):
            captured.extend(messages)
            return _llm_response()

        with mock.patch("agents.reviser.load_config", return_value=_CFG), \
             mock.patch("agents.reviser.chat", side_effect=fake_chat):
            revise(draft, critic)

        all_content = " ".join(m.get("content", "") for m in captured)
        self.assertIn("Mulai dengan angka", all_content)

    def test_prompt_contains_original_tweet(self):
        from agents.reviser import revise
        draft = _make_draft(jp="これがオリジナル")
        critic = _make_critic()
        captured = []

        def fake_chat(messages, **kwargs):
            captured.extend(messages)
            return _llm_response()

        with mock.patch("agents.reviser.load_config", return_value=_CFG), \
             mock.patch("agents.reviser.chat", side_effect=fake_chat):
            revise(draft, critic)

        all_content = " ".join(m.get("content", "") for m in captured)
        self.assertIn("これがオリジナル", all_content)

    def test_revised_jp_within_140(self):
        from agents.reviser import revise, JP_MAX
        draft = _make_draft()
        critic = _make_critic()

        with mock.patch("agents.reviser.load_config", return_value=_CFG), \
             mock.patch("agents.reviser.chat", return_value=_llm_response(jp="い" * 150)):
            result = revise(draft, critic)

        self.assertLessEqual(len(result.japanese), JP_MAX)


class TestBuildPrompt(unittest.TestCase):
    def test_prompt_contains_score(self):
        from agents.reviser import _build_prompt
        draft = _make_draft()
        critic = _make_critic(score=72)
        with mock.patch("agents.reviser.load_config", return_value=_CFG):
            prompt = _build_prompt(draft, critic, {}, "casual")
        self.assertIn("72", prompt)

    def test_prompt_contains_topic(self):
        from agents.reviser import _build_prompt
        draft = _make_draft()
        critic = _make_critic()
        with mock.patch("agents.reviser.load_config", return_value=_CFG):
            prompt = _build_prompt(draft, critic, {}, "casual")
        self.assertIn("AIニュース", prompt)

    def test_formal_register_uses_keigo(self):
        from agents.reviser import _build_prompt
        draft = _make_draft()
        critic = _make_critic()
        with mock.patch("agents.reviser.load_config", return_value=_CFG):
            prompt = _build_prompt(draft, critic, {}, "formal")
        self.assertIn("keigo", prompt)

    def test_weak_aspects_highlighted(self):
        """Aspek dengan skor < 70% dari maks harus muncul sebagai 'lemah'."""
        from agents.reviser import _build_prompt
        draft = _make_draft()
        # Hook sangat rendah (5/25 = 20%)
        critic = CriticResult(
            score=50,
            breakdown={"hook": 5, "engagement": 10, "naturalness": 10,
                       "japanese_quality": 8, "relevance": 9, "format": 8},
            issues=[], suggestions=[],
        )
        with mock.patch("agents.reviser.load_config", return_value=_CFG):
            prompt = _build_prompt(draft, critic, {}, "casual")
        self.assertIn("hook", prompt)


if __name__ == "__main__":
    unittest.main()
