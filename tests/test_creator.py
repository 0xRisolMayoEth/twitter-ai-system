"""
tests/test_creator.py — Unit test untuk WRITER AGENT (agents/creator.py).

LLM di-mock agar test cepat dan tidak butuh API key.
"""
import json
import unittest
from unittest import mock

from core.models import StrategistOutput, TweetDraft
from agents import creator as cr


def _sel(topic="満員電車", category="commute"):
    return StrategistOutput(
        topic=topic, source_url="http://x", topic_category=category,
        why_relatable="毎朝つらい", search_query_for_image="tokyo train",
        angle_type=category,
    )


def _resp(jp="満員電車つらい😮‍💨 #サラリーマン", indo="kereta penuh",
          tone="capek tapi lucu", bpt="通勤中"):
    return json.dumps({"tweet_text": jp, "indonesian": indo,
                       "tone": tone, "best_posting_time": bpt})


class TestParse(unittest.TestCase):
    def test_valid_parsed(self):
        d = cr._parse(_resp(), _sel())
        self.assertEqual(d.tone, "capek tapi lucu")
        self.assertEqual(d.best_posting_time, "通勤中")
        self.assertIn("#サラリーマン", d.japanese)
        self.assertFalse(d.is_fallback)

    def test_accepts_japanese_key_alias(self):
        raw = json.dumps({"japanese": "残業なう😴 #あるある", "indonesian": "lembur"})
        d = cr._parse(raw, _sel())
        self.assertIn("#あるある", d.japanese)

    def test_empty_indo_falls_back_to_jp(self):
        raw = json.dumps({"tweet_text": "給料日まだ？💸 #サラリーマン", "indonesian": ""})
        d = cr._parse(raw, _sel())
        self.assertEqual(d.indonesian, d.japanese)

    def test_truncate_caps_jp_when_enabled(self):
        long_jp = "あ" * 200
        raw = json.dumps({"tweet_text": long_jp, "indonesian": "x"})
        d = cr._parse(raw, _sel(), truncate=True)
        self.assertLessEqual(len(d.japanese), cr.JP_MAX)

    def test_no_truncate_preserves_overlong(self):
        long_jp = "あ" * 200
        raw = json.dumps({"tweet_text": long_jp, "indonesian": "x"})
        d = cr._parse(raw, _sel(), truncate=False)
        self.assertGreater(len(d.japanese), cr.JP_MAX)

    def test_invalid_json_falls_back(self):
        d = cr._parse("bukan json", _sel())
        self.assertTrue(d.is_fallback)


class TestFallback(unittest.TestCase):
    def test_fallback_marked(self):
        d = cr._fallback(_sel())
        self.assertTrue(d.is_fallback)
        self.assertIn("#サラリーマン", d.japanese)

    def test_fallback_within_limit(self):
        d = cr._fallback(_sel(topic="あ" * 200))
        self.assertLessEqual(len(d.japanese), cr.JP_MAX)


class TestCreateTweet(unittest.TestCase):
    def test_successful_create(self):
        with mock.patch("agents.creator.load_config", return_value={"persona": {"name": "田中サトル"}}), \
             mock.patch("agents.creator.chat", return_value=_resp()):
            d = cr.create_tweet(_sel())
        self.assertFalse(d.is_fallback)
        self.assertIn("#サラリーマン", d.japanese)

    def test_retry_when_too_long(self):
        long_resp = _resp(jp="あ" * 200)
        good_resp = _resp(jp="短いツイート😮‍💨 #サラリーマン")
        calls = []

        def fake_chat(messages, **kw):
            calls.append(messages)
            return long_resp if len(calls) == 1 else good_resp

        with mock.patch("agents.creator.load_config", return_value={"persona": {}}), \
             mock.patch("agents.creator.chat", side_effect=fake_chat):
            d = cr.create_tweet(_sel())
        self.assertEqual(len(calls), 2)            # retry terjadi
        self.assertLessEqual(len(d.japanese), cr.JP_MAX)

    def test_llm_error_returns_fallback(self):
        with mock.patch("agents.creator.load_config", return_value={"persona": {}}), \
             mock.patch("agents.creator.chat", side_effect=Exception("API down")):
            d = cr.create_tweet(_sel())
        self.assertTrue(d.is_fallback)

    def test_prompt_contains_persona_and_topic(self):
        captured = []

        def fake_chat(messages, **kw):
            captured.extend(messages)
            return _resp()

        with mock.patch("agents.creator.load_config",
                        return_value={"persona": {"name": "田中サトル", "description": "salaryman"}}), \
             mock.patch("agents.creator.chat", side_effect=fake_chat):
            cr.create_tweet(_sel(topic="昼休みのラーメン"))
        content = " ".join(m.get("content", "") for m in captured)
        self.assertIn("田中サトル", content)
        self.assertIn("昼休みのラーメン", content)
        self.assertIn("#サラリーマン", content)


if __name__ == "__main__":
    unittest.main()
