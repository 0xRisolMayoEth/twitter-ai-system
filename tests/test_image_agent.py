"""
tests/test_image_agent.py — Unit test untuk IMAGE AGENT (agents/image_agent.py).

LLM di-mock. Fokus: parsing 3 query, fallback, dan kontrak output.
"""
import json
import unittest
from unittest import mock

from core.models import ImageRec
from agents import image_agent as ia


def _resp(queries=None):
    return json.dumps({
        "image_description": "crowded tokyo train",
        "google_search_queries": queries or ["tokyo rush hour", "tired commuter", "満員電車 イラスト"],
        "image_style": "moody",
        "reason": "cocok dengan mood",
    })


class TestParse(unittest.TestCase):
    def test_valid_parsed(self):
        rec = ia._parse(_resp(), "train")
        self.assertEqual(len(rec.google_search_queries), 3)
        self.assertEqual(rec.image_style, "moody")
        self.assertFalse(rec.is_fallback)

    def test_queries_capped_at_three(self):
        rec = ia._parse(_resp(["a", "b", "c", "d", "e"]), "x")
        self.assertEqual(len(rec.google_search_queries), 3)

    def test_empty_queries_raises(self):
        raw = json.dumps({"image_description": "x", "google_search_queries": []})
        with self.assertRaises(ValueError):
            ia._parse(raw, "seed")

    def test_invalid_json_raises(self):
        with self.assertRaises(ValueError):
            ia._parse("bukan json", "seed")


class TestFallback(unittest.TestCase):
    def test_fallback_has_three_queries(self):
        rec = ia._fallback("salaryman lunch")
        self.assertEqual(len(rec.google_search_queries), 3)
        self.assertTrue(rec.is_fallback)


class TestRecommendImage(unittest.TestCase):
    def test_successful(self):
        with mock.patch("agents.image_agent.chat", return_value=_resp()):
            rec = ia.recommend_image("満員電車つらい", "満員電車", "tokyo train")
        self.assertEqual(len(rec.google_search_queries), 3)
        self.assertFalse(rec.is_fallback)

    def test_llm_error_returns_fallback(self):
        with mock.patch("agents.image_agent.chat", side_effect=Exception("API down")):
            rec = ia.recommend_image("満員電車つらい", "満員電車", "tokyo train")
        self.assertTrue(rec.is_fallback)
        self.assertEqual(len(rec.google_search_queries), 3)

    def test_prompt_contains_inputs(self):
        captured = []

        def fake_chat(messages, **kw):
            captured.extend(messages)
            return _resp()

        with mock.patch("agents.image_agent.chat", side_effect=fake_chat):
            ia.recommend_image("残業なう", "残業", "office overtime")
        content = " ".join(m.get("content", "") for m in captured)
        self.assertIn("残業なう", content)
        self.assertIn("office overtime", content)


if __name__ == "__main__":
    unittest.main()
