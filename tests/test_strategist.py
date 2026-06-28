"""
tests/test_strategist.py — Unit test untuk TREND AGENT (agents/strategist.py).

LLM di-mock. Fokus: pemilihan topik salaryman-relatable, filter terlarang,
parsing JSON, dan fallback.
"""
import json
import unittest
from unittest import mock

from core.models import TrendCandidate, StrategistOutput
from agents import strategist as st


def _cands(*topics):
    return [TrendCandidate(topic=t, source="test", url=f"http://x/{i}")
            for i, t in enumerate(topics)]


class TestBannedFilter(unittest.TestCase):
    def test_politics_flagged(self):
        self.assertTrue(st._looks_banned("選挙の結果が話題"))

    def test_disaster_flagged(self):
        self.assertTrue(st._looks_banned("地震 速報"))

    def test_normal_not_flagged(self):
        self.assertFalse(st._looks_banned("満員電車がつらい"))


class TestParse(unittest.TestCase):
    def test_valid_picks_parsed(self):
        raw = json.dumps({"picks": [
            {"trending_topic": "満員電車", "topic_category": "commute",
             "why_relatable": "毎朝つらい", "search_query_for_image": "tokyo train"},
        ]})
        out = st._parse(raw, _cands("満員電車"))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].topic_category, "commute")
        self.assertEqual(out[0].search_query_for_image, "tokyo train")
        self.assertEqual(out[0].source_url, "http://x/0")
        self.assertEqual(out[0].angle_type, "commute")  # disimpan ke DB

    def test_banned_pick_dropped(self):
        raw = json.dumps({"picks": [
            {"trending_topic": "選挙速報", "topic_category": "x", "why_relatable": "", "search_query_for_image": ""},
            {"trending_topic": "昼休みのラーメン", "topic_category": "lunch", "why_relatable": "", "search_query_for_image": ""},
        ]})
        out = st._parse(raw, _cands("選挙速報", "昼休みのラーメン"))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].topic, "昼休みのラーメン")

    def test_capped_at_three(self):
        picks = [{"trending_topic": f"残業{i}日目", "topic_category": "overtime",
                  "why_relatable": "", "search_query_for_image": ""} for i in range(5)]
        raw = json.dumps({"picks": picks})
        out = st._parse(raw, _cands(*[f"残業{i}日目" for i in range(5)]))
        self.assertLessEqual(len(out), 3)

    def test_invalid_json_raises(self):
        with self.assertRaises(ValueError):
            st._parse("bukan json", _cands("x"))


class TestSelectTrends(unittest.TestCase):
    def test_empty_candidates_returns_empty(self):
        self.assertEqual(st.select_trends([]), [])

    def test_all_banned_returns_empty(self):
        with mock.patch("agents.strategist.chat") as mc:
            out = st.select_trends(_cands("選挙速報", "大地震の被害"))
        self.assertEqual(out, [])
        mc.assert_not_called()  # tidak perlu panggil LLM

    def test_successful_selection(self):
        raw = json.dumps({"picks": [
            {"trending_topic": "満員電車", "topic_category": "commute",
             "why_relatable": "毎朝", "search_query_for_image": "train"},
        ]})
        with mock.patch("agents.strategist.chat", return_value=raw):
            out = st.select_trends(_cands("満員電車", "給料日"))
        self.assertEqual(out[0].topic, "満員電車")

    def test_llm_error_falls_back(self):
        with mock.patch("agents.strategist.chat", side_effect=Exception("API down")):
            out = st.select_trends(_cands("満員電車", "残業"))
        self.assertGreaterEqual(len(out), 1)
        self.assertIn("fallback", out[0].reasoning)


class TestBuildPrompt(unittest.TestCase):
    def test_prompt_lists_candidates_and_rules(self):
        prompt = st._build_prompt(_cands("満員電車", "給料日"))
        self.assertIn("満員電車", prompt)
        self.assertIn("salaryman", prompt.lower())
        self.assertIn("HINDARI", prompt)


if __name__ == "__main__":
    unittest.main()
