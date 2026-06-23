"""
tests/test_strategist.py — Unit test untuk agents/strategist.py

LLM dan DB di-mock agar test cepat dan deterministik.
"""
import json
import unittest
from unittest import mock

from core.models import TrendCandidate, StrategistOutput
from datetime import datetime, timezone


def _make_trend(topic="AI技術", source="test", url="http://example.com"):
    return TrendCandidate(
        topic=topic,
        source=source,
        url=url,
        freshness=datetime.now(timezone.utc),
        raw_summary="Summary singkat tentang topik ini.",
        category="ja",
    )


class TestDetectOverused(unittest.TestCase):
    def test_empty_history(self):
        from agents.strategist import _detect_overused
        self.assertEqual(_detect_overused([]), [])

    def test_single_occurrences_not_overused(self):
        from agents.strategist import _detect_overused
        recent = ["curiosity_gap", "contrarian", "relatable"]
        self.assertEqual(_detect_overused(recent), [])

    def test_double_occurrence_flagged(self):
        from agents.strategist import _detect_overused
        recent = ["news_insight", "news_insight", "curiosity_gap"]
        result = _detect_overused(recent)
        self.assertIn("news_insight", result)
        self.assertNotIn("curiosity_gap", result)

    def test_triple_occurrence_flagged_once(self):
        from agents.strategist import _detect_overused
        recent = ["listicle", "listicle", "listicle"]
        result = _detect_overused(recent)
        self.assertEqual(result.count("listicle"), 1)


class TestParse(unittest.TestCase):
    def test_valid_json_parsed(self):
        from agents.strategist import _parse, ANGLE_TYPES
        trend = _make_trend()
        raw = json.dumps({
            "angle_type": "curiosity_gap",
            "angle_description": "Bikin penasaran tentang AI terbaru",
            "reasoning": "Topik teknis cocok dengan hook misteri",
        })
        result = _parse(raw, trend)
        self.assertEqual(result.angle_type, "curiosity_gap")
        self.assertEqual(result.angle_description, "Bikin penasaran tentang AI terbaru")
        self.assertEqual(result.topic, trend.topic)
        self.assertEqual(result.source_url, trend.url)

    def test_unknown_angle_type_defaults_to_news_insight(self):
        from agents.strategist import _parse
        trend = _make_trend()
        raw = json.dumps({
            "angle_type": "tidak_ada_angle_ini",
            "angle_description": "Deskripsi",
            "reasoning": "alasan",
        })
        result = _parse(raw, trend)
        self.assertEqual(result.angle_type, "news_insight")

    def test_json_embedded_in_text(self):
        """LLM kadang membungkus JSON dalam teks biasa."""
        from agents.strategist import _parse
        trend = _make_trend()
        raw = 'Berikut hasilnya:\n{"angle_type": "contrarian", "angle_description": "Opini berlawanan", "reasoning": "r"}\nSemoga membantu.'
        result = _parse(raw, trend)
        self.assertEqual(result.angle_type, "contrarian")

    def test_invalid_json_falls_back(self):
        from agents.strategist import _parse
        trend = _make_trend()
        result = _parse("ini bukan json", trend)
        # Fallback returns a valid StrategistOutput
        self.assertIsInstance(result, StrategistOutput)
        self.assertIn(result.angle_type, ["curiosity_gap", "contrarian", "relatable", "news_insight", "listicle"])

    def test_missing_fields_use_defaults(self):
        from agents.strategist import _parse, ANGLE_TYPES
        trend = _make_trend()
        raw = json.dumps({"angle_type": "relatable"})
        result = _parse(raw, trend)
        self.assertEqual(result.angle_type, "relatable")
        self.assertEqual(result.angle_description, ANGLE_TYPES["relatable"])


class TestFallback(unittest.TestCase):
    def test_fallback_picks_least_used(self):
        from agents.strategist import _fallback, ANGLE_TYPES
        trend = _make_trend()
        # Semua angle sudah dipakai kecuali "listicle"
        recent = ["curiosity_gap", "contrarian", "relatable", "news_insight",
                  "curiosity_gap", "contrarian", "relatable", "news_insight"]
        result = _fallback(trend, recent)
        self.assertEqual(result.angle_type, "listicle")
        self.assertEqual(result.topic, trend.topic)

    def test_fallback_no_history(self):
        from agents.strategist import _fallback, ANGLE_TYPES
        trend = _make_trend()
        result = _fallback(trend, [])
        self.assertIn(result.angle_type, ANGLE_TYPES)

    def test_fallback_source_url_from_trend(self):
        from agents.strategist import _fallback
        trend = _make_trend(url="http://specific-url.com")
        result = _fallback(trend, [])
        self.assertEqual(result.source_url, "http://specific-url.com")


class TestPickAngle(unittest.TestCase):
    """Test pick_angle() dengan LLM dan DB di-mock."""

    def _mock_llm_response(self, angle_type="news_insight"):
        return json.dumps({
            "angle_type": angle_type,
            "angle_description": f"Deskripsi untuk {angle_type}",
            "reasoning": "alasan pemilihan",
        })

    def test_successful_llm_call(self):
        from agents.strategist import pick_angle
        trend = _make_trend()

        with mock.patch("agents.strategist.chat", return_value=self._mock_llm_response("curiosity_gap")), \
             mock.patch("agents.strategist.get_recent_angle_types", return_value=[]):
            result = pick_angle(trend)

        self.assertEqual(result.angle_type, "curiosity_gap")
        self.assertEqual(result.topic, trend.topic)

    def test_llm_error_falls_back(self):
        from agents.strategist import pick_angle, ANGLE_TYPES
        trend = _make_trend()

        with mock.patch("agents.strategist.chat", side_effect=Exception("LLM down")), \
             mock.patch("agents.strategist.get_recent_angle_types", return_value=[]):
            result = pick_angle(trend)

        self.assertIsInstance(result, StrategistOutput)
        self.assertIn(result.angle_type, ANGLE_TYPES)

    def test_overused_angles_passed_in_prompt(self):
        """Verifikasi bahwa overused angles disertakan dalam prompt ke LLM."""
        from agents.strategist import pick_angle

        trend = _make_trend()
        captured_messages = []

        def fake_chat(messages, **kwargs):
            captured_messages.extend(messages)
            return self._mock_llm_response("listicle")

        with mock.patch("agents.strategist.chat", side_effect=fake_chat), \
             mock.patch("agents.strategist.get_recent_angle_types",
                        return_value=["news_insight", "news_insight", "contrarian", "contrarian"]):
            pick_angle(trend)

        # Cek bahwa prompt memuat "overused" mention
        user_content = " ".join(m.get("content", "") for m in captured_messages)
        self.assertIn("news_insight", user_content)
        self.assertIn("contrarian", user_content)

    def test_result_has_all_required_fields(self):
        from agents.strategist import pick_angle
        trend = _make_trend(topic="量子コンピュータ", url="http://q.com")

        with mock.patch("agents.strategist.chat", return_value=self._mock_llm_response("contrarian")), \
             mock.patch("agents.strategist.get_recent_angle_types", return_value=["news_insight"]):
            result = pick_angle(trend)

        self.assertEqual(result.topic, "量子コンピュータ")
        self.assertEqual(result.source_url, "http://q.com")
        self.assertIsNotNone(result.angle_description)
        self.assertIsNotNone(result.reasoning)

    def test_pick_angle_with_all_angles_overused(self):
        """Jika semua angle overused, fallback tetap berjalan."""
        from agents.strategist import pick_angle, ANGLE_TYPES
        trend = _make_trend()
        all_overused = list(ANGLE_TYPES.keys()) * 3

        with mock.patch("agents.strategist.chat", side_effect=Exception("timeout")), \
             mock.patch("agents.strategist.get_recent_angle_types", return_value=all_overused):
            result = pick_angle(trend)

        self.assertIn(result.angle_type, ANGLE_TYPES)


class TestBuildPrompt(unittest.TestCase):
    def test_prompt_contains_topic(self):
        from agents.strategist import _build_prompt
        trend = _make_trend(topic="ChatGPT新機能")
        with mock.patch("agents.strategist.load_config", return_value={"niche": "AI tech"}):
            prompt = _build_prompt(trend, "AI tech", [], [])
        self.assertIn("ChatGPT新機能", prompt)

    def test_prompt_contains_overused_warning(self):
        from agents.strategist import _build_prompt
        trend = _make_trend()
        with mock.patch("agents.strategist.load_config", return_value={"niche": ""}):
            prompt = _build_prompt(trend, "", ["news_insight", "relatable"], ["news_insight"])
        self.assertIn("news_insight", prompt)

    def test_prompt_lists_all_angle_types(self):
        from agents.strategist import _build_prompt, ANGLE_TYPES
        trend = _make_trend()
        with mock.patch("agents.strategist.load_config", return_value={"niche": ""}):
            prompt = _build_prompt(trend, "", [], [])
        for angle in ANGLE_TYPES:
            self.assertIn(angle, prompt)


if __name__ == "__main__":
    unittest.main()
