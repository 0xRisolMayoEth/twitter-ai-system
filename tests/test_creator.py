"""
tests/test_creator.py — Unit test untuk agents/creator.py

LLM di-mock agar test cepat dan tidak butuh API key.
"""
import json
import unittest
from unittest import mock

from core.models import StrategistOutput, TweetDraft


def _make_strategy(
    topic="AI最新ニュース",
    angle_type="news_insight",
    angle_desc="Fakta utama berita AI",
    url="http://example.com",
):
    return StrategistOutput(
        topic=topic,
        source_url=url,
        angle_type=angle_type,
        angle_description=angle_desc,
        reasoning="test reasoning",
    )


def _valid_llm_response(jp="これは日本語のツイートです。", indo="Ini tweet Indonesia."):
    return json.dumps({"japanese": jp, "indonesian": indo})


class TestParse(unittest.TestCase):
    def test_valid_json_returns_draft(self):
        from agents.creator import _parse
        strategy = _make_strategy()
        raw = _valid_llm_response("日本語ツイート", "Tweet Indonesia")
        result = _parse(raw, strategy)
        self.assertEqual(result.japanese, "日本語ツイート")
        self.assertEqual(result.indonesian, "Tweet Indonesia")
        self.assertEqual(result.topic, strategy.topic)
        self.assertEqual(result.angle_type, strategy.angle_type)

    def test_json_embedded_in_text(self):
        """LLM kadang membungkus JSON dalam teks."""
        from agents.creator import _parse
        strategy = _make_strategy()
        raw = 'Berikut tweet:\n{"japanese": "日本語", "indonesian": "Indonesia"}\nSekian.'
        result = _parse(raw, strategy)
        self.assertEqual(result.japanese, "日本語")

    def test_jp_truncated_at_140_chars(self):
        from agents.creator import _parse, JP_MAX
        strategy = _make_strategy()
        long_jp = "あ" * 200
        raw = _valid_llm_response(jp=long_jp, indo="OK")
        result = _parse(raw, strategy)
        self.assertLessEqual(len(result.japanese), JP_MAX)

    def test_indo_truncated_at_280_chars(self):
        from agents.creator import _parse, ID_MAX
        strategy = _make_strategy()
        long_indo = "a" * 300
        raw = _valid_llm_response(jp="日本語", indo=long_indo)
        result = _parse(raw, strategy)
        self.assertLessEqual(len(result.indonesian), ID_MAX)

    def test_empty_indonesian_falls_back_to_japanese(self):
        from agents.creator import _parse
        strategy = _make_strategy()
        raw = json.dumps({"japanese": "日本語のみ", "indonesian": ""})
        result = _parse(raw, strategy)
        self.assertEqual(result.indonesian, "日本語のみ")

    def test_missing_japanese_raises_to_fallback(self):
        from agents.creator import _parse, _fallback
        strategy = _make_strategy()
        raw = json.dumps({"japanese": "", "indonesian": "Indo"})
        result = _parse(raw, strategy)
        # Should return fallback (original=None here)
        self.assertIsInstance(result, TweetDraft)

    def test_invalid_json_returns_fallback(self):
        from agents.creator import _parse
        strategy = _make_strategy()
        result = _parse("ini bukan json", strategy)
        self.assertIsInstance(result, TweetDraft)

    def test_original_returned_on_parse_failure_when_provided(self):
        from agents.creator import _parse
        strategy = _make_strategy()
        original = TweetDraft(
            japanese="オリジナル", indonesian="Original",
            topic=strategy.topic, angle_type=strategy.angle_type
        )
        result = _parse("bad json", strategy, original=original)
        self.assertEqual(result.japanese, "オリジナル")


class TestFallback(unittest.TestCase):
    def test_fallback_never_none(self):
        from agents.creator import _fallback
        strategy = _make_strategy()
        result = _fallback(strategy)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, TweetDraft)

    def test_fallback_jp_within_limit(self):
        from agents.creator import _fallback, JP_MAX
        strategy = _make_strategy(topic="あ" * 200)
        result = _fallback(strategy)
        self.assertLessEqual(len(result.japanese), JP_MAX)

    def test_fallback_topic_preserved(self):
        from agents.creator import _fallback
        strategy = _make_strategy(topic="量子コンピュータ")
        result = _fallback(strategy)
        self.assertEqual(result.topic, "量子コンピュータ")
        self.assertEqual(result.angle_type, strategy.angle_type)


class TestCreateTweet(unittest.TestCase):
    """Test create_tweet() dengan LLM di-mock."""

    _CFG = {
        "persona": {
            "name": "TestBot",
            "description": "Bot untuk testing.",
        },
        "tweet_register": "casual",
        "niche": "AI tech",
    }

    def test_successful_creation(self):
        from agents.creator import create_tweet
        strategy = _make_strategy()

        with mock.patch("agents.creator.load_config", return_value=self._CFG), \
             mock.patch("agents.creator.chat", return_value=_valid_llm_response("AIのニュース！", "Berita AI!")):
            result = create_tweet(strategy)

        self.assertEqual(result.japanese, "AIのニュース！")
        self.assertEqual(result.indonesian, "Berita AI!")

    def test_retry_when_jp_too_long(self):
        """Jika JP > 140 chars, harus ada retry kedua dengan prompt lebih ketat."""
        from agents.creator import create_tweet, JP_MAX

        long_jp = "あ" * 150  # > 140
        short_jp = "短いツイート"

        call_count = [0]

        def fake_chat(messages, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _valid_llm_response(jp=long_jp, indo="Indo")
            else:
                return _valid_llm_response(jp=short_jp, indo="Indo singkat")

        with mock.patch("agents.creator.load_config", return_value=self._CFG), \
             mock.patch("agents.creator.chat", side_effect=fake_chat):
            result = create_tweet(strategy=_make_strategy())

        self.assertEqual(call_count[0], 2)  # dipanggil 2x
        self.assertEqual(result.japanese, short_jp)
        self.assertLessEqual(len(result.japanese), JP_MAX)

    def test_no_retry_when_jp_within_limit(self):
        """Jika JP ≤ 140 chars, tidak ada retry."""
        from agents.creator import create_tweet

        call_count = [0]

        def fake_chat(messages, **kwargs):
            call_count[0] += 1
            return _valid_llm_response("短い", "Singkat")

        with mock.patch("agents.creator.load_config", return_value=self._CFG), \
             mock.patch("agents.creator.chat", side_effect=fake_chat):
            result = create_tweet(_make_strategy())

        self.assertEqual(call_count[0], 1)

    def test_llm_error_returns_fallback(self):
        from agents.creator import create_tweet
        strategy = _make_strategy(topic="フォールバックテスト")

        with mock.patch("agents.creator.load_config", return_value=self._CFG), \
             mock.patch("agents.creator.chat", side_effect=Exception("API error")):
            result = create_tweet(strategy)

        self.assertIsInstance(result, TweetDraft)
        self.assertIsNotNone(result.japanese)

    def test_formal_register_changes_prompt(self):
        """tweet_register=formal harus menghasilkan prompt berbeda."""
        from agents.creator import create_tweet, _build_prompt

        cfg_formal = {**self._CFG, "tweet_register": "formal"}
        strategy = _make_strategy()

        with mock.patch("agents.creator.load_config", return_value=cfg_formal):
            prompt_formal = _build_prompt(strategy, cfg_formal["persona"], "formal")

        with mock.patch("agents.creator.load_config", return_value=self._CFG):
            prompt_casual = _build_prompt(strategy, self._CFG["persona"], "casual")

        self.assertNotEqual(prompt_formal, prompt_casual)
        self.assertIn("keigo", prompt_formal)
        self.assertIn("futsū", prompt_casual)

    def test_source_url_in_draft(self):
        from agents.creator import create_tweet
        strategy = _make_strategy(url="http://source.com/article")

        with mock.patch("agents.creator.load_config", return_value=self._CFG), \
             mock.patch("agents.creator.chat", return_value=_valid_llm_response("ツイート", "Tweet")):
            result = create_tweet(strategy)

        self.assertEqual(result.source_url, "http://source.com/article")


class TestBuildPrompt(unittest.TestCase):
    _CFG = {"niche": "AI & tech", "persona": {"name": "Bot", "description": "Desc"}}

    def test_prompt_contains_topic(self):
        from agents.creator import _build_prompt
        strategy = _make_strategy(topic="量子コンピュータ")
        with mock.patch("agents.creator.load_config", return_value=self._CFG):
            prompt = _build_prompt(strategy, {}, "casual")
        self.assertIn("量子コンピュータ", prompt)

    def test_prompt_contains_angle(self):
        from agents.creator import _build_prompt
        strategy = _make_strategy(angle_type="contrarian", angle_desc="Opini berlawanan arus")
        with mock.patch("agents.creator.load_config", return_value=self._CFG):
            prompt = _build_prompt(strategy, {}, "casual")
        self.assertIn("contrarian", prompt)
        self.assertIn("Opini berlawanan arus", prompt)

    def test_prompt_contains_char_limit(self):
        from agents.creator import _build_prompt, JP_MAX
        strategy = _make_strategy()
        with mock.patch("agents.creator.load_config", return_value=self._CFG):
            prompt = _build_prompt(strategy, {}, "casual")
        self.assertIn(str(JP_MAX), prompt)

    def test_prompt_contains_persona_name(self):
        from agents.creator import _build_prompt
        strategy = _make_strategy()
        persona = {"name": "テストペルソナ", "description": "A test persona"}
        with mock.patch("agents.creator.load_config", return_value=self._CFG):
            prompt = _build_prompt(strategy, persona, "casual")
        self.assertIn("テストペルソナ", prompt)


class TestBuildRetryPrompt(unittest.TestCase):
    _CFG = {"niche": "", "persona": {}, "tweet_register": "casual"}

    def test_retry_prompt_contains_too_long_warning(self):
        from agents.creator import _build_retry_prompt
        strategy = _make_strategy()
        too_long = "あ" * 150
        with mock.patch("agents.creator.load_config", return_value=self._CFG):
            prompt = _build_retry_prompt(strategy, {}, "casual", too_long)
        self.assertIn("TERLALU PANJANG", prompt)
        self.assertIn(str(len(too_long)), prompt)

    def test_retry_prompt_includes_previous_version(self):
        from agents.creator import _build_retry_prompt
        strategy = _make_strategy()
        too_long = "あ" * 150
        with mock.patch("agents.creator.load_config", return_value=self._CFG):
            prompt = _build_retry_prompt(strategy, {}, "casual", too_long)
        self.assertIn(too_long, prompt)


if __name__ == "__main__":
    unittest.main()
