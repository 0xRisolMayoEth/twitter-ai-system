"""
tests/test_dispatcher.py — Unit test untuk agents/dispatcher.py

Telegram HTTP dan DB di-mock agar test cepat dan tidak butuh jaringan.
"""
import unittest
from unittest import mock

from core.models import ContentPackage


def _make_package(
    jp="AIのニュース、知ってた？",
    indo="Tau nggak soal AI terbaru?",
    score=82,
    angle="news_insight",
    below=False,
    url="http://example.com/article",
    verdict="GOOD",
):
    return ContentPackage(
        topic="AI最新情報",
        source_url=url,
        angle_type=angle,
        japanese=jp,
        indonesian=indo,
        score=score,
        verdict=verdict,
        score_breakdown={"hook": 20, "engagement": 16},
        below_threshold=below,
        revision_count=1,
    )


class TestFormatMessage(unittest.TestCase):
    def test_contains_jp_tweet(self):
        from agents.dispatcher import format_message
        pkg = _make_package(jp="日本語ツイート")
        with mock.patch("agents.dispatcher.load_config", return_value={"scoring": {"threshold": 80}}):
            msg = format_message(pkg)
        self.assertIn("日本語ツイート", msg)

    def test_contains_indo_tweet(self):
        from agents.dispatcher import format_message
        pkg = _make_package(indo="Tweet Indonesia di sini")
        with mock.patch("agents.dispatcher.load_config", return_value={"scoring": {"threshold": 80}}):
            msg = format_message(pkg)
        self.assertIn("Tweet Indonesia di sini", msg)

    def test_contains_score(self):
        from agents.dispatcher import format_message
        pkg = _make_package(score=88)
        with mock.patch("agents.dispatcher.load_config", return_value={"scoring": {"threshold": 80}}):
            msg = format_message(pkg)
        self.assertIn("88", msg)

    def test_contains_angle(self):
        from agents.dispatcher import format_message
        pkg = _make_package(angle="curiosity_gap")
        with mock.patch("agents.dispatcher.load_config", return_value={"scoring": {"threshold": 80}}):
            msg = format_message(pkg)
        self.assertIn("curiosity_gap", msg)

    def test_contains_source_url(self):
        from agents.dispatcher import format_message
        pkg = _make_package(url="http://nhk.or.jp/news/123")
        with mock.patch("agents.dispatcher.load_config", return_value={"scoring": {"threshold": 80}}):
            msg = format_message(pkg)
        self.assertIn("http://nhk.or.jp/news/123", msg)

    def test_no_url_section_when_empty(self):
        from agents.dispatcher import format_message
        pkg = _make_package(url="")
        with mock.patch("agents.dispatcher.load_config", return_value={"scoring": {"threshold": 80}}):
            msg = format_message(pkg)
        self.assertNotIn("Sumber berita", msg)

    def test_approved_verdict_tag_shown(self):
        from agents.dispatcher import format_message
        pkg = _make_package(score=95, verdict="APPROVED")
        with mock.patch("agents.dispatcher.load_config", return_value={"scoring": {"threshold": 75}}):
            msg = format_message(pkg)
        self.assertIn("✅", msg)
        self.assertIn("APPROVED", msg)

    def test_good_verdict_tag_shown(self):
        from agents.dispatcher import format_message
        pkg = _make_package(score=80, verdict="GOOD")
        with mock.patch("agents.dispatcher.load_config", return_value={"scoring": {"threshold": 75}}):
            msg = format_message(pkg)
        self.assertIn("⚡", msg)
        self.assertIn("GOOD", msg)

    def test_no_below_threshold_warning_ever(self):
        """Warning 'di bawah threshold' sudah dihapus — konten buruk tak dikirim."""
        from agents.dispatcher import format_message
        pkg = _make_package(score=90, verdict="APPROVED")
        with mock.patch("agents.dispatcher.load_config", return_value={"scoring": {"threshold": 75}}):
            msg = format_message(pkg)
        self.assertNotIn("⚠️", msg)
        self.assertNotIn("di bawah threshold", msg)

    def test_html_special_chars_escaped(self):
        """Karakter HTML dalam tweet harus di-escape agar tidak merusak parse mode."""
        from agents.dispatcher import format_message
        pkg = _make_package(jp="Score: <100> & more", indo="<test>")
        with mock.patch("agents.dispatcher.load_config", return_value={"scoring": {"threshold": 80}}):
            msg = format_message(pkg)
        self.assertIn("&lt;100&gt;", msg)
        self.assertIn("&amp;", msg)
        self.assertNotIn("<100>", msg)

    def test_separator_present(self):
        from agents.dispatcher import format_message
        pkg = _make_package()
        with mock.patch("agents.dispatcher.load_config", return_value={"scoring": {"threshold": 80}}):
            msg = format_message(pkg)
        self.assertIn("━", msg)

    def test_flag_emojis_present(self):
        from agents.dispatcher import format_message
        pkg = _make_package()
        with mock.patch("agents.dispatcher.load_config", return_value={"scoring": {"threshold": 80}}):
            msg = format_message(pkg)
        self.assertIn("🇯🇵", msg)
        self.assertIn("🇮🇩", msg)


class TestSendTelegram(unittest.TestCase):
    def test_returns_false_when_no_token(self):
        from agents.dispatcher import send_telegram
        with mock.patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": "123"}):
            result = send_telegram("test msg")
        self.assertFalse(result)

    def test_returns_false_when_no_chat_id(self):
        from agents.dispatcher import send_telegram
        with mock.patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "TOKEN", "TELEGRAM_CHAT_ID": ""}):
            result = send_telegram("test msg")
        self.assertFalse(result)

    def test_returns_true_on_success(self):
        from agents.dispatcher import send_telegram
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"ok": True, "result": {}}
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "TOKEN", "TELEGRAM_CHAT_ID": "123"}), \
             mock.patch("agents.dispatcher.load_config", return_value={"telegram": {}}), \
             mock.patch("requests.post", return_value=mock_resp):
            result = send_telegram("hello")

        self.assertTrue(result)

    def test_returns_false_on_api_error(self):
        from agents.dispatcher import send_telegram
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"ok": False, "description": "Bad Request"}
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "TOKEN", "TELEGRAM_CHAT_ID": "123"}), \
             mock.patch("agents.dispatcher.load_config", return_value={"telegram": {}}), \
             mock.patch("requests.post", return_value=mock_resp):
            result = send_telegram("hello")

        self.assertFalse(result)

    def test_retries_on_timeout(self):
        """Timeout harus diretry, bukan langsung return False."""
        import requests as req
        from agents.dispatcher import send_telegram

        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status = mock.MagicMock()

        call_count = [0]
        def fake_post(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise req.exceptions.Timeout()
            return mock_resp

        with mock.patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "T", "TELEGRAM_CHAT_ID": "C"}), \
             mock.patch("agents.dispatcher.load_config", return_value={"telegram": {}}), \
             mock.patch("requests.post", side_effect=fake_post), \
             mock.patch("time.sleep"):
            result = send_telegram("test")

        self.assertTrue(result)
        self.assertEqual(call_count[0], 2)

    def test_no_retry_on_4xx(self):
        """HTTP 4xx (kecuali 429) tidak perlu diretry."""
        import requests as req
        from agents.dispatcher import send_telegram

        http_err = req.exceptions.HTTPError(response=mock.MagicMock(
            status_code=400, headers={}
        ))

        call_count = [0]
        def fake_post(*args, **kwargs):
            call_count[0] += 1
            raise http_err

        with mock.patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "T", "TELEGRAM_CHAT_ID": "C"}), \
             mock.patch("agents.dispatcher.load_config", return_value={"telegram": {}}), \
             mock.patch("requests.post", side_effect=fake_post):
            result = send_telegram("bad")

        self.assertFalse(result)
        self.assertEqual(call_count[0], 1)

    def test_rate_limit_sleep_applied(self):
        """rate_limit_seconds dari config harus menyebabkan sleep."""
        from agents.dispatcher import send_telegram
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status = mock.MagicMock()

        sleep_calls = []
        with mock.patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "T", "TELEGRAM_CHAT_ID": "C"}), \
             mock.patch("agents.dispatcher.load_config", return_value={"telegram": {"rate_limit_seconds": 3}}), \
             mock.patch("requests.post", return_value=mock_resp), \
             mock.patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            send_telegram("test")

        self.assertIn(3, sleep_calls)


class TestDispatch(unittest.TestCase):
    """Test dispatch() — DB dan Telegram di-mock."""

    def _make_db_mocks(self, tweet_id=42):
        return {
            "save_tweet_full": mock.MagicMock(return_value=tweet_id),
            "save_topic_used": mock.MagicMock(),
            "update_tweet_sent": mock.MagicMock(),
            "check_and_save": mock.MagicMock(return_value=False),
        }

    def test_returns_tweet_id(self):
        from agents.dispatcher import dispatch
        pkg = _make_package()
        db = self._make_db_mocks(tweet_id=99)

        with mock.patch("agents.dispatcher.save_tweet_full", db["save_tweet_full"]), \
             mock.patch("agents.dispatcher.save_topic_used", db["save_topic_used"]), \
             mock.patch("agents.dispatcher.update_tweet_sent", db["update_tweet_sent"]), \
             mock.patch("agents.dispatcher.check_and_save", db["check_and_save"]), \
             mock.patch("agents.dispatcher.send_telegram", return_value=True):
            result = dispatch(pkg)

        self.assertEqual(result, 99)

    def test_update_sent_called_on_success(self):
        from agents.dispatcher import dispatch
        pkg = _make_package()
        db = self._make_db_mocks(tweet_id=5)

        with mock.patch("agents.dispatcher.save_tweet_full", db["save_tweet_full"]), \
             mock.patch("agents.dispatcher.save_topic_used", db["save_topic_used"]), \
             mock.patch("agents.dispatcher.update_tweet_sent", db["update_tweet_sent"]), \
             mock.patch("agents.dispatcher.check_and_save", db["check_and_save"]), \
             mock.patch("agents.dispatcher.send_telegram", return_value=True):
            dispatch(pkg)

        db["update_tweet_sent"].assert_called_once_with(5)

    def test_update_sent_not_called_when_telegram_fails(self):
        from agents.dispatcher import dispatch
        pkg = _make_package()
        db = self._make_db_mocks()

        with mock.patch("agents.dispatcher.save_tweet_full", db["save_tweet_full"]), \
             mock.patch("agents.dispatcher.save_topic_used", db["save_topic_used"]), \
             mock.patch("agents.dispatcher.update_tweet_sent", db["update_tweet_sent"]), \
             mock.patch("agents.dispatcher.check_and_save", db["check_and_save"]), \
             mock.patch("agents.dispatcher.send_telegram", return_value=False):
            dispatch(pkg)

        db["update_tweet_sent"].assert_not_called()

    def test_telegram_failure_does_not_raise(self):
        """Gagal kirim Telegram tidak boleh raise exception."""
        from agents.dispatcher import dispatch
        pkg = _make_package()
        db = self._make_db_mocks()

        with mock.patch("agents.dispatcher.save_tweet_full", db["save_tweet_full"]), \
             mock.patch("agents.dispatcher.save_topic_used", db["save_topic_used"]), \
             mock.patch("agents.dispatcher.update_tweet_sent", db["update_tweet_sent"]), \
             mock.patch("agents.dispatcher.check_and_save", db["check_and_save"]), \
             mock.patch("agents.dispatcher.send_telegram", return_value=False):
            tweet_id = dispatch(pkg)  # harus tidak raise

        self.assertIsNotNone(tweet_id)

    def test_save_topic_called_with_angle(self):
        from agents.dispatcher import dispatch
        pkg = _make_package(angle="contrarian")
        db = self._make_db_mocks()

        with mock.patch("agents.dispatcher.save_tweet_full", db["save_tweet_full"]), \
             mock.patch("agents.dispatcher.save_topic_used", db["save_topic_used"]), \
             mock.patch("agents.dispatcher.update_tweet_sent", db["update_tweet_sent"]), \
             mock.patch("agents.dispatcher.check_and_save", db["check_and_save"]), \
             mock.patch("agents.dispatcher.send_telegram", return_value=True):
            dispatch(pkg)

        db["save_topic_used"].assert_called_once_with("AI最新情報", angle_type="contrarian")

    def test_message_sent_to_telegram(self):
        """Format message harus diteruskan ke send_telegram."""
        from agents.dispatcher import dispatch
        pkg = _make_package(jp="テスト送信")
        db = self._make_db_mocks()

        captured = []
        def fake_send(text, **kwargs):
            captured.append(text)
            return True

        with mock.patch("agents.dispatcher.save_tweet_full", db["save_tweet_full"]), \
             mock.patch("agents.dispatcher.save_topic_used", db["save_topic_used"]), \
             mock.patch("agents.dispatcher.update_tweet_sent", db["update_tweet_sent"]), \
             mock.patch("agents.dispatcher.check_and_save", db["check_and_save"]), \
             mock.patch("agents.dispatcher.send_telegram", side_effect=fake_send), \
             mock.patch("agents.dispatcher.load_config", return_value={"scoring": {"threshold": 80}, "telegram": {}}):
            dispatch(pkg)

        self.assertEqual(len(captured), 1)
        self.assertIn("テスト送信", captured[0])


if __name__ == "__main__":
    unittest.main()
