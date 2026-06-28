"""
tests/test_dispatcher.py — Unit test untuk agents/dispatcher.py

Telegram HTTP dan DB di-mock agar test cepat dan tidak butuh jaringan.
"""
import unittest
from unittest import mock

from core.models import ContentPackage, ImageRec


def _make_package(jp="満員電車つらい😮‍💨 #サラリーマン", indo="Kereta penuh sesak",
                  score=8, angle="commute", verdict="APPROVE",
                  with_image=True):
    img = ImageRec(google_search_queries=["tokyo rush hour", "tired commuter", "満員電車 イラスト"]) if with_image else None
    return ContentPackage(
        topic="満員電車", source_url="http://example.com", angle_type=angle,
        japanese=jp, indonesian=indo, score=score, verdict=verdict,
        score_breakdown={"relatability": 9, "naturalness": 8, "engagement": 7, "topic_fit": 8},
        image=img, revision_count=0,
    )


class TestSlotLabel(unittest.TestCase):
    def test_labels(self):
        from agents.dispatcher import _slot_label
        self.assertEqual(_slot_label(6), "pagi")
        self.assertEqual(_slot_label(12), "siang")
        self.assertEqual(_slot_label(18), "sore")
        self.assertEqual(_slot_label(0), "malam")


class TestFormatMessage(unittest.TestCase):
    def test_contains_draft_tweet_copyable(self):
        from agents.dispatcher import format_message
        msg = format_message(_make_package(jp="残業なう😴 #あるある"))
        self.assertIn("DRAFT TWEET", msg)
        self.assertIn("<code>残業なう😴 #あるある</code>", msg)

    def test_contains_indonesian(self):
        from agents.dispatcher import format_message
        msg = format_message(_make_package(indo="Lembur lagi hari ini"))
        self.assertIn("Lembur lagi hari ini", msg)

    def test_score_out_of_ten(self):
        from agents.dispatcher import format_message
        msg = format_message(_make_package(score=8))
        self.assertIn("SCORE: <b>8/10</b>", msg)

    def test_image_queries_listed(self):
        from agents.dispatcher import format_message
        msg = format_message(_make_package())
        self.assertIn("REKOMENDASI GAMBAR", msg)
        self.assertIn("tokyo rush hour", msg)
        self.assertIn("満員電車 イラスト", msg)

    def test_no_image_section_when_absent(self):
        from agents.dispatcher import format_message
        msg = format_message(_make_package(with_image=False))
        self.assertNotIn("REKOMENDASI GAMBAR", msg)

    def test_approve_reject_instructions(self):
        from agents.dispatcher import format_message
        msg = format_message(_make_package())
        self.assertIn("/approve", msg)
        self.assertIn("/reject", msg)

    def test_html_escaped(self):
        from agents.dispatcher import format_message
        msg = format_message(_make_package(jp="<b>hack</b> & co", indo="<x>"))
        self.assertIn("&lt;b&gt;", msg)
        self.assertIn("&amp;", msg)


class TestSendTelegram(unittest.TestCase):
    def test_returns_false_when_no_token(self):
        from agents.dispatcher import send_telegram
        with mock.patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": "123"}):
            self.assertFalse(send_telegram("test"))

    def test_returns_true_on_success(self):
        from agents.dispatcher import send_telegram
        resp = mock.MagicMock()
        resp.json.return_value = {"ok": True}
        resp.raise_for_status = mock.MagicMock()
        with mock.patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "T", "TELEGRAM_CHAT_ID": "C"}), \
             mock.patch("agents.dispatcher.load_config", return_value={"telegram": {}}), \
             mock.patch("requests.post", return_value=resp):
            self.assertTrue(send_telegram("hello"))

    def test_returns_false_on_api_error(self):
        from agents.dispatcher import send_telegram
        resp = mock.MagicMock()
        resp.json.return_value = {"ok": False, "description": "Bad"}
        resp.raise_for_status = mock.MagicMock()
        with mock.patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "T", "TELEGRAM_CHAT_ID": "C"}), \
             mock.patch("agents.dispatcher.load_config", return_value={"telegram": {}}), \
             mock.patch("requests.post", return_value=resp):
            self.assertFalse(send_telegram("hello"))


class TestDispatch(unittest.TestCase):
    def _mocks(self, tweet_id=42):
        return {
            "save_tweet_full": mock.MagicMock(return_value=tweet_id),
            "save_topic_used": mock.MagicMock(),
            "update_tweet_sent": mock.MagicMock(),
            "check_and_save": mock.MagicMock(return_value=False),
        }

    def test_returns_tweet_id(self):
        from agents.dispatcher import dispatch
        db = self._mocks(99)
        with mock.patch("agents.dispatcher.save_tweet_full", db["save_tweet_full"]), \
             mock.patch("agents.dispatcher.save_topic_used", db["save_topic_used"]), \
             mock.patch("agents.dispatcher.update_tweet_sent", db["update_tweet_sent"]), \
             mock.patch("agents.dispatcher.check_and_save", db["check_and_save"]), \
             mock.patch("agents.dispatcher.send_telegram", return_value=True):
            self.assertEqual(dispatch(_make_package()), 99)

    def test_update_sent_on_success(self):
        from agents.dispatcher import dispatch
        db = self._mocks(5)
        with mock.patch("agents.dispatcher.save_tweet_full", db["save_tweet_full"]), \
             mock.patch("agents.dispatcher.save_topic_used", db["save_topic_used"]), \
             mock.patch("agents.dispatcher.update_tweet_sent", db["update_tweet_sent"]), \
             mock.patch("agents.dispatcher.check_and_save", db["check_and_save"]), \
             mock.patch("agents.dispatcher.send_telegram", return_value=True):
            dispatch(_make_package())
        db["update_tweet_sent"].assert_called_once_with(5)

    def test_telegram_failure_does_not_raise(self):
        from agents.dispatcher import dispatch
        db = self._mocks()
        with mock.patch("agents.dispatcher.save_tweet_full", db["save_tweet_full"]), \
             mock.patch("agents.dispatcher.save_topic_used", db["save_topic_used"]), \
             mock.patch("agents.dispatcher.update_tweet_sent", db["update_tweet_sent"]), \
             mock.patch("agents.dispatcher.check_and_save", db["check_and_save"]), \
             mock.patch("agents.dispatcher.send_telegram", return_value=False):
            self.assertIsNotNone(dispatch(_make_package()))
        db["update_tweet_sent"].assert_not_called()

    def test_message_passed_to_telegram(self):
        from agents.dispatcher import dispatch
        db = self._mocks()
        captured = []
        with mock.patch("agents.dispatcher.save_tweet_full", db["save_tweet_full"]), \
             mock.patch("agents.dispatcher.save_topic_used", db["save_topic_used"]), \
             mock.patch("agents.dispatcher.update_tweet_sent", db["update_tweet_sent"]), \
             mock.patch("agents.dispatcher.check_and_save", db["check_and_save"]), \
             mock.patch("agents.dispatcher.send_telegram", side_effect=lambda t, **k: captured.append(t) or True):
            dispatch(_make_package(jp="テスト送信 #サラリーマン"))
        self.assertIn("テスト送信", captured[0])


if __name__ == "__main__":
    unittest.main()
