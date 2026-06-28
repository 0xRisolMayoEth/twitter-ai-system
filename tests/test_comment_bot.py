"""
tests/test_comment_bot.py — Unit test untuk handler bot Comment Agent
(tg_bot/bot.py). HTTP & agent di-mock.
"""
import unittest
from unittest import mock

from tg_bot import bot


class TestPhotoHelpers(unittest.TestCase):
    def test_largest_photo_id_picks_last(self):
        msg = {"photo": [{"file_id": "small"}, {"file_id": "mid"}, {"file_id": "big"}]}
        self.assertEqual(bot._largest_photo_id(msg), "big")

    def test_no_photo_returns_empty(self):
        self.assertEqual(bot._largest_photo_id({"text": "hi"}), "")


class TestHandleUpdate(unittest.TestCase):
    def test_photo_triggers_process(self):
        upd = {"message": {"chat": {"id": 555},
                           "photo": [{"file_id": "a"}, {"file_id": "b"}]}}
        with mock.patch("tg_bot.bot.process_photo") as mp:
            bot.handle_update(upd)
        mp.assert_called_once_with("b", 555)

    def test_text_sends_help(self):
        upd = {"message": {"chat": {"id": 7}, "text": "halo"}}
        with mock.patch("tg_bot.bot._send") as ms, \
             mock.patch("tg_bot.bot.process_photo") as mp:
            bot.handle_update(upd)
        mp.assert_not_called()
        ms.assert_called_once()
        self.assertEqual(ms.call_args[0][0], 7)  # balas ke chat pengirim

    def test_no_chat_id_ignored(self):
        with mock.patch("tg_bot.bot._send") as ms:
            bot.handle_update({"message": {}})
        ms.assert_not_called()


class TestProcessPhoto(unittest.TestCase):
    def test_download_fail_sends_warning(self):
        with mock.patch("tg_bot.bot._download_file", return_value=b""), \
             mock.patch("tg_bot.bot._send") as ms:
            bot.process_photo("fid", 9)
        # hanya pesan peringatan (1 kali), tidak proses lebih lanjut
        ms.assert_called_once()
        self.assertIn("Gagal", ms.call_args[0][1])

    def test_success_flow_sends_reply(self):
        from core.models import CommentSet, Comment
        cs = CommentSet(comments=[Comment(japanese="x", indonesian="y")], source_text="t")
        with mock.patch("tg_bot.bot._download_file", return_value=b"imgbytes"), \
             mock.patch("agents.comment_agent.comments_from_image", return_value=cs), \
             mock.patch("tg_bot.bot._send", return_value=True) as ms:
            bot.process_photo("fid", 42)
        # 1) pesan "lagi baca", 2) balasan komentar
        self.assertEqual(ms.call_count, 2)
        self.assertEqual(ms.call_args_list[-1][0][0], 42)


if __name__ == "__main__":
    unittest.main()
