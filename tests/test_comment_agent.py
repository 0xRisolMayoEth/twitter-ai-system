"""
tests/test_comment_agent.py — Unit test untuk agents/comment_agent.py

LLM (teks + vision) di-mock. Fokus: parsing 3 komentar, fallback,
pipeline gambar→komentar, dan format balasan.
"""
import json
import unittest
from unittest import mock

from core.models import Comment, CommentSet
from agents import comment_agent as ca


def _resp(n=3):
    angles = ["empati", "pertanyaan", "humor"]
    return json.dumps({"comments": [
        {"japanese": f"コメント{i}", "indonesian": f"komentar {i}", "angle": angles[i % 3]}
        for i in range(n)
    ]})


class TestParse(unittest.TestCase):
    def test_three_comments(self):
        cs = ca._parse(_resp(3), "元の投稿")
        self.assertEqual(len(cs.comments), 3)
        self.assertEqual(cs.source_text, "元の投稿")
        self.assertEqual(cs.comments[0].angle, "empati")

    def test_capped_at_n(self):
        cs = ca._parse(_resp(6), "x")
        self.assertEqual(len(cs.comments), ca.N_COMMENTS)

    def test_empty_indo_falls_back_to_jp(self):
        raw = json.dumps({"comments": [{"japanese": "コメント", "indonesian": ""}]})
        cs = ca._parse(raw, "x")
        self.assertEqual(cs.comments[0].indonesian, "コメント")

    def test_skips_empty_japanese(self):
        raw = json.dumps({"comments": [
            {"japanese": "", "indonesian": "x"},
            {"japanese": "有効", "indonesian": "valid"},
        ]})
        cs = ca._parse(raw, "x")
        self.assertEqual(len(cs.comments), 1)

    def test_invalid_json_raises(self):
        with self.assertRaises(ValueError):
            ca._parse("bukan json", "x")

    def test_no_valid_comments_raises(self):
        with self.assertRaises(ValueError):
            ca._parse(json.dumps({"comments": []}), "x")


class TestGenerate(unittest.TestCase):
    def test_successful(self):
        with mock.patch("agents.comment_agent.chat", return_value=_resp()):
            cs = ca.generate_comments("満員電車つらい")
        self.assertEqual(len(cs.comments), 3)
        self.assertFalse(cs.is_fallback)

    def test_llm_error_returns_fallback(self):
        with mock.patch("agents.comment_agent.chat", side_effect=Exception("down")):
            cs = ca.generate_comments("満員電車つらい")
        self.assertTrue(cs.is_fallback)
        self.assertEqual(len(cs.comments), 3)  # fallback tetap 3 saran aman

    def test_prompt_contains_post_and_safety(self):
        captured = []

        def fake_chat(messages, **kw):
            captured.extend(messages)
            return _resp()

        with mock.patch("agents.comment_agent.chat", side_effect=fake_chat):
            ca.generate_comments("残業つらい")
        content = " ".join(m.get("content", "") for m in captured)
        self.assertIn("残業つらい", content)
        self.assertIn("DILARANG menyinggung", content)


class TestImagePipeline(unittest.TestCase):
    def test_full_pipeline(self):
        with mock.patch("agents.comment_agent.vision_chat", return_value="満員電車つらい"), \
             mock.patch("agents.comment_agent.chat", return_value=_resp()):
            cs = ca.comments_from_image(b"imgbytes")
        self.assertEqual(len(cs.comments), 3)
        self.assertEqual(cs.source_text, "満員電車つらい")

    def test_vision_failure_returns_fallback_empty(self):
        with mock.patch("agents.comment_agent.vision_chat", side_effect=Exception("no vision")):
            cs = ca.comments_from_image(b"x")
        self.assertTrue(cs.is_fallback)
        self.assertEqual(len(cs.comments), 0)

    def test_vision_empty_text_no_crash(self):
        with mock.patch("agents.comment_agent.vision_chat", return_value="   "):
            cs = ca.comments_from_image(b"x")
        self.assertTrue(cs.is_fallback)


class TestFormatReply(unittest.TestCase):
    def test_contains_copyable_jp_and_indo(self):
        cs = ca._parse(_resp(), "投稿テキスト")
        msg = ca.format_reply(cs)
        self.assertIn("3 Saran Komentar", msg)
        self.assertIn("<code>コメント0</code>", msg)
        self.assertIn("🇮🇩", msg)

    def test_empty_commentset_message(self):
        msg = ca.format_reply(CommentSet(comments=[], is_fallback=True))
        self.assertIn("tidak ada teks", msg)

    def test_html_escaped(self):
        cs = CommentSet(comments=[Comment(japanese="<b>x</b>", indonesian="<i>y</i>")])
        msg = ca.format_reply(cs)
        self.assertIn("&lt;b&gt;", msg)


if __name__ == "__main__":
    unittest.main()
