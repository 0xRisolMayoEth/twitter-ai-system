"""
tests/test_db.py — Unit test untuk database/db_manager.py

Menggunakan DB sementara di memory (:memory:) agar tidak kotor DB production.
"""
import json
import os
import sqlite3
import tempfile
import unittest
from unittest import mock


class TestDbManager(unittest.TestCase):

    def setUp(self):
        """Pakai file DB sementara per test agar isolasi sempurna."""
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = self.tmp.name

        # Patch DB_PATH ke file sementara
        import database.db_manager as dbm
        self._orig_path = dbm.DB_PATH
        dbm.DB_PATH = self.db_path
        dbm.init_db()

    def tearDown(self):
        import database.db_manager as dbm
        dbm.DB_PATH = self._orig_path
        os.unlink(self.db_path)

    # --- save_tweet & get_tweet_by_id ---

    def test_save_and_get_tweet(self):
        from database.db_manager import save_tweet, get_tweet_by_id
        tid = save_tweet("テストツイート", "AI news")
        self.assertIsInstance(tid, int)
        self.assertGreater(tid, 0)

        row = get_tweet_by_id(tid)
        self.assertIsNotNone(row)
        self.assertEqual(row["content"], "テストツイート")
        self.assertEqual(row["topic"], "AI news")
        self.assertEqual(row["status"], "pending")

    def test_get_nonexistent_tweet(self):
        from database.db_manager import get_tweet_by_id
        self.assertIsNone(get_tweet_by_id(9999))

    # --- update_tweet_status ---

    def test_update_status(self):
        from database.db_manager import save_tweet, update_tweet_status, get_tweet_by_id
        tid = save_tweet("tweet", "topic")
        update_tweet_status(tid, "sent")
        self.assertEqual(get_tweet_by_id(tid)["status"], "sent")

    def test_update_status_with_tweet_id(self):
        from database.db_manager import save_tweet, update_tweet_status, get_tweet_by_id
        tid = save_tweet("tweet", "topic")
        update_tweet_status(tid, "posted", tweet_id="x_12345")
        row = get_tweet_by_id(tid)
        self.assertEqual(row["status"], "posted")
        self.assertEqual(row["tweet_id"], "x_12345")

    # --- save_topic_used & get_recent_topics ---

    def test_topic_memory(self):
        from database.db_manager import save_topic_used, get_recent_topics
        save_topic_used("AI news", angle_type="news_insight")
        save_topic_used("Anime trend", angle_type="curiosity_gap")
        topics = get_recent_topics(10)
        self.assertIn("Anime trend", topics)
        self.assertIn("AI news", topics)
        # Harus diurutkan dari yang terbaru
        self.assertEqual(topics[0], "Anime trend")

    # --- save_tweet_full ---

    def test_save_tweet_full(self):
        from database.db_manager import save_tweet_full, get_tweet_by_id
        tid = save_tweet_full(
            topic="テスト",
            content_jp="日本語ツイート",
            content_indo="Tweet Indonesia",
            draft_jp="ドラフト",
            score=85,
            score_breakdown={"hook": 22, "engagement": 18},
            angle_type="news_insight",
        )
        row = get_tweet_by_id(tid)
        self.assertIsNotNone(row)
        self.assertEqual(row["content_jp"], "日本語ツイート")
        self.assertEqual(row["content_indo"], "Tweet Indonesia")
        self.assertEqual(row["score"], 85)
        self.assertEqual(row["angle_type"], "news_insight")
        # backward compat: content = content_jp
        self.assertEqual(row["content"], "日本語ツイート")

    # --- save_topic & mark_topic_used ---

    def test_save_topic(self):
        from database.db_manager import save_topic, mark_topic_used
        topic_id = save_topic("AI robots", source="NHK", url="https://nhk.jp/1")
        self.assertIsInstance(topic_id, int)
        # Tidak ada exception saat mark_topic_used
        mark_topic_used(topic_id)

    # --- embeddings ---

    def test_save_and_get_embeddings(self):
        import numpy as np
        from database.db_manager import save_tweet, save_embedding, get_recent_embeddings
        tid = save_tweet("embedding test", "topic")
        vec = np.array([1.0, 0.0, 0.5], dtype=np.float32)
        save_embedding(tid, vec.tobytes())

        rows = get_recent_embeddings(days=30)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], tid)
        result_vec = np.frombuffer(rows[0][1], dtype=np.float32)
        self.assertTrue(np.allclose(vec, result_vec))

    def test_embeddings_cutoff(self):
        """Embedding yang lebih tua dari lookback_days tidak dikembalikan."""
        import numpy as np
        from datetime import datetime, timezone, timedelta
        from database.db_manager import save_embedding, get_recent_embeddings
        import database.db_manager as dbm

        conn = sqlite3.connect(self.db_path)
        old_date = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        conn.execute(
            "INSERT INTO tweets (content, topic, created_at) VALUES (?, ?, ?)",
            ("old tweet", "old", old_date),
        )
        tweet_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
        conn.close()

        vec = np.array([1.0, 0.0], dtype=np.float32)
        save_embedding(tweet_id, vec.tobytes())

        rows = get_recent_embeddings(days=30)
        self.assertEqual(len(rows), 0)  # harus kosong (di luar window)

    # --- runs ---

    def test_run_lifecycle(self):
        from database.db_manager import start_run, finish_run
        run_id = start_run()
        self.assertIsInstance(run_id, int)
        # Tidak ada exception
        finish_run(run_id, produced=5, errors=1)

    # --- get_recent_angle_types ---

    def test_recent_angle_types(self):
        from database.db_manager import save_topic_used, get_recent_angle_types
        save_topic_used("topic1", angle_type="news_insight")
        save_topic_used("topic2", angle_type="contrarian")
        save_topic_used("topic3", angle_type="curiosity_gap")
        angles = get_recent_angle_types(limit=10)
        self.assertEqual(angles[0], "curiosity_gap")  # terbaru pertama
        self.assertIn("news_insight", angles)


if __name__ == "__main__":
    unittest.main()
