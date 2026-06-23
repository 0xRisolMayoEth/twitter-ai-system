"""
tests/test_dedup.py — Unit test untuk database/dedup.py

LLM & sentence-transformers di-mock, jadi test berjalan tanpa model nyata.
Yang diuji: logika cosine similarity, threshold, dan flow check_and_save.
"""
import os
import tempfile
import unittest
from unittest import mock

import numpy as np


def _make_vec(*values) -> np.ndarray:
    """Buat numpy float32 array ternormalisasi untuk testing."""
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestCosineSimilarity(unittest.TestCase):

    def test_identical_vectors(self):
        """Vektor identik → similarity = 1.0"""
        from database.dedup import cosine_similarity
        a = _make_vec(1.0, 0.0, 0.0)
        self.assertAlmostEqual(cosine_similarity(a, a), 1.0, places=5)

    def test_orthogonal_vectors(self):
        """Vektor tegak lurus → similarity = 0.0"""
        from database.dedup import cosine_similarity
        a = _make_vec(1.0, 0.0)
        b = _make_vec(0.0, 1.0)
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0, places=5)

    def test_similar_vectors(self):
        """Vektor hampir sama → similarity tinggi"""
        from database.dedup import cosine_similarity
        a = _make_vec(1.0, 0.1)
        b = _make_vec(1.0, 0.15)
        self.assertGreater(cosine_similarity(a, b), 0.9)

    def test_bytes_roundtrip(self):
        """Konversi ndarray → bytes → ndarray harus presisi."""
        from database.dedup import bytes_to_vector
        original = _make_vec(0.3, 0.7, 0.5)
        roundtrip = bytes_to_vector(original.tobytes())
        self.assertTrue(np.allclose(original, roundtrip))


class TestIsDuplicate(unittest.TestCase):
    """Test is_duplicate() dengan mock model dan DB."""

    def _make_mock_model(self, return_vec: np.ndarray):
        m = mock.MagicMock()
        m.encode.return_value = return_vec
        return m

    def test_no_history_not_duplicate(self):
        """Tanpa history, tidak ada duplikat."""
        vec = _make_vec(1.0, 0.0)
        with mock.patch("database.dedup._get_model", return_value=self._make_mock_model(vec)), \
             mock.patch("database.dedup.compute_embedding", return_value=vec.tobytes()), \
             mock.patch("database.db_manager.get_recent_embeddings", return_value=[]):
            from database.dedup import is_duplicate
            self.assertFalse(is_duplicate("test text"))

    def test_identical_content_is_duplicate(self):
        """Konten identik dengan history → duplikat."""
        vec = _make_vec(1.0, 0.0)
        existing = (42, vec.tobytes())
        with mock.patch("database.dedup._get_model", return_value=self._make_mock_model(vec)), \
             mock.patch("database.dedup.compute_embedding", return_value=vec.tobytes()), \
             mock.patch("database.db_manager.get_recent_embeddings", return_value=[existing]):
            from database.dedup import is_duplicate
            self.assertTrue(is_duplicate("same text", threshold=0.85))

    def test_different_content_not_duplicate(self):
        """Konten berbeda → bukan duplikat."""
        new_vec = _make_vec(1.0, 0.0)
        old_vec = _make_vec(0.0, 1.0)  # orthogonal
        existing = (42, old_vec.tobytes())
        with mock.patch("database.dedup._get_model", return_value=self._make_mock_model(new_vec)), \
             mock.patch("database.dedup.compute_embedding", return_value=new_vec.tobytes()), \
             mock.patch("database.db_manager.get_recent_embeddings", return_value=[existing]):
            from database.dedup import is_duplicate
            self.assertFalse(is_duplicate("different text", threshold=0.85))

    def test_below_threshold_not_duplicate(self):
        """Similarity di bawah threshold → bukan duplikat."""
        new_vec = _make_vec(1.0, 0.5)
        old_vec = _make_vec(0.0, 1.0)
        # cosine sim antara ini < 0.85
        existing = (42, old_vec.tobytes())
        with mock.patch("database.dedup._get_model", return_value=self._make_mock_model(new_vec)), \
             mock.patch("database.dedup.compute_embedding", return_value=new_vec.tobytes()), \
             mock.patch("database.db_manager.get_recent_embeddings", return_value=[existing]):
            from database.dedup import is_duplicate
            self.assertFalse(is_duplicate("somewhat different", threshold=0.85))

    def test_model_unavailable_returns_false(self):
        """Jika model tidak tersedia, dedup dinonaktifkan (bukan false positive)."""
        with mock.patch("database.dedup._get_model", return_value=None):
            from database.dedup import is_duplicate
            self.assertFalse(is_duplicate("any text"))


class TestCheckAndSave(unittest.TestCase):
    """Test check_and_save() — cek + simpan embedding."""

    def setUp(self):
        """Pakai DB sementara."""
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        import database.db_manager as dbm
        self._orig_path = dbm.DB_PATH
        dbm.DB_PATH = self.tmp.name
        dbm.init_db()

    def tearDown(self):
        import database.db_manager as dbm
        dbm.DB_PATH = self._orig_path
        os.unlink(self.tmp.name)

    def test_new_content_passes_and_saves_embedding(self):
        """Konten baru lolos dedup dan embedding tersimpan ke DB."""
        from database.db_manager import save_tweet, get_recent_embeddings
        tweet_id = save_tweet("新しいツイート", "topic")

        new_vec = _make_vec(1.0, 0.0, 0.0)
        mock_model = mock.MagicMock()
        mock_model.encode.return_value = new_vec

        with mock.patch("database.dedup._get_model", return_value=mock_model), \
             mock.patch("database.dedup.compute_embedding", return_value=new_vec.tobytes()):
            from database.dedup import check_and_save
            result = check_and_save(tweet_id, "新しいツイート", lookback_days=30, threshold=0.85)

        self.assertTrue(result)
        # Embedding harus tersimpan di DB
        embeddings = get_recent_embeddings(days=30)
        self.assertEqual(len(embeddings), 1)

    def test_duplicate_content_rejected_no_embedding_saved(self):
        """Konten duplikat ditolak dan embedding TIDAK tersimpan."""
        from database.db_manager import save_tweet, save_embedding, get_recent_embeddings

        # Simpan konten pertama dan embedding-nya
        old_id = save_tweet("既存のツイート", "topic")
        same_vec = _make_vec(1.0, 0.0, 0.0)
        save_embedding(old_id, same_vec.tobytes())

        # Coba simpan konten kedua yang identik
        new_id = save_tweet("同じコンテンツ", "topic")
        mock_model = mock.MagicMock()
        mock_model.encode.return_value = same_vec

        with mock.patch("database.dedup._get_model", return_value=mock_model), \
             mock.patch("database.dedup.compute_embedding", return_value=same_vec.tobytes()):
            from database.dedup import check_and_save
            result = check_and_save(new_id, "同じコンテンツ", lookback_days=30, threshold=0.85)

        self.assertFalse(result)
        # Hanya 1 embedding (milik old_id), bukan 2
        embeddings = get_recent_embeddings(days=30)
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(embeddings[0][0], old_id)


if __name__ == "__main__":
    unittest.main()
