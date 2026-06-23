"""
tests/test_trend_scout.py — Unit test untuk agents/trend_scout.py

HTTP dan DB di-mock agar test cepat dan tidak bergantung jaringan.
"""
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from core.models import TrendCandidate


def _make_candidate(topic="テスト", hours_old=0, source="test", lang="ja"):
    """Helper: buat TrendCandidate dengan freshness relatif terhadap sekarang."""
    return TrendCandidate(
        topic=topic,
        source=source,
        freshness=datetime.now(timezone.utc) - timedelta(hours=hours_old),
        category=lang,
    )


class TestFreshnessFilter(unittest.TestCase):
    """Test _filter_fresh() — filter berdasarkan usia topik."""

    def setUp(self):
        from agents.trend_scout import TrendScout
        self.scout = TrendScout()
        self.scout.max_age_hours = 48

    def test_fresh_topic_passes(self):
        c = _make_candidate(hours_old=1)
        self.assertEqual(len(self.scout._filter_fresh([c])), 1)

    def test_stale_topic_removed(self):
        c = _make_candidate(hours_old=50)  # lebih tua dari 48 jam
        self.assertEqual(len(self.scout._filter_fresh([c])), 0)

    def test_exactly_at_boundary_passes(self):
        c = _make_candidate(hours_old=47)  # 47 jam = masih dalam window
        self.assertEqual(len(self.scout._filter_fresh([c])), 1)

    def test_mixed_list(self):
        fresh = _make_candidate("segar", hours_old=5)
        stale = _make_candidate("basi", hours_old=72)
        result = self.scout._filter_fresh([fresh, stale])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].topic, "segar")

    def test_naive_datetime_treated_as_utc(self):
        """Datetime tanpa timezone (naive) dianggap UTC dan tidak dibuang."""
        c = TrendCandidate(
            topic="naive dt",
            source="test",
            freshness=datetime.utcnow() - timedelta(hours=10),  # naive
        )
        result = self.scout._filter_fresh([c])
        self.assertEqual(len(result), 1)


class TestSeenFilter(unittest.TestCase):
    """Test _filter_seen() — filter topik yang sudah pernah dipakai."""

    def setUp(self):
        from agents.trend_scout import TrendScout
        self.scout = TrendScout()

    def test_unseen_topic_passes(self):
        c = _make_candidate("全く新しいトピック")
        with mock.patch("agents.trend_scout.get_recent_topics", return_value=["別のトピック"]):
            result = self.scout._filter_seen([c])
        self.assertEqual(len(result), 1)

    def test_exact_match_removed(self):
        c = _make_candidate("AI最新ニュース")
        with mock.patch("agents.trend_scout.get_recent_topics", return_value=["AI最新ニュース"]):
            result = self.scout._filter_seen([c])
        self.assertEqual(len(result), 0)

    def test_case_insensitive_match(self):
        c = _make_candidate("ai news today")
        with mock.patch("agents.trend_scout.get_recent_topics", return_value=["AI News Today"]):
            result = self.scout._filter_seen([c])
        self.assertEqual(len(result), 0)

    def test_substring_match_removed(self):
        """Topik baru yang merupakan substring topik lama → dibuang."""
        c = _make_candidate("AI")
        with mock.patch("agents.trend_scout.get_recent_topics", return_value=["AI最新ニュース"]):
            result = self.scout._filter_seen([c])
        self.assertEqual(len(result), 0)

    def test_no_seen_history_all_pass(self):
        """Jika DB kosong, semua topik lolos."""
        candidates = [_make_candidate(f"Topic {i}") for i in range(5)]
        with mock.patch("agents.trend_scout.get_recent_topics", return_value=[]):
            result = self.scout._filter_seen(candidates)
        self.assertEqual(len(result), 5)


class TestParseFeedDate(unittest.TestCase):
    """Test helper _parse_entry_date()."""

    def test_uses_published_parsed(self):
        from agents.trend_scout import _parse_entry_date
        entry = mock.MagicMock()
        entry.published_parsed = (2024, 1, 15, 10, 0, 0, 0, 0, 0)
        entry.updated_parsed = None
        result = _parse_entry_date(entry)
        self.assertEqual(result.year, 2024)
        self.assertEqual(result.month, 1)
        self.assertEqual(result.tzinfo, timezone.utc)

    def test_falls_back_to_updated_parsed(self):
        from agents.trend_scout import _parse_entry_date
        entry = mock.MagicMock()
        entry.published_parsed = None
        entry.updated_parsed = (2024, 6, 1, 12, 0, 0, 0, 0, 0)
        result = _parse_entry_date(entry)
        self.assertEqual(result.month, 6)
        self.assertEqual(result.day, 1)

    def test_defaults_to_now_if_no_date(self):
        """Entry tanpa tanggal → default sekarang (agar tidak dibuang)."""
        from agents.trend_scout import _parse_entry_date
        entry = mock.MagicMock()
        entry.published_parsed = None
        entry.updated_parsed = None
        before = datetime.now(timezone.utc)
        result = _parse_entry_date(entry)
        after = datetime.now(timezone.utc)
        self.assertGreaterEqual(result, before)
        self.assertLessEqual(result, after)

    def test_handles_invalid_date(self):
        """published_parsed yang korup → fallback ke sekarang."""
        from agents.trend_scout import _parse_entry_date
        entry = mock.MagicMock()
        entry.published_parsed = (99999,)  # invalid
        entry.updated_parsed = None
        # Harus tidak raise
        result = _parse_entry_date(entry)
        self.assertIsInstance(result, datetime)


class TestParseFeed(unittest.TestCase):
    """Test _parse_feed() dengan mock HTTP."""

    def _mock_entry(self, title, link="http://example.com"):
        entry = mock.MagicMock()
        entry.get = lambda k, d="": {"title": title, "link": link, "summary": ""}.get(k, d)
        entry.published_parsed = (2024, 6, 23, 10, 0, 0, 0, 0, 0)
        entry.updated_parsed = None
        return entry

    def _make_mock_resp(self):
        r = mock.MagicMock()
        r.content = b"<rss/>"
        r.raise_for_status = mock.MagicMock()
        return r

    def test_returns_candidates(self):
        from agents.trend_scout import TrendScout
        scout = TrendScout()
        scout.max_per_feed = 5

        mock_feed = mock.MagicMock()
        mock_feed.entries = [self._mock_entry(f"ニュース{i}") for i in range(3)]

        with mock.patch("requests.get", return_value=self._make_mock_resp()), \
             mock.patch("feedparser.parse", return_value=mock_feed):
            results = scout._parse_feed("http://test.rss", "Test Feed", "ja")

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].topic, "ニュース0")
        self.assertEqual(results[0].source, "Test Feed")
        self.assertEqual(results[0].category, "ja")

    def test_respects_max_per_feed(self):
        from agents.trend_scout import TrendScout
        scout = TrendScout()
        scout.max_per_feed = 2

        mock_feed = mock.MagicMock()
        mock_feed.entries = [self._mock_entry(f"Entry{i}") for i in range(10)]

        with mock.patch("requests.get", return_value=self._make_mock_resp()), \
             mock.patch("feedparser.parse", return_value=mock_feed):
            results = scout._parse_feed("http://test.rss", "Test", "en")

        self.assertEqual(len(results), 2)  # max_per_feed ditegakkan

    def test_skips_empty_title(self):
        from agents.trend_scout import TrendScout
        scout = TrendScout()
        scout.max_per_feed = 5

        empty_entry = mock.MagicMock()
        empty_entry.get = lambda k, d="": {"title": "", "link": "", "summary": ""}.get(k, d)
        empty_entry.published_parsed = None
        empty_entry.updated_parsed = None

        mock_feed = mock.MagicMock()
        mock_feed.entries = [empty_entry, self._mock_entry("Valid Title")]

        with mock.patch("requests.get", return_value=self._make_mock_resp()), \
             mock.patch("feedparser.parse", return_value=mock_feed):
            results = scout._parse_feed("http://test.rss", "Test", "ja")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].topic, "Valid Title")


class TestFallback(unittest.TestCase):
    """Test fallback saat semua sumber gagal."""

    def test_fallback_returned_when_rss_fails(self):
        from agents.trend_scout import TrendScout
        scout = TrendScout()
        scout.rss_sources = [{"url": "http://fail.rss", "name": "Fail", "language": "ja"}]
        scout.gt_config = {"enabled": False}
        scout.fallback_topics = ["テストフォールバック1", "テストフォールバック2"]

        with mock.patch("requests.get", side_effect=Exception("network error")), \
             mock.patch("agents.trend_scout.get_recent_topics", return_value=[]):
            results = scout.scout()

        self.assertGreater(len(results), 0)
        self.assertTrue(all(c.source == "fallback" for c in results))
        self.assertIn("テストフォールバック1", [c.topic for c in results])

    def test_make_fallback_uses_config(self):
        from agents.trend_scout import TrendScout
        scout = TrendScout()
        scout.fallback_topics = ["FallbackA", "FallbackB"]
        result = scout._make_fallback()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].topic, "FallbackA")
        self.assertEqual(result[0].source, "fallback")

    def test_make_fallback_has_builtin_default(self):
        """Jika config tidak punya fallback_topics → pakai built-in."""
        from agents.trend_scout import TrendScout
        scout = TrendScout()
        scout.fallback_topics = []
        result = scout._make_fallback()
        self.assertGreater(len(result), 0)


class TestScoutIntegration(unittest.TestCase):
    """Test scout() method dengan semua mock terpasang."""

    def _mock_rss_entry(self, title):
        entry = mock.MagicMock()
        entry.get = lambda k, d="": {"title": title, "link": "http://t.co", "summary": ""}.get(k, d)
        entry.published_parsed = None
        entry.updated_parsed = None
        return entry

    def test_scout_returns_at_most_max_candidates(self):
        from agents.trend_scout import TrendScout
        scout = TrendScout()
        scout.max_per_feed = 10
        scout.max_candidates = 5

        # Buat banyak entry
        mock_feed = mock.MagicMock()
        mock_feed.entries = [self._mock_rss_entry(f"Topic{i}") for i in range(20)]

        mock_resp = mock.MagicMock()
        mock_resp.content = b"<rss/>"
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("requests.get", return_value=mock_resp), \
             mock.patch("feedparser.parse", return_value=mock_feed), \
             mock.patch("agents.trend_scout.get_recent_topics", return_value=[]):
            scout.gt_config = {"enabled": False}
            results = scout.scout()

        self.assertLessEqual(len(results), 5)  # max_candidates ditegakkan

    def test_scout_filters_seen_before_returning(self):
        from agents.trend_scout import TrendScout
        scout = TrendScout()
        scout.max_per_feed = 3
        scout.max_candidates = 20

        mock_feed = mock.MagicMock()
        mock_feed.entries = [
            self._mock_rss_entry("Topic Already Seen"),
            self._mock_rss_entry("Topic Brand New"),
        ]

        mock_resp = mock.MagicMock()
        mock_resp.content = b"<rss/>"
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("requests.get", return_value=mock_resp), \
             mock.patch("feedparser.parse", return_value=mock_feed), \
             mock.patch("agents.trend_scout.get_recent_topics",
                        return_value=["Topic Already Seen"]):
            scout.gt_config = {"enabled": False}
            results = scout.scout()

        topics = [r.topic for r in results]
        self.assertNotIn("Topic Already Seen", topics)
        self.assertIn("Topic Brand New", topics)


if __name__ == "__main__":
    unittest.main()
