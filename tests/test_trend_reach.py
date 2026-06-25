"""
tests/test_trend_reach.py — Unit test untuk agents/trend_reach.py

CLI, HTTP, dan DB di-mock agar test cepat & tidak bergantung jaringan
atau binary eksternal (twitter/mcporter). Fokus:
  - parsing JSON toleran
  - filter likes/retweets & upvotes
  - graceful degradation saat CLI tidak ada
  - dedup (internal + terhadap DB)
  - scout_trends() selalu mengembalikan list (jatuh ke RSS/fallback)
  - fetch_trends() raise TrendFetchError saat hasil < min_results
"""
import json
import unittest
from unittest import mock

from core.models import TrendCandidate
from agents import trend_reach as tr


# ----------------------------------------------------------------------
# Parsing JSON toleran
# ----------------------------------------------------------------------
class TestJsonParsing(unittest.TestCase):
    def test_plain_list(self):
        self.assertEqual(tr._iter_json_items('[{"a":1}]'), [{"a": 1}])

    def test_wrapped_keys(self):
        for key in ("results", "data", "tweets", "items", "hits", "posts"):
            raw = json.dumps({key: [{"x": 1}]})
            self.assertEqual(tr._iter_json_items(raw), [{"x": 1}])

    def test_single_object(self):
        self.assertEqual(tr._iter_json_items('{"k":9}'), [{"k": 9}])

    def test_jsonl(self):
        self.assertEqual(tr._iter_json_items('{"a":1}\n{"b":2}'), [{"a": 1}, {"b": 2}])

    def test_empty_and_garbage(self):
        self.assertEqual(tr._iter_json_items(""), [])
        self.assertEqual(tr._iter_json_items("not json"), [])


class TestExtractors(unittest.TestCase):
    def test_dig_nested(self):
        item = {"public_metrics": {"like_count": 5}}
        self.assertEqual(tr._dig(item, "public_metrics.like_count"), 5)
        self.assertIsNone(tr._dig(item, "public_metrics.missing"))

    def test_first_int_prefers_first_present(self):
        item = {"favorite_count": 120, "likes": 999}
        self.assertEqual(tr._first_int(item, ("favorite_count", "likes")), 120)

    def test_first_int_string_digit(self):
        self.assertEqual(tr._first_int({"ups": "500"}, ("ups",)), 500)

    def test_first_int_default_zero(self):
        self.assertEqual(tr._first_int({}, ("nope",)), 0)

    def test_first_str(self):
        self.assertEqual(tr._first_str({"text": "hi"}, ("full_text", "text")), "hi")
        self.assertEqual(tr._first_str({}, ("x",)), "")


# ----------------------------------------------------------------------
# Sumber Twitter (CLI di-mock)
# ----------------------------------------------------------------------
class TestTwitterSource(unittest.TestCase):
    def _conf(self, **over):
        base = {
            "enabled": True,
            "command": ["twitter", "search", "{kw}", "--json"],
            "keywords": ["AI"],
            "per_keyword": 10,
            "min_likes": 100,
            "min_retweets": 20,
        }
        base.update(over)
        return base

    def test_skipped_when_cli_absent(self):
        with mock.patch("agents.trend_reach._have_cli", return_value=False):
            self.assertEqual(tr._from_twitter(self._conf()), [])

    def test_filters_low_engagement(self):
        payload = json.dumps([
            {"text": "viral banget", "favorite_count": 500, "retweet_count": 0},
            {"text": "sepi", "favorite_count": 5, "retweet_count": 1},
        ])
        with mock.patch("agents.trend_reach._have_cli", return_value=True), \
             mock.patch("agents.trend_reach._run_cli", return_value=payload):
            out = tr._from_twitter(self._conf())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].topic, "viral banget")
        self.assertEqual(out[0].source, "Twitter/X")

    def test_retweet_threshold_alone_passes(self):
        payload = json.dumps([{"text": "rt heavy", "favorite_count": 0, "retweet_count": 50}])
        with mock.patch("agents.trend_reach._have_cli", return_value=True), \
             mock.patch("agents.trend_reach._run_cli", return_value=payload):
            out = tr._from_twitter(self._conf())
        self.assertEqual(len(out), 1)

    def test_disabled_returns_empty(self):
        self.assertEqual(tr._from_twitter(self._conf(enabled=False)), [])


# ----------------------------------------------------------------------
# Sumber Reddit (HTTP di-mock)
# ----------------------------------------------------------------------
class TestRedditSource(unittest.TestCase):
    def _conf(self, **over):
        base = {
            "enabled": True,
            "subreddits": ["Japan"],
            "min_upvotes": 500,
            "per_subreddit": 10,
            "timeframe": "day",
        }
        base.update(over)
        return base

    def _resp(self, children):
        r = mock.MagicMock()
        r.raise_for_status = mock.MagicMock()
        r.json.return_value = {"data": {"children": children}}
        return r

    def test_filters_upvotes_and_stickied(self):
        children = [
            {"data": {"title": "hot post", "ups": 800, "permalink": "/r/Japan/x"}},
            {"data": {"title": "low", "ups": 100}},
            {"data": {"title": "pinned", "ups": 9999, "stickied": True}},
        ]
        with mock.patch("requests.get", return_value=self._resp(children)):
            out = tr._from_reddit(self._conf())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].topic, "hot post")
        self.assertTrue(out[0].url.endswith("/r/Japan/x"))

    def test_network_error_skips_gracefully(self):
        import requests as req
        with mock.patch("requests.get", side_effect=req.exceptions.Timeout()):
            self.assertEqual(tr._from_reddit(self._conf()), [])

    def test_disabled_returns_empty(self):
        self.assertEqual(tr._from_reddit(self._conf(enabled=False)), [])


class TestRedditCliPath(unittest.TestCase):
    """Jalur rdt-cli opsional + dispatcher _from_reddit()."""

    def _conf(self, **over):
        base = {
            "enabled": True,
            "use_cli": True,
            "command": ["rdt", "hot", "{subreddit}", "--json", "--limit", "{n}"],
            "subreddits": ["Japan"],
            "min_upvotes": 500,
            "per_subreddit": 10,
            "timeframe": "day",
        }
        base.update(over)
        return base

    def test_cli_path_parses_and_filters(self):
        payload = json.dumps([
            {"title": "cli hot post", "ups": 700, "permalink": "/r/Japan/abc"},
            {"title": "low", "ups": 10},
            {"title": "pinned", "ups": 9999, "stickied": True},
        ])
        with mock.patch("agents.trend_reach._have_cli", return_value=True), \
             mock.patch("agents.trend_reach._run_cli", return_value=payload):
            out = tr._from_reddit(self._conf())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].topic, "cli hot post")
        self.assertTrue(out[0].url.endswith("/r/Japan/abc"))

    def test_cli_path_handles_nested_data_schema(self):
        payload = json.dumps([{"data": {"title": "nested", "score": 600,
                                        "permalink": "/r/Japan/n"}}])
        with mock.patch("agents.trend_reach._have_cli", return_value=True), \
             mock.patch("agents.trend_reach._run_cli", return_value=payload):
            out = tr._from_reddit(self._conf())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].topic, "nested")

    def test_falls_back_to_json_when_cli_absent(self):
        """use_cli=true tapi rdt tidak terpasang → pakai JSON publik."""
        json_resp = mock.MagicMock()
        json_resp.raise_for_status = mock.MagicMock()
        json_resp.json.return_value = {"data": {"children": [
            {"data": {"title": "via json", "ups": 800, "permalink": "/r/Japan/j"}},
        ]}}
        with mock.patch("agents.trend_reach._have_cli", return_value=False), \
             mock.patch("requests.get", return_value=json_resp) as mget:
            out = tr._from_reddit(self._conf())
        mget.assert_called_once()           # benar-benar lewat jalur HTTP
        self.assertEqual(out[0].topic, "via json")

    def test_use_cli_false_uses_json(self):
        json_resp = mock.MagicMock()
        json_resp.raise_for_status = mock.MagicMock()
        json_resp.json.return_value = {"data": {"children": [
            {"data": {"title": "json default", "ups": 900, "permalink": "/r/Japan/d"}},
        ]}}
        with mock.patch("agents.trend_reach._run_cli") as mcli, \
             mock.patch("requests.get", return_value=json_resp):
            out = tr._from_reddit(self._conf(use_cli=False))
        mcli.assert_not_called()            # CLI tidak dipanggil sama sekali
        self.assertEqual(out[0].topic, "json default")


# ----------------------------------------------------------------------
# Dedup
# ----------------------------------------------------------------------
class TestDedup(unittest.TestCase):
    def test_internal_removes_near_duplicates(self):
        c = [
            TrendCandidate(topic="AI最新ニュースまとめ", source="a"),
            TrendCandidate(topic="AI最新ニュースまとめ", source="b"),
            TrendCandidate(topic="全く別の話題だよこれ", source="c"),
        ]
        self.assertEqual(len(tr._dedupe_internal(c)), 2)

    def test_against_db_filters_seen(self):
        c = [TrendCandidate(topic="新しいゲームの話", source="a"),
             TrendCandidate(topic="昨日も話したこと", source="b")]
        with mock.patch("agents.trend_reach.get_recent_topics",
                        return_value=["昨日も話したこと"]):
            out = tr._dedupe_against_db(c)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].topic, "新しいゲームの話")

    def test_against_db_empty_history_keeps_all(self):
        c = [TrendCandidate(topic="x", source="a")]
        with mock.patch("agents.trend_reach.get_recent_topics", return_value=[]):
            self.assertEqual(len(tr._dedupe_against_db(c)), 1)


# ----------------------------------------------------------------------
# fetch_trends() / scout_trends() — orkestrasi & graceful degradation
# ----------------------------------------------------------------------
class TestOrchestration(unittest.TestCase):
    def test_fetch_raises_when_too_few(self):
        with mock.patch("agents.trend_reach._from_twitter", return_value=[]), \
             mock.patch("agents.trend_reach._from_web_search", return_value=[]), \
             mock.patch("agents.trend_reach._from_reddit", return_value=[]), \
             mock.patch("agents.trend_reach.get_recent_topics", return_value=[]):
            with self.assertRaises(tr.TrendFetchError):
                tr.fetch_trends()

    def test_fetch_succeeds_with_enough(self):
        reddit = [
            TrendCandidate(topic="新型ゲーム機の発表がきた", source="Reddit r/gaming"),
            TrendCandidate(topic="AIモデルがまた進化したらしい", source="Reddit r/artificial"),
            TrendCandidate(topic="京都の観光客が過去最多に", source="Reddit r/Japan"),
        ]
        with mock.patch("agents.trend_reach._from_twitter", return_value=[]), \
             mock.patch("agents.trend_reach._from_web_search", return_value=[]), \
             mock.patch("agents.trend_reach._from_reddit", return_value=reddit), \
             mock.patch("agents.trend_reach.get_recent_topics", return_value=[]):
            out = tr.fetch_trends()
        self.assertEqual(len(out), 3)

    def test_scout_falls_back_to_rss(self):
        """Sumber kaya gagal → scout_trends() pakai RSS, tetap mengembalikan list."""
        rss = [TrendCandidate(topic="RSS見出しのニュース記事", source="NHK")]
        with mock.patch("agents.trend_reach.fetch_trends",
                        side_effect=tr.TrendFetchError("kurang")), \
             mock.patch("agents.trend_reach._fallback_rss", return_value=rss), \
             mock.patch("agents.trend_reach.get_recent_topics", return_value=[]):
            out = tr.scout_trends()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].source, "NHK")

    def test_scout_returns_candidates_on_success(self):
        rich = [TrendCandidate(topic=f"成功した話題その{i}番", source="Reddit r/Japan")
                for i in range(5)]
        with mock.patch("agents.trend_reach.fetch_trends", return_value=rich):
            out = tr.scout_trends()
        self.assertEqual(len(out), 5)

    def test_scout_returns_list_type(self):
        with mock.patch("agents.trend_reach.fetch_trends",
                        side_effect=tr.TrendFetchError("x")), \
             mock.patch("agents.trend_reach._fallback_rss", return_value=[]), \
             mock.patch("agents.trend_reach.get_recent_topics", return_value=[]):
            out = tr.scout_trends()
        self.assertIsInstance(out, list)


if __name__ == "__main__":
    unittest.main()
