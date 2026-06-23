"""
agents/trend_scout.py — TrendScout agent.

Mengumpulkan kandidat topik segar dari:
  1. RSS feeds (dikonfigurasi di config.yaml)
  2. Google Trends Japan via pytrends (jika diaktifkan)

Output: List[TrendCandidate] — topik segar yang belum pernah dipakai,
        diurutkan secara acak agar tidak monoton satu sumber.
"""
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import feedparser
import requests

from core.config import load_config
from core.logger import get_logger
from core.models import TrendCandidate
from database.db_manager import get_recent_topics

logger = get_logger("trend_scout")

# Header agar tidak diblok feed yang memeriksa User-Agent
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RSS-Reader/1.0)"}


class TrendScout:
    """Mengumpulkan kandidat topik dari RSS + Google Trends."""

    def __init__(self):
        cfg = load_config()
        self.rss_sources: list = cfg.get("rss_sources", [])
        self.gt_config: dict = cfg.get("google_trends", {})
        ts = cfg.get("trend_scout", {})
        self.max_age_hours: int = ts.get("max_age_hours", 48)
        self.max_per_feed: int = ts.get("max_per_feed", 5)
        self.max_candidates: int = ts.get("max_candidates", 20)
        self.fallback_topics: list = cfg.get("fallback_topics", [])

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def scout(self) -> List[TrendCandidate]:
        """
        Kumpulkan kandidat dari semua sumber, filter basi + sudah dipakai.
        Jika semua sumber gagal, kembalikan fallback_topics dari config.
        """
        candidates: List[TrendCandidate] = []

        # Sumber 1: RSS
        rss = self._from_rss()
        candidates.extend(rss)
        logger.info("RSS: %d entries dari %d feed", len(rss), len(self.rss_sources))

        # Sumber 2: Google Trends JP (opsional)
        if self.gt_config.get("enabled", False):
            gt = self._from_google_trends()
            candidates.extend(gt)
            logger.info("Google Trends JP: %d topics", len(gt))

        # Filter: buang yang basi dan yang sudah dipakai
        fresh = self._filter_fresh(candidates)
        new_topics = self._filter_seen(fresh)

        logger.info(
            "TrendScout total: %d → segar: %d → baru: %d",
            len(candidates), len(fresh), len(new_topics),
        )

        if not new_topics:
            logger.warning("Tidak ada topik baru, menggunakan fallback_topics")
            new_topics = self._make_fallback()

        random.shuffle(new_topics)
        return new_topics[:self.max_candidates]

    # ------------------------------------------------------------------
    # Sumber 1 — RSS Feeds
    # ------------------------------------------------------------------

    def _from_rss(self) -> List[TrendCandidate]:
        """Iterasi semua feed dari config, toleran terhadap kegagalan per-feed."""
        results: List[TrendCandidate] = []
        for source in self.rss_sources:
            url = source.get("url", "")
            name = source.get("name", url)
            lang = source.get("language", "en")
            try:
                entries = self._parse_feed(url, name, lang)
                results.extend(entries)
                logger.debug("Feed '%s': %d entries", name, len(entries))
            except requests.exceptions.Timeout:
                logger.warning("Feed '%s' timeout (10s), dilewati", name)
            except requests.exceptions.RequestException as e:
                logger.warning("Feed '%s' HTTP error: %s", name, e)
            except Exception as e:
                logger.warning("Feed '%s' parse error: %s", name, e)
        return results

    def _parse_feed(self, url: str, source_name: str, lang: str) -> List[TrendCandidate]:
        """
        Fetch dengan requests (agar timeout berlaku) lalu parse dengan feedparser.
        feedparser.parse(bytes) lebih andal daripada parse(url) untuk timeout.
        """
        resp = requests.get(url, timeout=10, headers=_HTTP_HEADERS)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)

        candidates: List[TrendCandidate] = []
        for entry in feed.entries[:self.max_per_feed]:
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            candidates.append(TrendCandidate(
                topic=title,
                source=source_name,
                url=entry.get("link") or "",
                freshness=_parse_entry_date(entry),
                raw_summary=(entry.get("summary") or "")[:300],
                category=lang,
            ))
        return candidates

    # ------------------------------------------------------------------
    # Sumber 2 — Google Trends Japan
    # ------------------------------------------------------------------

    def _from_google_trends(self) -> List[TrendCandidate]:
        """
        Ambil realtime trending dari Google Trends JP via pytrends.
        Disabled by default di config karena sering kena rate-limit.
        """
        try:
            from pytrends.request import TrendReq
        except ImportError:
            logger.warning("pytrends belum terinstall, Google Trends dilewati")
            return []

        try:
            geo = self.gt_config.get("geo", "JP")
            # tz=-540: JST (UTC+9 = -540 menit dalam konvensi pytrends)
            pytrends = TrendReq(hl="ja-JP", tz=-540, timeout=(10, 30))
            df = pytrends.realtime_trending_searches(pn=geo)
            now = datetime.now(timezone.utc)

            candidates: List[TrendCandidate] = []
            for _, row in df.head(15).iterrows():
                title = row.get("title") or row.get("entityNames") or ""
                if isinstance(title, list):
                    title = title[0] if title else ""
                title = str(title).strip()
                if title:
                    candidates.append(TrendCandidate(
                        topic=title,
                        source="Google Trends JP",
                        url="",
                        freshness=now,
                        raw_summary="",
                        category="ja",
                    ))
            return candidates
        except Exception as e:
            logger.warning("Google Trends gagal: %s", e)
            return []

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def _filter_fresh(self, candidates: List[TrendCandidate]) -> List[TrendCandidate]:
        """
        Buang topik yang lebih tua dari max_age_hours.
        Jika entry tidak punya tanggal → dianggap segar (defaultnya 'now').
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.max_age_hours)
        return [c for c in candidates if _ensure_utc(c.freshness) >= cutoff]

    def _filter_seen(self, candidates: List[TrendCandidate]) -> List[TrendCandidate]:
        """
        Buang topik yang sudah pernah dipakai (pencocokan substring kasar).
        Dedup semantik yang lebih akurat akan dilakukan oleh Dedup agent.
        """
        seen_lower = [t.lower() for t in get_recent_topics(50)]
        if not seen_lower:
            return candidates

        new: List[TrendCandidate] = []
        for c in candidates:
            topic_lower = c.topic.lower()
            already_seen = any(
                seen_t in topic_lower or topic_lower in seen_t
                for seen_t in seen_lower
            )
            if not already_seen:
                new.append(c)
        return new

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _make_fallback(self) -> List[TrendCandidate]:
        """Buat TrendCandidate dari fallback_topics di config."""
        now = datetime.now(timezone.utc)
        topics = self.fallback_topics or [
            "AI技術の最新動向",
            "アニメ新シーズン情報",
            "日本のゲーム新作",
            "SNSで話題のトレンド",
            "テクノロジー最新ニュース",
        ]
        return [
            TrendCandidate(topic=t, source="fallback", freshness=now, category="ja")
            for t in topics
        ]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_entry_date(entry) -> datetime:
    """
    Parse tanggal dari entry feedparser.
    Coba published_parsed, lalu updated_parsed.
    Default: sekarang (agar tidak dibuang filter freshness).
    """
    for attr in ("published_parsed", "updated_parsed"):
        val = getattr(entry, attr, None)
        if val:
            try:
                return datetime(*val[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                pass
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime) -> datetime:
    """Pastikan datetime adalah UTC-aware (tambahkan timezone jika naive)."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ------------------------------------------------------------------
# Shortcut untuk orchestrator
# ------------------------------------------------------------------

def scout_trends() -> List[TrendCandidate]:
    """Jalankan TrendScout dan kembalikan list kandidat."""
    return TrendScout().scout()
