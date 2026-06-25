"""
agents/trend_reach.py — TrendScout multi-source (varian "Agent Reach").

Mengambil kandidat topik dari beberapa sumber SECARA PARALEL lalu
mengembalikannya dalam format yang IDENTIK dengan agents/trend_scout:

    scout_trends() -> List[TrendCandidate]

Sumber:
  1. Twitter/X Search — via CLI `twitter` (mis. dari agent-reach), JIKA terpasang
  2. Web Search       — via CLI `mcporter`/`exa` (Exa MCP), JIKA terpasang
  3. Reddit           — via JSON publik (requests), TANPA cookie, jalan di mana saja
  4. RSS feeds        — fallback, memakai ulang logika TrendScout lama

Prinsip desain (graceful degradation):
  - Tiap sumber dibungkus try/except per-source. CLI yang tidak ada → di-skip
    dengan log warning, BUKAN meng-crash seluruh fetch.
  - Sumber berbasis CLI memakai TEMPLATE perintah dari config (`agent_reach`),
    sehingga perintah persisnya bisa disetel di VPS tanpa mengubah kode.
  - `fetch_trends()` mengikuti brief: raise TrendFetchError jika hasil < 3.
  - `scout_trends()` (entry point pipeline) menangkap kegagalan itu dan jatuh
    ke RSS + fallback_topics, sehingga pipeline TIDAK PERNAH crash.

Feature flag: orchestrator memilih modul ini bila USE_AGENT_REACH=true.
"""
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Callable, List, Optional

import requests

from core.config import load_config
from core.logger import get_logger
from core.models import TrendCandidate
from database.db_manager import get_recent_topics

logger = get_logger("trend_reach")

_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TrendReach/1.0)"}
_SIMILARITY_CUTOFF = 0.82   # ambang kemiripan judul untuk dianggap duplikat
_CLI_TIMEOUT = 25           # detik, batas tiap pemanggilan CLI


class TrendFetchError(RuntimeError):
    """Dilempar saat sumber-sumber kaya gagal mengumpulkan cukup topik."""


# ======================================================================
# Konfigurasi
# ======================================================================

def _reach_config() -> dict:
    """
    Ambil blok `agent_reach` dari config.yaml dengan default yang aman.
    Semua nilai bisa di-override lewat config tanpa menyentuh kode.
    """
    cfg = load_config().get("agent_reach", {}) or {}

    tw = cfg.get("twitter", {}) or {}
    web = cfg.get("web_search", {}) or {}
    rd = cfg.get("reddit", {}) or {}

    return {
        "twitter": {
            "enabled": tw.get("enabled", True),
            # Template argv; {kw} diganti keyword. Output diharap JSON di stdout.
            "command": tw.get("command", ["twitter", "search", "{kw}", "--json"]),
            "keywords": tw.get("keywords", ["AI", "ゲーム", "アニメ", "日本 バズ"]),
            "per_keyword": int(tw.get("per_keyword", 10)),
            "min_likes": int(tw.get("min_likes", 100)),
            "min_retweets": int(tw.get("min_retweets", 20)),
        },
        "web_search": {
            "enabled": web.get("enabled", True),
            # {query} & {n} diganti. Default mengarah ke mcporter+Exa.
            "command": web.get("command",
                               ["mcporter", "call", "exa", "search",
                                "--query", "{query}", "--num", "{n}"]),
            "queries": web.get("queries", [
                "日本 トレンド 最新", "日本 テクノロジー ニュース 今週",
                "Japan viral this week",
            ]),
            "per_query": int(web.get("per_query", 8)),
        },
        "reddit": {
            "enabled": rd.get("enabled", True),
            "use_cli": rd.get("use_cli", False),    # true → rdt-cli; default JSON publik
            "command": rd.get("command",
                              ["rdt", "hot", "{subreddit}", "--json", "--limit", "{n}"]),
            "subreddits": rd.get("subreddits",
                                 ["Japan", "technology", "gaming", "artificial"]),
            "min_upvotes": int(rd.get("min_upvotes", 500)),
            "per_subreddit": int(rd.get("per_subreddit", 10)),
            "timeframe": rd.get("timeframe", "day"),
        },
        "min_results": int(cfg.get("min_results", 3)),
    }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _have_cli(argv: List[str]) -> bool:
    """True bila binary pertama dari template perintah ada di PATH."""
    return bool(argv) and shutil.which(argv[0]) is not None


# ======================================================================
# Sumber 1 — Twitter/X Search (CLI, opsional)
# ======================================================================

def _from_twitter(conf: dict) -> List[TrendCandidate]:
    cmd_tmpl = conf["command"]
    if not conf["enabled"]:
        return []
    if not _have_cli(cmd_tmpl):
        logger.warning("twitter-cli (%s) tidak terpasang — sumber Twitter dilewati",
                       cmd_tmpl[0] if cmd_tmpl else "?")
        return []

    out: List[TrendCandidate] = []
    for kw in conf["keywords"]:
        argv = [part.replace("{kw}", kw) for part in cmd_tmpl]
        raw = _run_cli(argv)
        if not raw:
            continue
        for item in _iter_json_items(raw)[: conf["per_keyword"]]:
            likes = _first_int(item, ("favorite_count", "like_count", "likes",
                                      "public_metrics.like_count"))
            rts = _first_int(item, ("retweet_count", "retweets",
                                    "public_metrics.retweet_count"))
            if likes < conf["min_likes"] and rts < conf["min_retweets"]:
                continue
            text = _first_str(item, ("text", "full_text", "content")).strip()
            if not text:
                continue
            out.append(TrendCandidate(
                topic=text[:200],
                source="Twitter/X",
                url=_first_str(item, ("url", "permalink", "link")),
                freshness=_now(),
                raw_summary=text[:300],
                category="ja",
            ))
    return out


# ======================================================================
# Sumber 2 — Web Search via Exa/mcporter (CLI, opsional)
# ======================================================================

def _from_web_search(conf: dict) -> List[TrendCandidate]:
    cmd_tmpl = conf["command"]
    if not conf["enabled"]:
        return []
    if not _have_cli(cmd_tmpl):
        logger.warning("web-search CLI (%s) tidak terpasang — sumber Web dilewati",
                       cmd_tmpl[0] if cmd_tmpl else "?")
        return []

    out: List[TrendCandidate] = []
    n = str(conf["per_query"])
    for query in conf["queries"]:
        argv = [part.replace("{query}", query).replace("{n}", n) for part in cmd_tmpl]
        raw = _run_cli(argv)
        if not raw:
            continue
        for item in _iter_json_items(raw)[: conf["per_query"]]:
            title = _first_str(item, ("title", "text", "name")).strip()
            if not title:
                continue
            out.append(TrendCandidate(
                topic=title[:200],
                source="Web Search",
                url=_first_str(item, ("url", "link")),
                freshness=_now(),
                raw_summary=_first_str(item, ("snippet", "summary", "text"))[:300],
                category="ja",
            ))
    return out


# ======================================================================
# Sumber 3 — Reddit (JSON publik, tanpa cookie — jalan di mana saja)
# ======================================================================

def _from_reddit(conf: dict) -> List[TrendCandidate]:
    """
    Dispatcher Reddit:
      - use_cli=true + `rdt` terpasang → jalur rdt-cli (agent-reach)
      - selain itu                     → JSON publik (tanpa cookie, default)
    rdt-cli yang diminta tapi belum terpasang otomatis jatuh ke JSON publik.
    """
    if not conf["enabled"]:
        return []

    if conf.get("use_cli") and conf.get("command"):
        if _have_cli(conf["command"]):
            return _from_reddit_cli(conf)
        logger.warning("rdt-cli (%s) diminta tapi tidak terpasang — pakai JSON publik",
                       conf["command"][0])
    return _from_reddit_json(conf)


def _from_reddit_cli(conf: dict) -> List[TrendCandidate]:
    """Ambil hot posts via CLI `rdt` (agent-reach). Output JSON di stdout."""
    cmd_tmpl = conf["command"]
    n = str(conf["per_subreddit"])
    out: List[TrendCandidate] = []
    for sub in conf["subreddits"]:
        argv = [p.replace("{subreddit}", sub).replace("{sub}", sub).replace("{n}", n)
                for p in cmd_tmpl]
        raw = _run_cli(argv)
        if not raw:
            continue
        for item in _iter_json_items(raw)[: conf["per_subreddit"]]:
            if item.get("stickied") or _dig(item, "data.stickied"):
                continue
            ups = _first_int(item, ("ups", "score", "upvotes",
                                    "data.ups", "data.score"))
            if ups < conf["min_upvotes"]:
                continue
            title = _first_str(item, ("title", "data.title")).strip()
            if not title:
                continue
            url = _first_str(item, ("url", "permalink", "data.permalink", "data.url"))
            if url.startswith("/r/"):
                url = "https://www.reddit.com" + url
            out.append(TrendCandidate(
                topic=title[:200],
                source=f"Reddit r/{sub}",
                url=url,
                freshness=_now(),
                raw_summary=_first_str(item, ("selftext", "data.selftext"))[:300],
                category="en",
            ))
    return out


def _from_reddit_json(conf: dict) -> List[TrendCandidate]:
    out: List[TrendCandidate] = []
    for sub in conf["subreddits"]:
        url = f"https://www.reddit.com/r/{sub}/hot.json"
        params = {"limit": conf["per_subreddit"], "t": conf["timeframe"]}
        try:
            resp = requests.get(url, params=params, headers=_HTTP_HEADERS, timeout=10)
            resp.raise_for_status()
            children = resp.json().get("data", {}).get("children", [])
        except requests.exceptions.RequestException as e:
            logger.warning("Reddit r/%s gagal: %s", sub, e)
            continue
        except (ValueError, KeyError) as e:
            logger.warning("Reddit r/%s respons tak terduga: %s", sub, e)
            continue

        for child in children:
            data = child.get("data", {})
            if data.get("stickied"):
                continue
            if int(data.get("ups", 0)) < conf["min_upvotes"]:
                continue
            title = (data.get("title") or "").strip()
            if not title:
                continue
            permalink = data.get("permalink", "")
            out.append(TrendCandidate(
                topic=title[:200],
                source=f"Reddit r/{sub}",
                url=f"https://www.reddit.com{permalink}" if permalink else "",
                freshness=_now(),
                raw_summary=(data.get("selftext") or "")[:300],
                category="en",
            ))
    return out


# ======================================================================
# Sumber 4 — RSS (fallback; memakai ulang TrendScout lama)
# ======================================================================

def _fallback_rss() -> List[TrendCandidate]:
    """
    Pakai ulang logika RSS TrendScout lama agar tidak duplikasi daftar feed.
    Tidak pernah meng-crash: TrendScout sudah toleran per-feed.
    """
    try:
        from agents.trend_scout import TrendScout
        return TrendScout()._from_rss()
    except Exception as e:  # pragma: no cover - defensif
        logger.warning("Fallback RSS gagal: %s", e)
        return []


# ======================================================================
# Helper CLI / JSON
# ======================================================================

def _run_cli(argv: List[str]) -> str:
    """Jalankan CLI, kembalikan stdout (str). String kosong jika gagal."""
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=_CLI_TIMEOUT,
        )
        if proc.returncode != 0:
            logger.warning("CLI %s exit %d: %s",
                           argv[0], proc.returncode, (proc.stderr or "").strip()[:200])
            return ""
        return proc.stdout or ""
    except FileNotFoundError:
        logger.warning("CLI %s tidak ditemukan saat dijalankan", argv[0])
        return ""
    except subprocess.TimeoutExpired:
        logger.warning("CLI %s timeout (%ds)", argv[0], _CLI_TIMEOUT)
        return ""
    except Exception as e:  # pragma: no cover - defensif
        logger.warning("CLI %s error: %s", argv[0], e)
        return ""


def _iter_json_items(raw: str) -> List[dict]:
    """
    Parse stdout JSON menjadi list dict secara toleran.
    Menerima: list langsung, {"results": [...]}, {"data": [...]},
    {"tweets": [...]}, atau JSONL (satu objek per baris).
    """
    raw = raw.strip()
    if not raw:
        return []
    # Coba JSON utuh dulu
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            for key in ("results", "data", "tweets", "items", "hits", "posts"):
                val = data.get(key)
                if isinstance(val, list):
                    return [x for x in val if isinstance(x, dict)]
            return [data]
    except json.JSONDecodeError:
        pass
    # Fallback: JSONL
    items: List[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                items.append(obj)
        except json.JSONDecodeError:
            continue
    return items


def _dig(item: dict, dotted: str):
    """Ambil nilai bersarang via key bertitik, mis. 'public_metrics.like_count'."""
    cur = item
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _first_int(item: dict, keys) -> int:
    for k in keys:
        val = _dig(item, k)
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, str) and val.isdigit():
            return int(val)
    return 0


def _first_str(item: dict, keys) -> str:
    for k in keys:
        val = _dig(item, k)
        if isinstance(val, str) and val:
            return val
    return ""


# ======================================================================
# Deduplikasi (string similarity sederhana, tanpa embedding)
# ======================================================================

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _dedupe_internal(cands: List[TrendCandidate]) -> List[TrendCandidate]:
    """Buang kandidat yang saling mirip (pertahankan yang pertama)."""
    kept: List[TrendCandidate] = []
    for c in cands:
        if any(_similar(c.topic, k.topic) >= _SIMILARITY_CUTOFF for k in kept):
            continue
        kept.append(c)
    return kept


def _dedupe_against_db(cands: List[TrendCandidate]) -> List[TrendCandidate]:
    """
    Buang topik yang sudah dipakai baru-baru ini (substring ATAU similarity).
    get_recent_topics sudah dibatasi memory; brief minta jangkauan ~7 hari.
    """
    try:
        recent = [t.lower() for t in get_recent_topics(50)]
    except Exception as e:  # DB belum di-init / hiccup → jangan crash, simpan semua
        logger.warning("Gagal baca recent topics (%s) — lewati dedup DB", e)
        return cands
    if not recent:
        return cands

    fresh: List[TrendCandidate] = []
    for c in cands:
        tl = c.topic.lower()
        seen = any(
            (r in tl or tl in r or _similar(tl, r) >= _SIMILARITY_CUTOFF)
            for r in recent
        )
        if not seen:
            fresh.append(c)
    return fresh


# ======================================================================
# Orkestrasi paralel
# ======================================================================

def fetch_trends() -> List[TrendCandidate]:
    """
    Ambil dari sumber-sumber kaya secara paralel, gabung, dedup.
    Mengikuti brief: raise TrendFetchError bila hasil akhir < min_results.

    CATATAN: ini TIDAK menyertakan RSS fallback — itu tugas scout_trends()
    agar pemisahan "sumber kaya" vs "jaring pengaman" tetap jelas.
    """
    conf = _reach_config()

    # Tiap entri: (nama, fungsi tanpa argumen)
    jobs: List[tuple[str, Callable[[], List[TrendCandidate]]]] = [
        ("twitter", lambda: _from_twitter(conf["twitter"])),
        ("web_search", lambda: _from_web_search(conf["web_search"])),
        ("reddit", lambda: _from_reddit(conf["reddit"])),
    ]

    collected: List[TrendCandidate] = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {}
        for name, fn in jobs:
            futures[pool.submit(_timed_source, name, fn)] = name
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                items = fut.result()
                collected.extend(items)
            except Exception as e:  # per-source tidak boleh menjatuhkan total
                logger.warning("Sumber '%s' gagal total: %s", name, e)

    deduped = _dedupe_against_db(_dedupe_internal(collected))
    logger.info("Agent Reach: %d mentah → %d setelah dedup", len(collected), len(deduped))

    if len(deduped) < conf["min_results"]:
        raise TrendFetchError(
            f"Hanya {len(deduped)} topik dari sumber kaya "
            f"(min {conf['min_results']})"
        )
    return deduped


def _timed_source(name: str, fn: Callable[[], List[TrendCandidate]]) -> List[TrendCandidate]:
    """Bungkus pemanggilan satu sumber: ukur waktu + log jumlah/eror."""
    t0 = time.monotonic()
    try:
        items = fn()
        dt = time.monotonic() - t0
        logger.info("Sumber '%s': %d item dalam %.2fs", name, len(items), dt)
        return items
    except Exception as e:
        dt = time.monotonic() - t0
        logger.warning("Sumber '%s' error setelah %.2fs: %s", name, dt, e)
        return []


# ======================================================================
# Entry point pipeline — IDENTIK dengan agents/trend_scout.scout_trends()
# ======================================================================

def scout_trends() -> List[TrendCandidate]:
    """
    Entry point untuk orchestrator. Selalu mengembalikan List[TrendCandidate]
    (bisa kosong hanya jika RSS + fallback juga kosong, sangat tidak mungkin).

    Urutan jaring pengaman:
      1. Sumber kaya (Twitter/Web/Reddit) via fetch_trends()
      2. Jika gagal / < min_results → RSS (TrendScout lama)
      3. Jika RSS juga kosong → fallback_topics dari TrendScout
    """
    import random

    cfg = load_config()
    max_candidates = cfg.get("trend_scout", {}).get("max_candidates", 20)

    candidates: List[TrendCandidate] = []
    try:
        candidates = fetch_trends()
    except TrendFetchError as e:
        logger.warning("Sumber kaya kurang (%s) — jatuh ke RSS", e)

    if not candidates:
        rss = _dedupe_against_db(_dedupe_internal(_fallback_rss()))
        logger.info("RSS fallback: %d topik segar", len(rss))
        candidates = rss

    if not candidates:
        # Jaring pengaman terakhir: fallback_topics dari TrendScout
        try:
            from agents.trend_scout import TrendScout
            candidates = TrendScout()._make_fallback()
            logger.warning("Memakai fallback_topics statis (%d)", len(candidates))
        except Exception as e:  # pragma: no cover
            logger.error("Fallback statis gagal: %s", e)
            candidates = []

    random.shuffle(candidates)
    return candidates[:max_candidates]


if __name__ == "__main__":
    for i, c in enumerate(scout_trends()[:10], 1):
        print(f"{i}. [{c.source}] {c.topic}")
