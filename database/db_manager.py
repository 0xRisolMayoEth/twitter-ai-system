"""
database/db_manager.py — Semua operasi SQLite.

Satu file DB terpusat: database/tweets.db
(konsolidasi dari tweets.db + memory.db versi lama)

Tabel:
  tweets       — konten yang dibuat (bilingual JP+ID, backward compat)
  topics       — topik hasil trend scouting
  topics_used  — memory topik & angle yang sudah dipakai (Strategist)
  embeddings   — vektor embedding untuk dedup semantik
  runs         — log tiap run (observability)
"""
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("twitter_ai.db")

DB_PATH = os.path.join(os.path.dirname(__file__), "tweets.db")


def _connect() -> sqlite3.Connection:
    """Buka koneksi baru per operasi (thread-safe)."""
    return sqlite3.connect(DB_PATH)


# ------------------------------------------------------------------
# Inisialisasi & migrasi skema
# ------------------------------------------------------------------

def init_db():
    """
    Buat / migrasi semua tabel. Aman dipanggil berulang (idempoten).
    Jika DB lama sudah ada, kolom baru ditambahkan tanpa menghapus data.
    """
    conn = _connect()
    c = conn.cursor()

    # --- tweets (extended dari skema lama) ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS tweets (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id        INTEGER REFERENCES topics(id),
            topic           TEXT,           -- nama topik (backward compat)
            content         TEXT,           -- alias content_jp (backward compat)
            content_jp      TEXT,           -- versi final Jepang
            content_indo    TEXT,           -- versi final Indonesia
            draft_jp        TEXT,           -- draft pertama JP
            score           INTEGER DEFAULT 0,
            score_breakdown TEXT DEFAULT '{}',   -- JSON per-aspek
            angle_type      TEXT DEFAULT '',
            status          TEXT DEFAULT 'pending',  -- pending|sent|failed
            created_at      TEXT,
            sent_at         TEXT,
            posted_at       TEXT,           -- alias sent_at (backward compat)
            tweet_id        TEXT            -- ID X post (referensi, tidak auto-post)
        )
    """)
    # Kolom baru pada tabel lama yang mungkin belum ada
    for col, typedef in [
        ("topic_id",        "INTEGER"),
        ("content_jp",      "TEXT"),
        ("content_indo",    "TEXT"),
        ("draft_jp",        "TEXT"),
        ("score",           "INTEGER DEFAULT 0"),
        ("score_breakdown", "TEXT DEFAULT '{}'"),
        ("angle_type",      "TEXT DEFAULT ''"),
        ("sent_at",         "TEXT"),
    ]:
        _add_column_safe(c, "tweets", col, typedef)

    # --- topics ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS topics (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            topic    TEXT NOT NULL,
            source   TEXT DEFAULT '',
            url      TEXT DEFAULT '',
            seen_at  TEXT NOT NULL,
            used     INTEGER DEFAULT 0
        )
    """)

    # --- topics_used (memory angle & topik untuk Strategist) ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS topics_used (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            topic      TEXT NOT NULL,
            angle_type TEXT DEFAULT '',
            used_at    TEXT NOT NULL
        )
    """)
    _add_column_safe(c, "topics_used", "angle_type", "TEXT DEFAULT ''")

    # --- embeddings (dedup semantik) ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            tweet_id   INTEGER NOT NULL REFERENCES tweets(id),
            vector     BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # --- runs (observability) ---
    c.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at  TEXT NOT NULL,
            finished_at TEXT,
            produced    INTEGER DEFAULT 0,
            errors      INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()

    _migrate_memory_db()
    logger.info("Database initialized: %s", DB_PATH)
    print("✅ Database initialized.")


def _add_column_safe(cursor, table: str, column: str, typedef: str):
    """Tambah kolom baru tanpa error jika sudah ada (SQLite migration helper)."""
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {typedef}")
    except sqlite3.OperationalError:
        pass  # kolom sudah ada


def _migrate_memory_db():
    """Pindahkan topics_used dari memory.db (lama) ke tweets.db jika ada."""
    memory_path = os.path.join(os.path.dirname(__file__), "memory.db")
    if not os.path.exists(memory_path):
        return
    try:
        old = sqlite3.connect(memory_path)
        rows = old.execute(
            "SELECT topic, used_at FROM topics_used ORDER BY used_at"
        ).fetchall()
        old.close()
        if not rows:
            return
        conn = _connect()
        conn.executemany(
            "INSERT OR IGNORE INTO topics_used (topic, used_at) VALUES (?, ?)",
            rows,
        )
        conn.commit()
        conn.close()
        logger.info("Migrasi memory.db: %d baris dipindahkan", len(rows))
    except Exception as e:
        logger.warning("Migrasi memory.db gagal (tidak masalah): %s", e)


# ------------------------------------------------------------------
# Fungsi backward-compat (dipanggil kode lama di scheduler & bot)
# ------------------------------------------------------------------

def save_tweet(content: str, topic: str, status: str = "pending") -> int:
    """Simpan tweet (API lama, tetap berfungsi)."""
    conn = _connect()
    c = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    c.execute(
        """INSERT INTO tweets (content, content_jp, topic, status, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (content, content, topic, status, now),
    )
    tweet_db_id = c.lastrowid
    conn.commit()
    conn.close()
    return tweet_db_id


def update_tweet_status(
    tweet_db_id: int, status: str, tweet_id: Optional[str] = None
):
    """Update status tweet (API lama, tetap berfungsi)."""
    conn = _connect()
    now = datetime.now(timezone.utc).isoformat()
    if status == "posted" and tweet_id:
        conn.execute(
            "UPDATE tweets SET status=?, tweet_id=?, posted_at=?, sent_at=? WHERE id=?",
            (status, tweet_id, now, now, tweet_db_id),
        )
    else:
        conn.execute("UPDATE tweets SET status=? WHERE id=?", (status, tweet_db_id))
    conn.commit()
    conn.close()


def get_tweet_by_id(tweet_db_id: int) -> Optional[dict]:
    """Ambil tweet berdasar ID, kembalikan dict (API lama, tetap berfungsi)."""
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT * FROM tweets WHERE id=?", (tweet_db_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    cols = [d[0] for d in c.description]
    conn.close()
    result = dict(zip(cols, row))
    # Pastikan field 'content' selalu ada (kode lama bergantung padanya)
    result.setdefault("content", result.get("content_jp", ""))
    return result


def save_topic_used(topic: str, angle_type: str = ""):
    """Simpan topik yang sudah dipakai ke memory (API lama + extended)."""
    conn = _connect()
    conn.execute(
        "INSERT INTO topics_used (topic, angle_type, used_at) VALUES (?, ?, ?)",
        (topic, angle_type, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_recent_topics(limit: int = 20) -> List[str]:
    """Ambil topik yang baru-baru ini dipakai (API lama, tetap berfungsi)."""
    conn = _connect()
    rows = conn.execute(
        "SELECT topic FROM topics_used ORDER BY used_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


# ------------------------------------------------------------------
# Fungsi baru — topics
# ------------------------------------------------------------------

def save_topic(topic: str, source: str = "", url: str = "") -> int:
    """Simpan kandidat topik dari TrendScout. Kembalikan topic_id."""
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "INSERT INTO topics (topic, source, url, seen_at) VALUES (?, ?, ?, ?)",
        (topic, source, url, datetime.now(timezone.utc).isoformat()),
    )
    topic_id = c.lastrowid
    conn.commit()
    conn.close()
    return topic_id


def mark_topic_used(topic_id: int):
    """Tandai topik sudah digunakan (hindari reuse)."""
    conn = _connect()
    conn.execute("UPDATE topics SET used=1 WHERE id=?", (topic_id,))
    conn.commit()
    conn.close()


# ------------------------------------------------------------------
# Fungsi baru — tweets bilingual (dipakai orchestrator baru)
# ------------------------------------------------------------------

def save_tweet_full(
    topic: str,
    content_jp: str,
    content_indo: str,
    draft_jp: str,
    score: int,
    score_breakdown: Dict,
    angle_type: str,
    topic_id: Optional[int] = None,
    status: str = "pending",
) -> int:
    """
    Simpan konten bilingual dengan semua metadata.
    Kembalikan tweet_id untuk penyimpanan embedding.
    """
    conn = _connect()
    c = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    c.execute(
        """INSERT INTO tweets
           (topic_id, topic, content, content_jp, content_indo, draft_jp,
            score, score_breakdown, angle_type, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            topic_id, topic,
            content_jp,       # backward compat alias
            content_jp, content_indo, draft_jp,
            score,
            json.dumps(score_breakdown, ensure_ascii=False),
            angle_type, status, now,
        ),
    )
    tweet_id = c.lastrowid
    conn.commit()
    conn.close()
    return tweet_id


def update_tweet_sent(tweet_id: int):
    """Tandai tweet sudah terkirim ke Telegram."""
    conn = _connect()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE tweets SET status='sent', sent_at=?, posted_at=? WHERE id=?",
        (now, now, tweet_id),
    )
    conn.commit()
    conn.close()


# ------------------------------------------------------------------
# Fungsi baru — embeddings
# ------------------------------------------------------------------

def save_embedding(tweet_id: int, vector_bytes: bytes):
    """Simpan vektor embedding (bytes) ke tabel embeddings."""
    conn = _connect()
    conn.execute(
        "INSERT INTO embeddings (tweet_id, vector, created_at) VALUES (?, ?, ?)",
        (tweet_id, vector_bytes, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_recent_embeddings(days: int = 30) -> List[Tuple[int, bytes]]:
    """
    Ambil semua (tweet_id, vector BLOB) dari tweet yang dibuat
    dalam N hari terakhir. Dipakai oleh dedup semantik.
    """
    conn = _connect()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """SELECT e.tweet_id, e.vector
           FROM embeddings e
           JOIN tweets t ON e.tweet_id = t.id
           WHERE t.created_at >= ?""",
        (cutoff,),
    ).fetchall()
    conn.close()
    return rows


# ------------------------------------------------------------------
# Fungsi baru — angle memory (untuk Strategist memvariasikan angle)
# ------------------------------------------------------------------

def get_recent_angle_types(limit: int = 10) -> List[str]:
    """
    Kembalikan list angle_type yang terakhir dipakai.
    Dipakai Strategist agar tidak memilih angle yang sama berturut-turut.
    """
    conn = _connect()
    rows = conn.execute(
        """SELECT angle_type FROM topics_used
           WHERE angle_type != ''
           ORDER BY used_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


# ------------------------------------------------------------------
# Fungsi baru — runs (observability)
# ------------------------------------------------------------------

def start_run() -> int:
    """Catat awal run baru ke tabel runs. Kembalikan run_id."""
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "INSERT INTO runs (started_at) VALUES (?)",
        (datetime.now(timezone.utc).isoformat(),),
    )
    run_id = c.lastrowid
    conn.commit()
    conn.close()
    return run_id


def finish_run(run_id: int, produced: int = 0, errors: int = 0):
    """Update run dengan waktu selesai dan statistik produksi."""
    conn = _connect()
    conn.execute(
        "UPDATE runs SET finished_at=?, produced=?, errors=? WHERE id=?",
        (datetime.now(timezone.utc).isoformat(), produced, errors, run_id),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
