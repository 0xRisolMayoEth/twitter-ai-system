import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "tweets.db")
MEMORY_PATH = os.path.join(os.path.dirname(__file__), "memory.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS tweets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            topic TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT,
            posted_at TEXT,
            tweet_id TEXT
        )
    """)
    conn.commit()
    conn.close()

    conn2 = sqlite3.connect(MEMORY_PATH)
    c2 = conn2.cursor()
    c2.execute("""
        CREATE TABLE IF NOT EXISTS topics_used (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            used_at TEXT
        )
    """)
    conn2.commit()
    conn2.close()
    print("✅ Database initialized.")

def save_tweet(content, topic, status="pending"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO tweets (content, topic, status, created_at) VALUES (?, ?, ?, ?)",
              (content, topic, status, datetime.now().isoformat()))
    tweet_db_id = c.lastrowid
    conn.commit()
    conn.close()
    return tweet_db_id

def update_tweet_status(tweet_db_id, status, tweet_id=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if status == "posted" and tweet_id:
        c.execute("UPDATE tweets SET status=?, tweet_id=?, posted_at=? WHERE id=?",
                  (status, tweet_id, datetime.now().isoformat(), tweet_db_id))
    else:
        c.execute("UPDATE tweets SET status=? WHERE id=?", (status, tweet_db_id))
    conn.commit()
    conn.close()

def get_tweet_by_id(tweet_db_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM tweets WHERE id=?", (tweet_db_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "content": row[1], "topic": row[2],
                "status": row[3], "created_at": row[4], "posted_at": row[5], "tweet_id": row[6]}
    return None

def save_topic_used(topic):
    conn = sqlite3.connect(MEMORY_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO topics_used (topic, used_at) VALUES (?, ?)",
              (topic, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_recent_topics(limit=20):
    conn = sqlite3.connect(MEMORY_PATH)
    c = conn.cursor()
    c.execute("SELECT topic FROM topics_used ORDER BY used_at DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]

if __name__ == "__main__":
    init_db()
