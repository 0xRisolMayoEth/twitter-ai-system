import sys, os, random
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from database.db_manager import get_recent_topics
import feedparser

RSS_FEEDS = [
    "https://www.ign.com/rss/articles",
    "https://kotaku.com/rss",
    "https://www.animenewsnetwork.com/all/rss.xml",
]

FALLBACK_TOPICS = [
    {"title": "SNES vs Sega Genesis debat nostalgia", "category": "gaming"},
    {"title": "Anime isekai terbaik sepanjang masa", "category": "anime"},
    {"title": "VTuber Hololive highlight minggu ini", "category": "vtuber"},
    {"title": "Retro game yang wajib dimainkan ulang", "category": "gaming"},
    {"title": "Manga yang lebih bagus dari animenya", "category": "anime"},
    {"title": "Hidden gem indie game yang jarang dibahas", "category": "gaming"},
    {"title": "Opening anime paling ikonik sepanjang masa", "category": "anime"},
    {"title": "Gaming accessories budget terbaik 2024", "category": "gaming"},
    {"title": "VTuber debut yang paling ditunggu-tunggu", "category": "vtuber"},
    {"title": "RPG klasik PS1 yang harus dicoba lagi", "category": "gaming"},
]

def get_rss_trends():
    topics = []
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:3]:
                title = entry.get("title", "")
                if title:
                    topics.append({"title": title, "source": feed_url, "category": "mixed"})
        except Exception as e:
            print(f"⚠️ RSS error: {e}")
    return topics

def get_trending_topics(avoid_recent=True):
    recent = get_recent_topics(20) if avoid_recent else []
    topics = get_rss_trends()

    if not topics:
        print("⚠️ RSS kosong, pakai fallback topics...")
        topics = FALLBACK_TOPICS.copy()

    if avoid_recent and recent:
        filtered = [t for t in topics if not any(r.lower() in t["title"].lower() for r in recent)]
        if filtered:
            topics = filtered

    random.shuffle(topics)
    return topics

if __name__ == "__main__":
    results = get_trending_topics(avoid_recent=False)
    for i, t in enumerate(results[:5], 1):
        print(f"{i}. {t['title']}")
