import sys, os, asyncio, random
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import schedule, time
from dotenv import load_dotenv
load_dotenv()

INTERVAL_HOURS = int(os.getenv("POST_INTERVAL_HOURS", 2))

def run_pipeline():
    print("\n🚀 Menjalankan pipeline tweet...")
    try:
        from agents.trend import get_trending_topics
        topics = get_trending_topics()
        if not topics:
            print("⚠️ Tidak ada topik.")
            return
        selected = random.choice(topics[:3])
        topic_title = selected["title"]
        print(f"📌 Topik: {topic_title}")

        from agents.writer import write_tweet
        draft = write_tweet(topic_title, f"perspektif gamer/otaku tentang {topic_title}", ["#Gaming", "#Anime"])

        from agents.critic import review_tweet
        review = review_tweet(draft)
        final = review.get("tweet_revisi", draft)
        if not review.get("layak_post", True):
            print("❌ Ditolak critic.")
            return

        from agents.persona import apply_persona
        final = apply_persona(final)

        from database.db_manager import save_tweet, save_topic_used
        tweet_db_id = save_tweet(final, topic_title)
        save_topic_used(topic_title)

        asyncio.run(kirim_telegram(tweet_db_id, final, topic_title))
        print(f"✅ Selesai! Tweet #{tweet_db_id} menunggu approval di Telegram.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback; traceback.print_exc()

async def kirim_telegram(tweet_db_id, content, topic):
    from tg_bot.bot import send_tweet_for_approval
    await send_tweet_for_approval(tweet_db_id, content, topic)

def start_scheduler():
    print(f"⏰ Scheduler aktif - setiap {INTERVAL_HOURS} jam")
    run_pipeline()
    schedule.every(INTERVAL_HOURS).hours.do(run_pipeline)
    while True:
        schedule.run_pending()
        time.sleep(60)
