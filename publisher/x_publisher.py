import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
from database.db_manager import update_tweet_status
load_dotenv()

def post_tweet(content, tweet_db_id):
    api_key = os.getenv("X_API_KEY")
    if not api_key:
        print(f"⚠️ X API tidak ada. Tweet disimpan tapi tidak diposting.")
        print(f"📝 Konten: {content}")
        update_tweet_status(tweet_db_id, "approved_no_api")
        return False
    try:
        import tweepy
        client = tweepy.Client(
            consumer_key=os.getenv("X_API_KEY"),
            consumer_secret=os.getenv("X_API_SECRET"),
            access_token=os.getenv("X_ACCESS_TOKEN"),
            access_token_secret=os.getenv("X_ACCESS_SECRET")
        )
        response = client.create_tweet(text=content)
        tweet_id = response.data["id"]
        update_tweet_status(tweet_db_id, "posted", tweet_id=str(tweet_id))
        print(f"✅ Tweet posted! ID: {tweet_id}")
        return True
    except Exception as e:
        print(f"❌ Gagal posting: {e}")
        update_tweet_status(tweet_db_id, "failed")
        return False
