import sys, os
from dotenv import load_dotenv
load_dotenv()

def check_env():
    required = ["OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"❌ Missing: {', '.join(missing)}")
        return False
    return True

def main():
    from database.db_manager import init_db
    init_db()
    args = sys.argv[1:]

    if "--once" in args:
        from scheduler.post_scheduler import run_pipeline
        run_pipeline()
        return

    if "--bot" in args:
        from tg_bot.bot import run_bot
        run_bot()
        return

    if not check_env():
        sys.exit(1)

    import threading
    from tg_bot.bot import run_bot
    from scheduler.post_scheduler import start_scheduler
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    start_scheduler()

if __name__ == "__main__":
    main()
