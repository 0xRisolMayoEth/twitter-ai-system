"""
main.py — Entry point sistem Twitter AI multi-agent.

Cara jalan:
    python main.py                  # mode penuh: bot + scheduler (lama)
    python main.py --once           # jalankan pipeline lama satu kali
    python main.py --orchestrator   # jalankan orchestrator baru satu siklus (dev/test)
    python main.py --bot            # jalankan Telegram bot saja
"""
import sys
import os
from dotenv import load_dotenv

load_dotenv()


def _check_env() -> bool:
    """Pastikan env vars wajib sudah diisi. Mendukung nama baru (LLM_*) dan lama (OPENAI_*)."""
    checks = {
        "LLM_API_KEY / OPENAI_API_KEY": (
            os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        ),
        "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN"),
        "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID"),
    }
    missing = [k for k, v in checks.items() if not v]
    if missing:
        print(f"❌ Env vars belum diisi: {', '.join(missing)}")
        print("   Lihat .env.example untuk panduan.")
        return False
    return True


def main():
    # Setup logging sebelum apapun
    from core.logger import setup_logger
    logger = setup_logger()
    logger.info("=== Twitter AI System starting ===")

    from database.db_manager import init_db
    init_db()

    args = sys.argv[1:]

    # --- Mode: jalankan orchestrator baru satu siklus (untuk dev/test) ---
    if "--orchestrator" in args:
        if not _check_env():
            sys.exit(1)
        from orchestrator import Orchestrator
        orch = Orchestrator()
        result = orch.run_cycle()
        logger.info(f"Orchestrator selesai: {result}")
        return

    # --- Mode: jalankan pipeline lama satu kali ---
    if "--once" in args:
        if not _check_env():
            sys.exit(1)
        from scheduler.post_scheduler import run_pipeline
        run_pipeline()
        return

    # --- Mode: jalankan bot Telegram saja ---
    if "--bot" in args:
        from tg_bot.bot import run_bot
        run_bot()
        return

    # --- Mode default: bot + scheduler berjalan bersamaan ---
    if not _check_env():
        sys.exit(1)

    import threading
    from tg_bot.bot import run_bot
    from scheduler.post_scheduler import start_scheduler

    logger.info("Memulai Telegram bot + scheduler...")
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    start_scheduler()


if __name__ == "__main__":
    main()
