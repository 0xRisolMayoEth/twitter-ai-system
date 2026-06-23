import sys, os, asyncio
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, ContextTypes
from dotenv import load_dotenv
from database.db_manager import update_tweet_status, get_tweet_by_id
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

async def send_tweet_for_approval(tweet_db_id, tweet_content, topic):
    bot = Bot(token=BOT_TOKEN)
    message = (
        f"🐦 *Tweet Baru untuk Review*\n\n"
        f"📌 *Topik:* {topic}\n\n"
        f"📝 *Draft:*\n`{tweet_content}`\n\n"
        f"📊 Panjang: {len(tweet_content)}/280 karakter"
    )
    keyboard = [[
        InlineKeyboardButton("✅ Approve", callback_data=f"approve_{tweet_db_id}"),
        InlineKeyboardButton("❌ Reject", callback_data=f"reject_{tweet_db_id}"),
    ]]
    await bot.send_message(chat_id=CHAT_ID, text=message,
                           parse_mode="Markdown",
                           reply_markup=InlineKeyboardMarkup(keyboard))
    print(f"📨 Tweet #{tweet_db_id} dikirim ke Telegram!")

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action, tweet_db_id = query.data.split("_", 1)
    tweet_db_id = int(tweet_db_id)
    tweet = get_tweet_by_id(tweet_db_id)
    if action == "approve":
        update_tweet_status(tweet_db_id, "approved")
        from publisher.x_publisher import post_tweet
        post_tweet(tweet["content"], tweet_db_id)
        await query.edit_message_text(f"✅ *Tweet #{tweet_db_id} APPROVED & diposting!*", parse_mode="Markdown")
    elif action == "reject":
        update_tweet_status(tweet_db_id, "rejected")
        await query.edit_message_text(f"❌ *Tweet #{tweet_db_id} REJECTED*", parse_mode="Markdown")

def run_bot():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CallbackQueryHandler(handle_callback))
    print("🤖 Telegram bot berjalan...")
    app.run_polling()

async def notify_simple(message):
    bot = Bot(token=BOT_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")
