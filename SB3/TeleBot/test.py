import sys
from pathlib import Path

ROOT_DIR = Path.cwd().parent.parent
print(ROOT_DIR)
sys.path.append(f"{ROOT_DIR}")
from config import BOT_TOKEN

from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

# Replace 'YOUR_TOKEN_HERE' with your bot's API token
API_TOKEN = BOT_TOKEN


def start(update: Update, context: CallbackContext) -> None:
    """Send a welcome message when the /start command is issued."""
    update.message.reply_text('Hello! Welcome to our bot. How can I assist you today?')


def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    REQUEST_KWARGS = {
        # "USERNAME:PASSWORD@" is optional, if you need authentication:
        'proxy_url': 'http://127.0.0.1:10809/',
    }
    updater = Updater(API_TOKEN, use_context=True, request_kwargs=REQUEST_KWARGS)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Register the /start command handler
    dispatcher.add_handler(CommandHandler("start", start))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT, SIGTERM or SIGABRT
    updater.idle()


if __name__ == '__main__':
    main()
