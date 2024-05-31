# Importing required libraries
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.types import ReplyKeyboardMarkup

from config import BOT_TOKEN

# Put the token that you received from BotFather in the quotes
bot = Bot(token=BOT_TOKEN)

# Initializing the dispatcher object
dp = Dispatcher(bot)

# Creating the reply keyboard
keyboard_reply = ReplyKeyboardMarkup(
    resize_keyboard=True, one_time_keyboard=True).add("_button1", "_button2")


# Handling the /start and /help commands
@dp.message_handler(commands=['start', 'help'])
async def welcome(message: types.Message):
    # Sending a greeting message that includes the reply keyboard
    await message.reply("Hello! how are you?", reply_markup=keyboard_reply)


# Handling all other messages
@dp.message_handler()
async def check_rp(message: types.Message):
    if message.text == '_button1':
        # Responding with a message for the first button
        await message.reply("Hi! this is first reply keyboards button.")

    elif message.text == '_button2':
        # Responding with a message for the second button
        await message.reply("Hi! this is second reply keyboards button.")

    else:
        # Responding with a message that includes the text of the user's message
        await message.reply(f"Your message is: {message.text}")

    # Starting the bot


executor.start_polling(dp)
