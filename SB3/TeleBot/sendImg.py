# Importing required libraries
import sys
from pathlib import Path
import os

ROOT_DIR = Path.cwd().parent
print(ROOT_DIR)
sys.path.append(f"{ROOT_DIR}")
import utils
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, KeyboardButton
from aiogram.types import ReplyKeyboardMarkup

from config import BOT_TOKEN

# Put the token that you received from BotFather in the quotes
bot = Bot(token=BOT_TOKEN, proxy='http://127.0.0.1:10809/')

# Initializing the dispatcher object
dp = Dispatcher(bot)

# Creating the reply keyboard
keyboard_reply = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True).add("See all saved config")
BackKey = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True).add("Back")


# Handling the /start and /help commands
@dp.message_handler(commands=['start', 'help'])
async def welcome(message: types.Message):
    # Sending a greeting message that includes the reply keyboard
    await message.reply("Hello!", reply_markup=keyboard_reply)


# Handling all other messages
@dp.message_handler()
async def check_rp(message: types.Message):
    if message.text == 'See all saved config':
        # Responding with a message for the first button
        configNum, configs = utils.readConfigList(f"{ROOT_DIR}/Graphs/configList")
        print(configNum)
        buttons = []
        for i in range(configNum):
            buttons.append([KeyboardButton(str(f"Config ID: {i + 1}"), hide_keyboard=True)])
        selectConfigKey = ReplyKeyboardMarkup(buttons, one_time_keyboard=True, resize_keyboard=True, selective=True)
        await message.reply(configs, reply_markup=selectConfigKey)

    elif "Config ID" in message.text:
        await message.reply(f"Sending Photos of config ID {message.text[9:].strip()}...")
        configID = str(message.text[10:]).strip()
        print(configID)
        path = f'{ROOT_DIR}/Graphs/{configID}/'

        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                if '.png' in file:
                    files.append(os.path.join(r, file))
        for f in files:
            await bot.send_photo(chat_id=message.chat.id, photo=open(f, 'rb'))
        await message.reply("Back", reply_markup=BackKey)

    elif message.text == "Back":
        await message.reply("Configs", reply_markup=keyboard_reply)



executor.start_polling(dp)
