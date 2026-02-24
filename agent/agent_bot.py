#!/usr/bin/env python3
"""
Telegram-бот с функционалом агента: общий запрос, курс крипты, погода, QR-код, калькулятор.
Использует реализацию из директории agent/.
"""

import asyncio
import logging
import os
import re
from pathlib import Path

from aiogram import Bot, Dispatcher, Router, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    FSInputFile,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from dotenv import load_dotenv

# Загрузка .env из корня проекта
load_dotenv(Path(__file__).resolve().parent / ".env")

# Импорт агента и инструментов (путь из корня проекта)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from agent.agent import run_agent
    from agent.tools import (
        get_weather,
        get_crypto_price,
        generate_qr_code,
        advanced_calculator,
        QR_OUTPUT_DIR,
    )
except ImportError:
    from agent import run_agent
    from agent.tools import (
        get_weather,
        get_crypto_price,
        generate_qr_code,
        advanced_calculator,
        QR_OUTPUT_DIR,
    )

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agent_bot")

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("В .env задайте BOT_TOKEN")

# Состояние пользователя: None или {"mode": "crypto"|"weather"|"qr"|"calculator", "expression": "..."}
user_state: dict[int, dict] = {}
# История диалога для run_agent: user_id -> list[{"role", "content"}]
user_chat_history: dict[int, list[dict]] = {}

router = Router()

# --- Описание при старте ---
START_TEXT = """<b>Я бот-агент</b> с доступом к инструментам. Выберите действие клавишей ниже или напишите произвольный запрос.

<b>Горячие клавиши:</b>

1️⃣ <b>Общий запрос</b> — произвольный вопрос или задача (поиск, расчёты, вопросы и т.д.).

2️⃣ <b>Курс криптовалюты</b> — введите название монеты (bitcoin, ethereum и т.п.), получу актуальный курс.

3️⃣ <b>Погода по городу</b> — введите город на русском, верну погоду.

4️⃣ <b>Генерация QR-кода</b> — введите ссылку или текст, отправлю готовый QR-код файлом.

5️⃣ <b>Калькулятор</b> — откроется клавиатура для ввода выражения (числа, +, -, *, /, sqrt, sin, cos и т.д.)."""


def get_main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="1️⃣ Общий запрос")],
            [
                KeyboardButton(text="2️⃣ Курс криптовалюты"),
                KeyboardButton(text="3️⃣ Погода по городу"),
            ],
            [
                KeyboardButton(text="4️⃣ Генерация QR кода"),
                KeyboardButton(text="5️⃣ Калькулятор"),
            ],
        ],
        resize_keyboard=True,
        input_field_placeholder="Выберите действие или введите запрос",
    )


def get_calculator_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="7"),
                KeyboardButton(text="8"),
                KeyboardButton(text="9"),
                KeyboardButton(text="/"),
                KeyboardButton(text="("),
                KeyboardButton(text=")"),
            ],
            [
                KeyboardButton(text="4"),
                KeyboardButton(text="5"),
                KeyboardButton(text="6"),
                KeyboardButton(text="*"),
                KeyboardButton(text="^"),
                KeyboardButton(text="sqrt("),
            ],
            [
                KeyboardButton(text="1"),
                KeyboardButton(text="2"),
                KeyboardButton(text="3"),
                KeyboardButton(text="-"),
                KeyboardButton(text="sin("),
                KeyboardButton(text="cos("),
            ],
            [
                KeyboardButton(text="0"),
                KeyboardButton(text="."),
                KeyboardButton(text="="),
                KeyboardButton(text="+"),
                KeyboardButton(text="tan("),
                KeyboardButton(text="log("),
            ],
            [
                KeyboardButton(text="ln("),
                KeyboardButton(text="pi"),
                KeyboardButton(text="e"),
            ],
            [
                KeyboardButton(text="C"),
                KeyboardButton(text="Назад"),
            ],
        ],
        resize_keyboard=True,
    )


def clear_state(user_id: int) -> None:
    user_state.pop(user_id, None)


def set_state(user_id: int, mode: str, expression: str = "") -> None:
    user_state[user_id] = {"mode": mode, "expression": expression}


def get_state(user_id: int) -> dict | None:
    return user_state.get(user_id)


# --- Вызов синхронных функций агента в executor ---


async def run_agent_async(user_input: str, chat_history: list | None) -> str:
    return await asyncio.to_thread(run_agent, user_input, chat_history)


async def tool_weather(city: str) -> str:
    return await asyncio.to_thread(get_weather.invoke, {"city": city})


async def tool_crypto(coin: str, currency: str = "usd") -> str:
    return await asyncio.to_thread(get_crypto_price.invoke, {"coin": coin, "currency": currency})


async def tool_qr(data: str) -> str:
    return await asyncio.to_thread(generate_qr_code.invoke, {"data": data})


async def tool_calculator(expression: str) -> str:
    return await asyncio.to_thread(advanced_calculator.invoke, {"expression": expression})


def extract_qr_path(tool_result: str) -> str | None:
    """Из ответа инструмента вида 'QR-код создан. Путь к файлу: /path/to/file.png' извлекает path."""
    m = re.search(r"Путь к файлу:\s*(.+)", tool_result)
    if m:
        return m.group(1).strip()
    if tool_result.strip().endswith(".png") and os.path.isfile(tool_result.strip()):
        return tool_result.strip()
    return None


# --- Обработчики ---


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    clear_state(message.from_user.id if message.from_user else 0)
    await message.answer(
        START_TEXT,
        parse_mode=ParseMode.HTML,
        reply_markup=get_main_keyboard(),
    )


@router.message(F.text == "1️⃣ Общий запрос")
async def btn_general(message: Message) -> None:
    uid = message.from_user.id if message.from_user else 0
    clear_state(uid)
    await message.answer(
        "Напишите ваш запрос (вопрос, задачу, поиск и т.д.) — я обработаю его как общий запрос.",
        reply_markup=get_main_keyboard(),
    )


@router.message(F.text == "2️⃣ Курс криптовалюты")
async def btn_crypto(message: Message) -> None:
    uid = message.from_user.id if message.from_user else 0
    set_state(uid, "crypto")
    await message.answer(
        "Введите название криптовалюты (например: bitcoin, ethereum):",
        reply_markup=get_main_keyboard(),
    )


@router.message(F.text == "3️⃣ Погода по городу")
async def btn_weather(message: Message) -> None:
    uid = message.from_user.id if message.from_user else 0
    set_state(uid, "weather")
    await message.answer(
        "Введите город на русском языке (например: Москва, Санкт-Петербург):",
        reply_markup=get_main_keyboard(),
    )


@router.message(F.text == "4️⃣ Генерация QR кода")
async def btn_qr(message: Message) -> None:
    uid = message.from_user.id if message.from_user else 0
    set_state(uid, "qr")
    await message.answer(
        "Введите текст или ссылку для QR-кода (например: https://example.com или Wi-Fi пароль):",
        reply_markup=get_main_keyboard(),
    )


@router.message(F.text == "5️⃣ Калькулятор")
async def btn_calculator(message: Message) -> None:
    uid = message.from_user.id if message.from_user else 0
    set_state(uid, "calculator", "")
    await message.answer(
        "Режим калькулятора. Вводите выражение кнопками или текстом. «=» — вычислить, «C» — очистить, «Назад» — выйти.",
        reply_markup=get_calculator_keyboard(),
    )


@router.message(F.text == "Назад")
async def btn_calculator_back(message: Message) -> None:
    uid = message.from_user.id if message.from_user else 0
    st = get_state(uid)
    if st and st.get("mode") == "calculator":
        clear_state(uid)
        await message.answer("Калькулятор закрыт.", reply_markup=get_main_keyboard())
    else:
        await message.answer("Выберите действие.", reply_markup=get_main_keyboard())


@router.message(F.text)
async def on_text(message: Message) -> None:
    uid = message.from_user.id if message.from_user else 0
    text = (message.text or "").strip()
    if not text:
        return

    st = get_state(uid)

    # Режим калькулятора
    if st and st.get("mode") == "calculator":
        if text == "=":
            expr = st.get("expression", "").strip()
            if not expr:
                await message.answer("Выражение пусто. Введите что-нибудь перед «=».")
                return
            clear_state(uid)
            await message.answer("Считаю…", reply_markup=get_main_keyboard())
            try:
                result = await tool_calculator(expr)
                await message.answer(result, reply_markup=get_main_keyboard())
            except Exception as e:
                logger.exception("Calculator error")
                await message.answer(f"Ошибка: {e}", reply_markup=get_main_keyboard())
            return
        if text == "C":
            set_state(uid, "calculator", "")
            await message.answer("Очищено. Введите выражение заново.", reply_markup=get_calculator_keyboard())
            return
        # Добавить символ к выражению
        new_expr = st.get("expression", "") + text
        set_state(uid, "calculator", new_expr)
        await message.answer(f"Выражение: <code>{new_expr}</code>\nНажмите «=» для вычисления.", reply_markup=get_calculator_keyboard(), parse_mode=ParseMode.HTML)
        return

    # Режим «ожидаю крипту»
    if st and st.get("mode") == "crypto":
        clear_state(uid)
        await message.answer("Проверяю курс…")
        try:
            result = await tool_crypto(text)
            await message.answer(result, reply_markup=get_main_keyboard())
        except Exception as e:
            logger.exception("Crypto error")
            await message.answer(f"Ошибка: {e}", reply_markup=get_main_keyboard())
        return

    # Режим «ожидаю город»
    if st and st.get("mode") == "weather":
        clear_state(uid)
        await message.answer("Запрашиваю погоду…")
        try:
            result = await tool_weather(text)
            await message.answer(result, reply_markup=get_main_keyboard())
        except Exception as e:
            logger.exception("Weather error")
            await message.answer(f"Ошибка: {e}", reply_markup=get_main_keyboard())
        return

    # Режим «ожидаю данные для QR»
    if st and st.get("mode") == "qr":
        clear_state(uid)
        await message.answer("Генерирую QR-код…")
        try:
            result = await tool_qr(text)
            path = extract_qr_path(result)
            if path and os.path.isfile(path):
                await message.answer_document(
                    FSInputFile(path),
                    caption="QR-код по вашему запросу.",
                    reply_markup=get_main_keyboard(),
                )
            else:
                await message.answer(result, reply_markup=get_main_keyboard())
        except Exception as e:
            logger.exception("QR error")
            await message.answer(f"Ошибка: {e}", reply_markup=get_main_keyboard())
        return

    # Общий запрос (произвольный текст) — передаём агенту
    history = user_chat_history.get(uid, [])
    await message.answer("Обрабатываю запрос…")
    try:
        reply = await run_agent_async(text, history)
        await message.answer(reply or "Готово.", reply_markup=get_main_keyboard())
        # Обновляем историю для контекста
        history = (history + [{"role": "user", "content": text}, {"role": "assistant", "content": reply or ""}])[-20:]
        user_chat_history[uid] = history
    except Exception as e:
        logger.exception("Agent error")
        await message.answer(f"Ошибка: {e}", reply_markup=get_main_keyboard())


async def main() -> None:
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
