#!/usr/bin/env python3
"""
Бот, который отправляет случайные мемы из интернета или генерирует через OpenAI DALL-E.
"""

import asyncio
import io
import logging
import os
import random

import aiohttp
import openai
from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command, CommandStart
from aiogram.filters.base import Filter
from aiogram.types import BufferedInputFile, Message
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Лимит Telegram для send_photo: 5 MB
TELEGRAM_PHOTO_LIMIT = 5 * 1024 * 1024

router = Router()


def compress_image(content: bytes, max_size: int = TELEGRAM_PHOTO_LIMIT) -> bytes:
    """Сжимает изображение, если превышает лимит Telegram (5 MB)."""
    if len(content) <= max_size:
        return content
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(content))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        scale = (max_size * 0.9 / len(content)) ** 0.5
        new_size = (max(int(img.width * scale), 128), max(int(img.height * scale), 128))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        for quality in (80, 70, 60, 50):
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=quality, optimize=True)
            result = buf.getvalue()
            if len(result) <= max_size:
                return result
        return result
    except Exception as e:
        logger.warning("Compress failed (%s), sending original", e)
        return content


async def get_meme_from_internet() -> tuple[bytes, str]:
    """Загружает случайный мем с meme-api.com (Reddit)."""
    async with aiohttp.ClientSession() as session:
        async with session.get("https://meme-api.com/gimme") as resp:
            if resp.status != 200:
                raise RuntimeError("Не удалось загрузить мем из интернета")
            data = await resp.json()
            url = data.get("url")
            if not url:
                raise RuntimeError("Нет URL в ответе API")
            async with session.get(url) as img_resp:
                if img_resp.status != 200:
                    raise RuntimeError("Не удалось скачать изображение")
                content = await img_resp.read()
                title = data.get("title", "Мем")
                return content, title


async def get_meme_from_openai() -> tuple[bytes, str]:
    """Генерирует мем через OpenAI DALL-E."""
    prompt = random.choice([
        "Funny meme style image, absurd humor, simple cartoon",
        "Смешной мем в стиле интернет-мемов, минималистичный",
        "Absurd funny meme image, internet culture style",
    ])
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    response = await client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    url = response.data[0].url
    if not url:
        raise RuntimeError("OpenAI не вернул изображение")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError("Не удалось загрузить сгенерированное изображение")
            return await resp.read(), "Сгенерированный мем"


@router.message(CommandStart())
async def start_handler(message: Message) -> None:
    await message.answer(
        "Привет! Я бот, который присылает случайные мемы.\n"
        "Напиши /meme, чтобы получить мем!"
    )


@router.message(Command("meme"))
async def meme_handler(message: Message) -> None:
    try:
        content, _ = await get_meme_from_internet()
    except Exception as e:
        logger.warning("meme-api.com недоступен (%s), fallback на OpenAI", e)
        try:
            content, _ = await get_meme_from_openai()
        except Exception as e2:
            logger.error("OpenAI fallback failed: %s", e2)
            await message.answer(f"Ошибка: {e}")
            return

    content = compress_image(content)
    photo = BufferedInputFile(content, filename="meme.jpg")
    await message.answer_photo(photo=photo)


class UnknownCommandFilter(Filter):
    async def __call__(self, message: Message) -> bool:
        if message.text and message.text.startswith("/"):
            cmd = message.text.split()[0].lower()
            return cmd not in ("/start", "/meme")
        return False


@router.message(UnknownCommandFilter())
async def unknown_command_handler(message: Message) -> None:
    logger.info("Unknown command: %s", message.text)
    await message.answer("Извините, я не знаю такой команды. Напиши /meme для мема!")


async def main() -> None:
    if not BOT_TOKEN:
        raise ValueError("BOT_TOKEN не задан в .env")

    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    logger.info("Bot started. Polling...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
