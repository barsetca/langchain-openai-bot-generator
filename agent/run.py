#!/usr/bin/env python3
"""
CLI-интерфейс запуска AI-агента. Диалоговый режим с сохранением контекста в memory.json.
"""

import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Загрузка .env из корня проекта
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Логи только в docs/agent.log, не в консоль диалога
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "docs" / "agent.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
MEMORY_FILE = Path(__file__).resolve().parent / "memory.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
logger = logging.getLogger("agent.run")


def load_chat_history() -> list:
    """Загружает историю диалога из memory.json для контекста агента (список dict с role, content)."""
    if not MEMORY_FILE.exists():
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Не удалось загрузить memory.json: %s", e)
        return []
    messages = []
    for item in data:
        if isinstance(item, dict) and item.get("type") == "summary":
            continue
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content") or item.get("text", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    return messages[-20:]


def append_to_memory(user_text: str, assistant_text: str) -> None:
    """Добавляет пару user/assistant в memory.json."""
    existing = []
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append({"role": "user", "content": user_text})
    existing.append({"role": "assistant", "content": assistant_text})
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    logger.info("Память обновлена (%d записей)", len(existing))


def main():
    try:
        from agent import run_agent  # когда запуск из каталога agent/
    except ImportError as e:
        if "langchain" in str(e):
            print("Установите зависимости агента: pip install -r requirements.txt")
            sys.exit(1)
        try:
            from agent.agent import run_agent  # когда запуск как python -m agent.run из корня
        except ImportError:
            raise e

    logger.info("Запуск агента. Введите запрос (пустая строка или Ctrl+D — выход).")
    print("Возможности: погода, курс крипты, поиск, QR-коды, вычисления. Примеры: «Сгенерируй QR для https://example.com», «Посчитай (2^5 + sqrt(7))».")
    chat_history = load_chat_history()

    while True:
        try:
            print("\nВы: ", end="", flush=True)
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            print("\nВыход.")
            break
        if not line:
            break
        user_input = line.strip()
        if not user_input:
            continue

        logger.info("Запрос пользователя: %s", user_input[:200])
        try:
            reply = run_agent(user_input, chat_history=chat_history)
        except Exception as e:
            logger.exception("Ошибка агента")
            reply = f"Произошла ошибка: {e}"
        print("\nАгент:", reply)
        logger.info("Ответ агента: %s", reply[:200] if reply else "(пусто)")

        append_to_memory(user_input, reply)
        chat_history = (chat_history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": reply},
        ])[-20:]

    logger.info("Агент остановлен.")


if __name__ == "__main__":
    main()
