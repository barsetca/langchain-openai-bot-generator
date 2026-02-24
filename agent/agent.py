#!/usr/bin/env python3
"""
Логика AI-агента: LangChain для агентной логики и инструментария, OpenAI Chat Completions через langchain-openai.
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

try:
    from .tools import get_all_tools
except ImportError:
    from tools import get_all_tools

# Загрузка .env из корня проекта
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger("agent")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "3000"))


SYSTEM_PROMPT = """Ты — полезный AI-агент, работающий в терминале. Ты понимаешь задачи на естественном языке и выбираешь подходящие инструменты.

Доступные инструменты:
- get_weather(city) — погода по названию города (на русском или английском)
- get_crypto_price(coin, currency) — курс криптовалюты (bitcoin, ethereum; usd, eur, rub)
- web_search(query) — поиск в интернете (DuckDuckGo)
- http_request(method, url, body?) — HTTP GET/POST
- read_file(file_path) — прочитать файл
- write_file(file_path, content) — записать в файл
- run_terminal_command(command) — безопасные команды (ls, pwd, date, cat, echo, python --version и т.п.)
- load_memory() — загрузить контекстную память (историю диалога)
- save_memory(summary_or_entries) — сохранить резюме или записи в память
- generate_qr_code(data, filename?) — генерация QR-кода по строке (ссылка, текст, Wi-Fi и т.д.); в ответе обязательно укажи путь к созданному файлу
- advanced_calculator(expression) — вычисление сложных математических выражений (скобки, sin, cos, tan, log, ln, sqrt, степени); выводи выражение и результат

Правила:
1) Выбор инструмента: погода — get_weather; курс одной криптовалюты — get_crypto_price; рейтинги/топ-N/актуальные данные — web_search; просьба сгенерировать QR-код (для ссылки, текста, Wi-Fi) — generate_qr_code; просьба посчитать выражение/формулу — advanced_calculator.
2) Для запросов про «топ», «самые популярные», рейтинги — всегда сначала web_search, не отвечай из своих знаний.
3) Если не хватает контекста — спроси уточнение у пользователя.
4) Отвечай кратко и по делу. После использования инструментов — дай понятный ответ; для QR-кода явно укажи путь к файлу; для калькулятора — выражение и результат.
5) При необходимости сохраняй резюме диалога через save_memory.
"""


def create_agent_llm():
    """Создаёт LLM с привязкой инструментов (LangChain)."""
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    tools = get_all_tools()
    return llm.bind_tools(tools), tools


def run_agent(user_input: str, chat_history: list | None = None) -> str:
    """
    Выполняет запрос пользователя: LLM + инструменты через LangChain,
    возвращает итоговый ответ.
    """
    llm_with_tools, tools = create_agent_llm()
    tool_map = {t.name: t for t in tools}
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if chat_history:
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user" and content:
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_input))

    max_steps = 15
    step = 0
    config = RunnableConfig(callbacks=[])

    while step < max_steps:
        step += 1
        response = llm_with_tools.invoke(messages, config=config)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            content = getattr(response, "content", "") or ""
            if isinstance(content, list):
                content = "".join(
                    (c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
                )
            return content.strip() or "Готово."

        for tc in tool_calls:
            name = tc.get("name") or ""
            raw_args = tc.get("args")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args = {}
            else:
                args = raw_args or {}
            tool = tool_map.get(name)
            if not tool:
                observation = f"Неизвестный инструмент: {name}"
            else:
                try:
                    observation = tool.invoke(args, config=config)
                except Exception as e:
                    logger.exception("Tool %s error", name)
                    observation = f"Ошибка: {e}"
            messages.append(
                ToolMessage(content=str(observation), tool_call_id=tc.get("id", str(step)))
            )

    return "Превышено максимальное число шагов. Попробуйте упростить запрос."
