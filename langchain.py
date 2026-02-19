#!/usr/bin/env python3
"""
Скрипт генерации Telegram-ботов по текстовому ТЗ с использованием LangChain и OpenAI.
"""

import ast
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Загружаем .env до импорта цепочек
load_dotenv()

# Настройка логирования
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("bot_generator")


def create_openai_llm():
    """Создание LLM на базе openai библиотеки (без langchain-openai)."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    import openai

    class OpenAILLM(BaseChatModel):
        """Кастомный LLM на базе openai библиотеки."""

        model: str = "gpt-4o-mini"
        temperature: float = 0.7
        max_tokens: int = 4096

        def _convert_messages(self, messages: list[BaseMessage]) -> list[dict]:
            result = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    result.append({"role": "system", "content": msg.content})
                elif isinstance(msg, HumanMessage):
                    result.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    result.append({"role": "assistant", "content": msg.content})
            return result

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs,
        ) -> ChatResult:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY не задан в .env")
            client = openai.OpenAI(api_key=api_key)
            formatted = self._convert_messages(messages)
            response = client.chat.completions.create(
                model=self.model,
                messages=formatted,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
            )
            content = response.choices[0].message.content or ""
            return ChatResult(
                generations=[
                    ChatGeneration(message=AIMessage(content=content))
                ]
            )

        @property
        def _llm_type(self) -> str:
            return "openai_custom"

    return OpenAILLM(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
    )


def build_chains():
    """Сборка всех цепочек."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = create_openai_llm()

    # 1. analysis_chain
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты эксперт по проектированию Telegram-ботов.
Проанализируй техническое задание и верни структурированный анализ в формате:

КОМАНДЫ:
- список команд (например: /start, /help)

ОБРАБОТЧИКИ:
- список обработчиков и что они делают

ФОРМАТ_ОТВЕТА:
- текстовый/медиа/кнопки и т.д.

БАЗА_ДАННЫХ:
- ДА/НЕТ и для чего нужна (если ДА)

ДОПОЛНИТЕЛЬНЫЕ_ЗАВИСИМОСТИ:
- список pip-пакетов кроме aiogram, openai, python-dotenv"""),
        ("human", "{task}"),
    ])
    analysis_chain = analysis_prompt | llm | StrOutputParser()

    # 2. tools_chain
    tools_prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты эксперт по подбору инструментов для Python-проектов.
На основе анализа задания определи:

БИБЛИОТЕКИ:
- список Python-библиотек (pip install) для реализации. ОБЯЗАТЕЛЬНО включи: aiogram, openai, python-dotenv

ВНЕШНИЕ_ИНСТРУМЕНТЫ:
- API, сервисы, которые понадобятся

ТЕХНОЛОГИИ:
- какие технологии используем (очереди, кэш, БД и т.д.)

ВАЖНО: использовать именно библиотеку openai (не langchain-openai)."""),
        ("human", "Анализ: {analysis}\n\nЗадание: {task}"),
    ])
    tools_chain = tools_prompt | llm | StrOutputParser()

    # 3. structure_chain
    structure_prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты архитектор Python-кода.
На основе анализа и инструментов определи структуру кода Telegram-бота:

ФУНКЦИИ_И_МОДУЛИ:
- какие функции/хендлеры должны быть
- за что отвечает каждый элемент

ВЗАИМОДЕЙСТВИЕ:
- как элементы связаны между собой
- порядок инициализации

КОНФИГУРАЦИЯ:
- что читать из .env (BOT_TOKEN, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS)"""),
        ("human", "Анализ: {analysis}\nИнструменты: {tools}\nЗадание: {task}"),
    ])
    structure_chain = structure_prompt | llm | StrOutputParser()

    # 4. code_chain
    code_prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты опытный Python-разработчик. Сгенерируй полный рабочий код Telegram-бота.

ТРЕБОВАНИЯ:
1. aiogram 3.x (Bot, Dispatcher, Router, CommandStart, Message)
2. Все хендлеры - async def
3. Токен: os.getenv("BOT_TOKEN")
4. Использовать библиотеку openai напрямую (не langchain-openai)
5. Без заглушек - код должен запускаться
6. Добавить logging
7. Переменные из .env (те же для всех ботов): BOT_TOKEN, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS
8. Используй load_dotenv() и os.getenv() для чтения переменных

Верни только код Python, без markdown-оболочки (```python и т.д.)."""),
        ("human", """Структура: {structure}

Задание: {task}

Сгенерируй полный код бота."""),
    ])
    code_chain = code_prompt | llm | StrOutputParser()

    # 5. review_chain
    review_prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты код-ревьюер. Проверь Python-код Telegram-бота.

Проверь:
1. Нет ли синтаксических ошибок
2. Корректная структура запуска (if __name__ == "__main__", asyncio.run(main()))
3. Корректные импорты (aiogram 3.x, openai)
4. Используется BOT_TOKEN из getenv
5. Все хендлеры async def

Верни: ОК если всё корректно, или список ошибок для исправления."""),
        ("human", "{code}"),
    ])
    review_chain = review_prompt | llm | StrOutputParser()

    return {
        "analysis": analysis_chain,
        "tools": tools_chain,
        "structure": structure_chain,
        "code": code_chain,
        "review": review_chain,
    }


def extract_code(text: str) -> str:
    """Извлечение кода из ответа LLM (убирает markdown)."""
    text = text.strip()
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        return text[start:end].strip()
    return text


def validate_python_syntax(code: str) -> tuple[bool, str]:
    """Проверка синтаксиса Python."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def run_generation(filename: str, task: str) -> str:
    """Запуск полной генерации бота."""
    chains = build_chains()
    logger.info("Старт генерации: analysis_chain")
    analysis = chains["analysis"].invoke({"task": task})
    logger.info("analysis_chain завершён")

    logger.info("Старт: tools_chain")
    tools = chains["tools"].invoke({"analysis": analysis, "task": task})
    logger.info("tools_chain завершён")

    logger.info("Старт: structure_chain")
    structure = chains["structure"].invoke({
        "analysis": analysis,
        "tools": tools,
        "task": task,
    })
    logger.info("structure_chain завершён")

    logger.info("Старт: code_chain")
    code_raw = chains["code"].invoke({"structure": structure, "task": task})
    code = extract_code(code_raw)
    logger.info("code_chain завершён")

    logger.info("Старт: review_chain")
    review = chains["review"].invoke({"code": code})
    logger.info("review_chain завершён")

    valid, syntax_err = validate_python_syntax(code)
    if not valid:
        logger.warning("Синтаксическая ошибка: %s", syntax_err)
        # Пытаемся сохранить даже с ошибками для отладки
    if "ОК" not in review.upper() and "OK" not in review:
        logger.warning("Review рекомендует проверку: %s", review[:200])

    return code


def main():
    """Главная функция CLI."""
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temp = os.getenv("OPENAI_TEMPERATURE", "0.7")

    print("\n" + "=" * 60)
    print("ГЕНЕРАТОР TELEGRAM-БОТОВ (LangChain + OpenAI)")
    print("=" * 60)
    print(f"Модель: {model} | Температура: {temp}")
    print("=" * 60)

    filename = input("\nВведите имя файла для скрипта бота (без .py, например mem): ").strip()
    if not filename:
        filename = "bot"
    if not filename.endswith(".py"):
        filename = f"{filename}.py"

    task = input("\nВведите техническое описание бота (например: Бот, который отправляет случайные мемы): ").strip()
    if not task:
        print("Ошибка: описание не может быть пустым.")
        sys.exit(1)

    print("\nГенерация запущена...")
    try:
        code = run_generation(filename, task)
        output_path = Path(filename)
        output_path.write_text(code, encoding="utf-8")
        print(f"\nГотово! Бот сохранён в: {output_path.absolute()}")
        print(f"Запуск: python3 {filename}")
    except Exception as e:
        logger.exception("Ошибка генерации: %s", e)
        print(f"\nОшибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
