# Генератор Telegram-ботов (LangChain + OpenAI)

Скрипт автоматической генерации Telegram-ботов по текстовому техническому заданию с использованием LangChain и OpenAI.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
cp .env.example .env
# Заполните .env своими ключами (OPENAI_API_KEY, BOT_TOKEN)
```

## Запуск

```bash
python3 langchain.py
```

При запуске скрипт запросит:
1. Имя файла для скрипта бота (например: `mem` → `mem.py`)
2. Техническое описание бота (например: "Бот, который отправляет случайные мемы")

## Цепочка генерации (5 звеньев)

1. **analysis_chain** — анализ ТЗ: команды, обработчики, формат ответа, нужна ли БД, зависимости
2. **tools_chain** — подбор библиотек, API, технологий (openai без langchain-openai)
3. **structure_chain** — структура кода: функции, модули, взаимодействия
4. **code_chain** — генерация полного кода (aiogram 3.x, async, BOT_TOKEN из .env)
5. **review_chain** — проверка: синтаксис, структура, импорты

## Переменные окружения (.env)

| Переменная | Описание |
|------------|----------|
| OPENAI_API_KEY | Ключ API OpenAI |
| OPENAI_MODEL | Модель (gpt-4o, gpt-4o-mini и т.д.) |
| OPENAI_TEMPERATURE | Температура генерации |
| OPENAI_MAX_TOKENS | Максимум токенов в ответе |
| BOT_TOKEN | Токен Telegram-бота |
| LOG_LEVEL | Уровень логирования (DEBUG, INFO, WARNING, ERROR) |

## Технологии

- LangChain
- OpenAI (прямое использование библиотеки openai)
- aiogram 3.x
