#!/usr/bin/env python3
"""
Инструменты для AI-агента: погода, криптовалюты, поиск, HTTP, файлы, терминал, память,
генерация QR-кодов, калькулятор сложных выражений.
LangChain @tool для агентной логики.
"""

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

import requests
from langchain_core.tools import tool

logger = logging.getLogger("agent.tools")

# Директория для сохранения QR-кодов
QR_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "qr"

SAFE_TERMINAL_COMMANDS = frozenset({
    "ls", "pwd", "date", "whoami", "echo", "id", "uname",
    "cat", "head", "tail", "wc", "find", "which", "env",
    "python", "python3", "--version", "pip", "pip3", "list",
})


def _geocode_city(city_name: str) -> tuple[float, float] | None:
    """Преобразует название города (в т.ч. на русском) в координаты через Open-Meteo Geocoding API."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city_name, "count": 1, "language": "ru"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        if not results:
            return None
        loc = results[0]
        return float(loc["latitude"]), float(loc["longitude"])
    except Exception as e:
        logger.warning("Geocoding failed for %s: %s", city_name, e)
        return None


@tool
def get_weather(city: str) -> str:
    """
    Получить текущую погоду по названию города (на русском или английском).
    Возвращает температуру, ветер и условия.
    """
    coords = _geocode_city(city)
    if not coords:
        return f"Не удалось найти координаты для города: {city}"
    lat, lon = coords
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        cur = data.get("current") or {}
        temp = cur.get("temperature_2m")
        wind = cur.get("wind_speed_10m")
        code = cur.get("weather_code")
        desc = "ясно" if (code or 0) == 0 else f"код погоды {code}"
        return f"Погода в {city}: температура {temp}°C, ветер {wind} м/с, {desc}."
    except Exception as e:
        logger.exception("Weather API error")
        return f"Ошибка при получении погоды: {e}"


@tool
def get_crypto_price(coin: str, currency: str = "usd") -> str:
    """
    Узнать текущий курс криптовалюты (например bitcoin, ethereum).
    coin: id монеты (bitcoin, ethereum, ...), currency: usd, eur, rub и т.д.
    """
    coin_id = coin.lower().strip()
    curr = currency.lower().strip()
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": coin_id, "vs_currencies": curr}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if coin_id not in data:
            return f"Монета '{coin}' не найдена. Попробуйте: bitcoin, ethereum, etc."
        price = data[coin_id].get(curr)
        if price is None:
            return f"Валюта '{currency}' не поддерживается для {coin}."
        return f"{coin}: {price} {curr.upper()}"
    except Exception as e:
        logger.exception("CoinGecko API error")
        return f"Ошибка при получении курса: {e}"


@tool
def web_search(query: str, max_results: int = 8) -> str:
    """Поиск в интернете (DuckDuckGo). Обязательно используй для: топ-N, самые популярные, рейтинги, списки, актуальные данные, текущие тренды — никогда не отвечай на такие вопросы без вызова поиска."""
    try:
        from ddgs import DDGS
        results = list(DDGS().text(query, max_results=max_results) or [])
        if not results:
            return "Ничего не найдено по запросу."
        lines = []
        for i, r in enumerate(results[:max_results], 1):
            title = r.get("title", "")
            body = r.get("body", "")[:300]
            lines.append(f"{i}. {title}\n   {body}")
        return "\n\n".join(lines)
    except Exception as e:
        logger.exception("Web search error")
        return f"Ошибка поиска: {e}"


@tool
def http_request(method: str, url: str, body: str | None = None) -> str:
    """
    Выполнить HTTP-запрос (GET или POST). method: GET или POST, url: полный URL, body: опционально для POST (JSON строка).
    """
    method = method.upper()
    if method not in ("GET", "POST"):
        return "Поддерживаются только GET и POST."
    try:
        if method == "GET":
            r = requests.get(url, timeout=15)
        else:
            r = requests.post(
                url,
                data=body,
                json=json.loads(body) if body and body.strip().startswith("{") else None,
                timeout=15,
            )
        return f"Status: {r.status_code}\n{r.text[:2000]}"
    except Exception as e:
        logger.exception("HTTP request error")
        return f"Ошибка запроса: {e}"


@tool
def read_file(file_path: str) -> str:
    """Прочитать содержимое файла. file_path — путь к файлу (относительный или абсолютный)."""
    path = Path(file_path).expanduser().resolve()
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        return f"Файл не найден: {file_path}"
    except Exception as e:
        return f"Ошибка чтения: {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Записать текст в файл. file_path — путь, content — содержимое."""
    path = Path(file_path).expanduser().resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Записано в {path}"
    except Exception as e:
        return f"Ошибка записи: {e}"


def _is_safe_command(cmd: str) -> bool:
    cmd = cmd.strip()
    if not cmd or re.search(r"[;&|$`<>()]", cmd):
        return False
    parts = cmd.split()
    if not parts:
        return False
    base = parts[0].lower()
    return base in SAFE_TERMINAL_COMMANDS or base.endswith("python") or base.endswith("python3")


@tool
def run_terminal_command(command: str) -> str:
    """
    Выполнить безопасную терминальную команду (ls, pwd, date, cat, echo, python --version и т.п.).
    Опасные команды (rm, mv, chmod, >, | и т.д.) запрещены.
    """
    if not _is_safe_command(command):
        return "Команда не разрешена. Разрешены: ls, pwd, date, whoami, echo, cat, head, tail, wc, find, which, env, python --version, pip list."
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30, cwd=os.getcwd()
        )
        out, err = result.stdout or "", result.stderr or ""
        if result.returncode != 0:
            return f"Exit code: {result.returncode}\nstdout: {out}\nstderr: {err}"
        return out or "(пустой вывод)"
    except subprocess.TimeoutExpired:
        return "Команда прервана по таймауту."
    except Exception as e:
        return f"Ошибка: {e}"


def get_memory_path() -> Path:
    return Path(__file__).resolve().parent / "memory.json"


@tool
def load_memory() -> str:
    """Загрузить контекстную память агента (историю диалога) из memory.json."""
    path = get_memory_path()
    try:
        if not path.exists():
            return "Память пуста (файл ещё не создан)."
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            return "Память пуста."
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Ошибка загрузки памяти: {e}"


@tool
def save_memory(summary_or_entries: str) -> str:
    """
    Сохранить в память агента резюме диалога или новые записи.
    summary_or_entries: текст резюме или JSON-массив записей вида [{"role":"user","content":"..."}, ...].
    """
    path = get_memory_path()
    try:
        existing = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        s = summary_or_entries.strip()
        if s.startswith("["):
            try:
                new_entries = json.loads(s)
                if isinstance(new_entries, list):
                    existing.extend(new_entries)
            except json.JSONDecodeError:
                existing.append({"type": "summary", "content": s})
        else:
            existing.append({"type": "summary", "content": s})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        return "Память обновлена."
    except Exception as e:
        return f"Ошибка сохранения памяти: {e}"


@tool
def generate_qr_code(data: str, filename: str | None = None) -> str:
    """
    Генерация QR-кода по переданной строке (ссылке, тексту, визитке, произвольным данным) и сохранение в файл.
    Используй, когда пользователь просит: «Сгенерируй QR-код для ссылки», «Сделай QR-код с этим текстом», «Создай QR-код для Wi-Fi пароля» и т.п.
    data: строка для кодирования (URL, текст, данные). filename: необязательное имя файла (например my_qr.png); если не указано — создаётся qr_<timestamp>.png.
    Возвращает путь к созданному файлу — обязательно укажи его в ответе пользователю.
    """
    if not data or not str(data).strip():
        logger.warning("generate_qr_code: пустые данные")
        return "Ошибка: не переданы данные для QR-кода. Укажите текст или ссылку."
    try:
        import qrcode
        QR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if filename:
            name = filename.strip()
            if not name.lower().endswith(".png"):
                name += ".png"
        else:
            name = f"qr_{int(time.time())}.png"
        filepath = QR_OUTPUT_DIR / name
        qr = qrcode.make(str(data).strip())
        qr.save(str(filepath))
        logger.info("QR-код сохранён: %s", filepath)
        return f"QR-код создан. Путь к файлу: {filepath}"
    except Exception as e:
        logger.exception("Ошибка сохранения QR-кода")
        return f"Не удалось сохранить QR-код: {e}"


@tool
def advanced_calculator(expression: str) -> str:
    """
    Калькулятор сложных математических выражений. Поддерживает скобки, степени (^ или **), функции: sin, cos, tan, log, ln, sqrt, константы pi, e.
    Используй, когда пользователь просит: «Посчитай ...», «Реши выражение ...», «Сколько будет (2^5 + 3*sqrt(7)) / sin(0.5)» и т.п.
    expression: математическое выражение в виде строки (например: "2**5 + 3*sqrt(7)", "sin(0.5) + ln(2)").
    Возвращает исходное выражение и числовой результат (до 6 знаков после запятой).
    """
    if not expression or not str(expression).strip():
        logger.warning("advanced_calculator: пустое выражение")
        return "Ошибка: не передано выражение для вычисления."
    expr_str = str(expression).strip()
    try:
        from sympy import sympify, N, sin, cos, tan, log, sqrt, pi, E
        # Безопасный разбор: только математика, без eval на сырой строке
        s = expr_str.replace("^", "**").replace("ln(", "log(")
        expr = sympify(s)
        result = float(N(expr, 6))
        logger.info("advanced_calculator: %s = %s", expr_str, result)
        return f"Выражение: {expr_str}\nРезультат: {result}"
    except Exception as e:
        logger.exception("advanced_calculator error: %s", expr_str)
        return f"Ошибка в выражении «{expr_str}». Проверьте формулу (скобки, функции sin/cos/tan/log/ln/sqrt, степень ^ или **). Текст ошибки: {e}"


def get_all_tools():
    """Возвращает список всех инструментов для агента (LangChain tools)."""
    return [
        get_weather,
        get_crypto_price,
        web_search,
        http_request,
        read_file,
        write_file,
        run_terminal_command,
        load_memory,
        save_memory,
        generate_qr_code,
        advanced_calculator,
    ]
