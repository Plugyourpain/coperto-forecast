"""Демонстрация работы приложения шаг за шагом.

Один запуск показывает весь пайплайн инференса:
данные → признаки → модель → прогноз → метрики на holdout.

Запуск:
    python demo.py
    python demo.py --restaurant 2 --date 2025-10-15
    python demo.py --pause              # пауза между шагами (для созвона)

Не является частью продакшн-кода. Продакшн — в src/* + predict.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import prepare
from src.features import DEFAULT_HORIZON, feature_columns, make_features
from src.model import MODEL_PATH, load, predict_for_dates


# ── Оформление вывода ─────────────────────────────────────────────────────

WIDTH = 72

def header(step: str, title: str) -> None:
    """Печатает заголовок шага."""
    print()
    print("=" * WIDTH)
    print(f"  ШАГ {step}. {title}")
    print("=" * WIDTH)


def note(text: str) -> None:
    """Печатает пояснение под шагом."""
    print(f"  → {text}")


def section(text: str) -> None:
    print()
    print(f"── {text} " + "─" * (WIDTH - len(text) - 4))


def wait(pause: bool, sec: float = 1.5) -> None:
    """Пауза между шагами, если включён --pause."""
    if pause:
        time.sleep(sec)


def df_to_str(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """DataFrame в читаемую строку с отступом."""
    if max_rows is not None and len(df) > max_rows:
        df = pd.concat([df.head(max_rows // 2), df.tail(max_rows // 2)])
    text = df.to_string(index=False)
    return "\n".join("    " + line for line in text.splitlines())


# ── Шаги демонстрации ─────────────────────────────────────────────────────

def step_load_data(history_path: Path) -> pd.DataFrame:
    header("1", "Загружаем историю и очищаем её")
    note("prepare() читает сырые данные, восстанавливает календарь,")
    note("помечает аномалии и импутирует пропуски. Тот же пайплайн,")
    note("что и при обучении — важно для консистентности признаков.")
    print()

    ds = prepare(history_path)
    print(f"    Всего строк: {len(ds.frame)}")
    print(f"    Точек: {ds.frame['restaurant_id'].nunique()}")
    print(f"    Период: {ds.frame['date'].min().date()} … {ds.frame['date'].max().date()}")
    print(f"    NaN в guests после очистки: {ds.frame['guests'].isna().sum()}")
    return ds.frame


def step_features_preview(history: pd.DataFrame, rid: int, forecast_start: pd.Timestamp,
                          horizon: int) -> pd.DataFrame:
    header("2", "Считаем признаки на будущие даты")
    note("Ключевой момент: make_features() вызывается ОДНА и та же")
    note("функция, что и при обучении. Лаги сдвинуты на horizon=7 дней —")
    note("для будущего дня D модель видит только прошлое, не подглядывает.")

    forecast_dates = pd.date_range(forecast_start, periods=horizon, freq="D")

    hist_r = history[history["restaurant_id"] == rid].copy()
    future = pd.DataFrame({
        "date": forecast_dates,
        "restaurant_id": rid,
        "guests": np.nan,
        "revenue": np.nan,
    })
    combined = (
        pd.concat([hist_r[["date", "restaurant_id", "guests", "revenue"]], future],
                  ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )

    featured = make_features(combined, target="guests", horizon=horizon)
    feat_cols = feature_columns(featured)
    forecast_rows = featured[featured["date"].isin(forecast_dates)].copy()

    print(f"\n    Признаков в модели: {len(feat_cols)}")
    print(f"    {feat_cols}")
    section("Что видит модель для будущих дат (первые 5 признаков)")
    preview = forecast_rows[["date"] + feat_cols[:5]].copy()
    preview["date"] = preview["date"].dt.strftime("%Y-%m-%d")
    print(df_to_str(preview.round(2)))
    return forecast_rows


def step_load_model() -> dict:
    header("3", "Загружаем сохранённую модель")
    note("Модель НЕ переобучается при каждом запуске — она обучена заранее")
    note("(python -m src.model) и сохранена в pickle. Инференс быстрый")
    note("и воспроизводимый.")

    bundle = load(MODEL_PATH)
    print()
    print(f"    Модель: {bundle['model_name']}")
    print(f"    Горизонт прогноза: {bundle['horizon']} дней")
    print(f"    Признаков ожидает: {len(bundle['feature_cols'])}")
    print(f"    Целевая переменная: {bundle['target']}")
    return bundle


def step_predict(bundle: dict, history: pd.DataFrame, rid: int,
                 forecast_start: pd.Timestamp, horizon: int) -> pd.DataFrame:
    header("4", "Прогноз на 7 дней")
    note("bundle['pipeline'].predict() — sklearn Pipeline делает")
    note("impute → scale → ridge за один вызов.")

    forecast_dates = pd.date_range(forecast_start, periods=horizon, freq="D")
    forecast = predict_for_dates(bundle, history, forecast_dates, rid)

    print()
    dow_ru = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    forecast_show = forecast.copy()
    forecast_show["день"] = pd.to_datetime(forecast_show["date"]).dt.dayofweek.map(
        lambda i: dow_ru[i]
    )
    forecast_show = forecast_show[["date", "день", "predicted_guests"]]
    print(df_to_str(forecast_show))
    print(f"\n    Итого за {horizon} дней: {forecast['predicted_guests'].sum()} гостей")
    return forecast


def step_metrics(metrics_path: Path) -> None:
    header("5", "Метрики на отложенном периоде (последние 6 недель)")
    note("Модель и бейзлайн оцениваются на ОДНИХ И ТЕХ ЖЕ 84 точках.")
    note("Это обязательное условие честного сравнения.")

    if not metrics_path.exists():
        note("metrics.json не найден. Сначала обучите модель: python -m src.model")
        return

    with open(metrics_path, encoding="utf-8") as f:
        m = json.load(f)

    rows = []
    for name, res in m["all_candidates"].items():
        rows.append({
            "модель": name,
            "MAE": res["MAE"],
            "MAPE": res["MAPE"],
        })
    rows.append({
        "модель": "baseline (lag_7)",
        "MAE": m["baseline"]["MAE"],
        "MAPE": m["baseline"]["MAPE"],
    })

    df = pd.DataFrame(rows)
    print()
    print(df_to_str(df))
    winner = list(m["model"].keys())[0]
    print(f"\n    Победитель по MAE: {winner}")
    print(f"    Конфиг: horizon={m['config']['horizon']}, "
          f"valid_weeks={m['config']['valid_weeks']}, "
          f"random_state={m['config']['random_state']}")


# ── Main ──────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="demo.py",
        description="Пошаговая демонстрация пайплайна прогноза.",
    )
    parser.add_argument("--restaurant", type=int, default=1,
                        help="Идентификатор ресторана (по умолчанию 1).")
    parser.add_argument("--date", type=str, default="2025-10-15",
                        help="Первый день прогноза YYYY-MM-DD (по умолчанию 2025-10-15).")
    parser.add_argument("--history", type=Path,
                        default=Path("data/raw/guests.csv"),
                        help="Путь к сырой истории.")
    parser.add_argument("--pause", action="store_true",
                        help="Пауза 1.5 с между шагами (для живой демонстрации).")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent
    history_path = project_root / args.history
    metrics_path = project_root / "models" / "metrics.json"

    # Заголовок
    print()
    print("█" * WIDTH)
    print(f"  COPERTO FORECAST — демонстрация пайплайна прогноза")
    print(f"  Ресторан: {args.restaurant}, старт прогноза: {args.date}")
    print("█" * WIDTH)

    if not history_path.exists():
        print(f"\nОшибка: файл истории не найден: {history_path}", file=sys.stderr)
        print("Сначала сгенерируйте данные:", file=sys.stderr)
        print("  python data/raw/generate_data.py --out data/raw/guests.csv", file=sys.stderr)
        return 1

    try:
        forecast_start = pd.Timestamp(args.date)
    except Exception:
        print(f"\nОшибка: некорректная дата {args.date!r}. Ожидается YYYY-MM-DD.",
              file=sys.stderr)
        return 2

    # Прогон шагов
    history = step_load_data(history_path)
    wait(args.pause)

    step_features_preview(history, args.restaurant, forecast_start, DEFAULT_HORIZON)
    wait(args.pause)

    bundle = step_load_model()
    wait(args.pause)

    # Проверка: дата прогноза после истории
    last_date = history.loc[history["restaurant_id"] == args.restaurant, "date"].max()
    if forecast_start <= last_date:
        print(f"\n    Внимание: дата прогноза {forecast_start.date()} "
              f"не позже конца истории {last_date.date()}.")
        print(f"    predict.py вернул бы ошибку. Для демо это допустимо,")
        print(f"    но на проде так делать нельзя — модель бы увидела факт.")
    else:
        step_predict(bundle, history, args.restaurant, forecast_start, DEFAULT_HORIZON)
    wait(args.pause)

    step_metrics(metrics_path)

    print()
    print("=" * WIDTH)
    print("  Готово. Продакшн-запуск: python predict.py --date ... --restaurant N")
    print("=" * WIDTH)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())