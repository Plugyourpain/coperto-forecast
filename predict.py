"""CLI для прогноза посещаемости на N дней вперёд.

Примеры:
    python predict.py --date 2025-10-01 --restaurant 1
    python predict.py --date 2025-10-01 --restaurant 1 --days 7 --out forecast.csv
    python predict.py --date 2025-10-01 --restaurant 1 --history data/processed/clean.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.data import prepare
from src.model import MODEL_PATH, load, predict_for_dates

DEFAULT_HISTORY = Path("data/raw/guests.csv")
DEFAULT_HORIZON = 7


def _parse_date(value: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Некорректная дата: {value!r}. Ожидается YYYY-MM-DD."
        ) from exc
    if pd.isna(ts):
        raise argparse.ArgumentTypeError(f"Некорректная дата: {value!r}")
    return ts


def _load_history(raw_path: Path) -> pd.DataFrame:
    """Загружает и очищает историю тем же пайплайном, что использовался при обучении."""
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Файл истории не найден: {raw_path}. "
            f"Сгенерируйте данные: python data/raw/generate_data.py --out {raw_path}"
        )
    return prepare(raw_path).frame


def _restaurant_available(df: pd.DataFrame, restaurant_id: int) -> bool:
    return restaurant_id in set(df["restaurant_id"].unique())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="Прогноз числа гостей ресторана на N дней вперёд.",
    )
    parser.add_argument(
        "--date", required=True, type=_parse_date,
        help="Первый день прогноза, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--restaurant", required=True, type=int,
        help="Идентификатор ресторана.",
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_HORIZON,
        help=f"Горизонт прогноза в днях (по умолчанию {DEFAULT_HORIZON}).",
    )
    parser.add_argument(
        "--history", type=Path, default=DEFAULT_HISTORY,
        help=f"Путь к сырой истории (по умолчанию {DEFAULT_HISTORY}).",
    )
    parser.add_argument(
        "--model", type=Path, default=MODEL_PATH,
        help=f"Путь к сохранённой модели (по умолчанию {MODEL_PATH}).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Если задано — сохранить прогноз в CSV по этому пути.",
    )
    args = parser.parse_args(argv)

    if args.days < 1:
        print("Ошибка: --days должен быть >= 1", file=sys.stderr)
        return 2

    # 1. Загружаем модель
    try:
        bundle = load(args.model)
    except FileNotFoundError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    # 2. Проверяем горизонт: модель обучалась на конкретном horizon
    if args.days != bundle["horizon"]:
        print(
            f"Ошибка: модель обучена на горизонт {bundle['horizon']} дней, "
            f"а запрошено {args.days}. Переобучите модель или измените --days.",
            file=sys.stderr,
        )
        return 2

    # 3. История
    try:
        history = _load_history(args.history)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    if not _restaurant_available(history, args.restaurant):
        available = sorted(history["restaurant_id"].unique().tolist())
        print(
            f"Ошибка: ресторан {args.restaurant} не найден в истории. "
            f"Доступные: {available}",
            file=sys.stderr,
        )
        return 1

    # 4. Проверяем, что начало прогноза строго после конца истории
    last_history_date = history.loc[
        history["restaurant_id"] == args.restaurant, "date"
    ].max()
    if args.date <= last_history_date:
        print(
            f"Ошибка: дата прогноза {args.date.date()} должна быть позже "
            f"конца истории {last_history_date.date()}.",
            file=sys.stderr,
        )
        return 2

    # 5. Прогноз
    forecast_dates = pd.date_range(args.date, periods=args.days, freq="D")
    try:
        forecast = predict_for_dates(
            bundle, history, forecast_dates, args.restaurant
        )
    except ValueError as exc:
        print(f"Ошибка прогноза: {exc}", file=sys.stderr)
        return 1

    # 6. Вывод
    print(f"\nПрогноз для ресторана {args.restaurant} "
          f"({bundle['model_name']}, горизонт {bundle['horizon']} дней):\n")
    print(forecast.to_string(index=False))

    total = int(forecast["predicted_guests"].sum())
    print(f"\nИтого за {args.days} дней: {total} гостей")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(args.out, index=False)
        print(f"Сохранено: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())