"""Генерация синтетического датасета посещаемости ресторанов.

Свойства ряда: тренд, недельная и годовая сезонность, праздничные всплески,
шум, три разных типа пропусков и один технический выброс.

Запуск:
    python data/raw/generate_data.py --out data/raw/guests.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── Параметры генерации ──────────────────────────────────────────────────────
START = "2024-04-01"
END = "2025-09-30"          # 18 месяцев
SEED = 42

RESTAURANTS: dict[int, dict[str, float]] = {
    1: {"base": 110.0, "trend_per_day": 0.03, "noise_std": 8.0, "avg_check": 1800.0},
    2: {"base": 75.0,  "trend_per_day": 0.02, "noise_std": 6.0, "avg_check": 2200.0},
}

# Пн..Вс — множители недельной сезонности
DOW_FACTOR = np.array([0.85, 0.90, 0.95, 1.00, 1.15, 1.25, 1.10])

# Янв..Дек — множители годовой сезонности
MONTH_FACTOR = np.array(
    [0.85, 0.85, 0.95, 1.00, 1.05, 1.10, 1.15, 1.10, 1.05, 1.00, 0.95, 1.20]
)

# Праздники: (месяц, день) -> множитель. Новогодняя ночь — событие, не выброс.
HOLIDAY_FACTOR: dict[tuple[int, int], float] = {
    (1, 1): 1.8, (1, 7): 1.4, (3, 8): 1.3, (5, 1): 1.2, (5, 9): 1.2,
    (12, 30): 1.5, (12, 31): 2.2,
}
# ─────────────────────────────────────────────────────────────────────────────


def _generate_series(
    restaurant_id: int,
    config: dict[str, float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Генерирует чистый (без пропусков) дневной ряд для одной точки."""
    dates = pd.date_range(START, END, freq="D")
    n = len(dates)

    trend = config["trend_per_day"] * np.arange(n)
    weekly = DOW_FACTOR[dates.dayofweek.values]
    yearly = MONTH_FACTOR[dates.month.values - 1]

    holiday = np.ones(n)
    for (m, d), factor in HOLIDAY_FACTOR.items():
        holiday[(dates.month == m) & (dates.day == d)] = factor

    mean = (config["base"] + trend) * weekly * yearly * holiday
    noise = rng.normal(0.0, config["noise_std"], n)
    guests = np.clip(np.round(mean + noise), 0, None).astype("int64")

    revenue = guests * config["avg_check"] * rng.normal(1.0, 0.05, n)
    revenue = np.clip(revenue, 0, None).round(2)

    return pd.DataFrame(
        {
            "date": dates,
            "restaurant_id": np.int64(restaurant_id),
            "guests": guests,
            "revenue": revenue,
        }
    )


def _inject_problems(
    df: pd.DataFrame,
    restaurant_id: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Вносит три типа пропусков и одну аномалию.

    Возвращает (df, список дат, которые вырежем целиком — имитация
    отсутствия точки в выгрузке).
    """
    df = df.copy()

    # Тип 1: закрытый день у ресторана 2 — каждый понедельник (сан. день).
    if restaurant_id == 2:
        closed = df["date"].dt.dayofweek == 0
        df.loc[closed, "guests"] = 0
        df.loc[closed, "revenue"] = 0.0

    # Тип 2: сбой выгрузки — 2% случайных NaN (НЕ 0!).
    nan_mask = rng.random(len(df)) < 0.02
    df.loc[nan_mask, ["guests", "revenue"]] = np.nan

    # Тип 3: точка временно отсутствует в выгрузке — вырежем 14-дневный блок.
    gap_start = pd.Timestamp("2025-02-03")
    gap_end = pd.Timestamp("2025-02-16")
    absent_dates = list(pd.date_range(gap_start, gap_end, freq="D"))

    # Аномалия: технический выброс (POS-баг) у ресторана 1.
    anomaly_dates: list[pd.Timestamp] = []
    if restaurant_id == 1:
        anomaly_day = pd.Timestamp("2024-11-12")
        mask = df["date"] == anomaly_day
        df.loc[mask, "guests"] = 1500     # x10 к норме
        df.loc[mask, "revenue"] = 2_700_000.0
        anomaly_dates.append(anomaly_day)

    df = df[~df["date"].isin(absent_dates)].reset_index(drop=True)
    return df, absent_dates


def generate(out_path: Path) -> pd.DataFrame:
    """Собирает финальный датасет по всем ресторанам и сохраняет в CSV."""
    rng = np.random.default_rng(SEED)
    frames: list[pd.DataFrame] = []
    for rid, cfg in RESTAURANTS.items():
        clean = _generate_series(rid, cfg, rng)
        with_problems, _ = _inject_problems(clean, rid, rng)
        frames.append(with_problems)

    df = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["restaurant_id", "date"])
        .reset_index(drop=True)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Сгенерировать синтетический датасет.")
    parser.add_argument("--out", type=Path, default=Path("data/raw/guests.csv"))
    args = parser.parse_args()

    df = generate(args.out)
    print(f"OK: {args.out} — {len(df)} строк, {df['restaurant_id'].nunique()} точки")
    print(df.dtypes.to_string())
    print("\nДоля NaN в guests:", round(df["guests"].isna().mean(), 4))
    print("Закрытых дней (guests==0, не NaN):", int((df["guests"] == 0).sum()))


if __name__ == "__main__":
    main()