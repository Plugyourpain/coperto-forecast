"""Загрузка и подготовка дневного ряда посещаемости.

Три РАЗНЫХ случая отсутствия данных обрабатываются по-разному:

  1. Закрытый день  — строка есть, guests = 0. Это факт, оставляем 0.
  2. Сбой выгрузки  — строка есть, но guests = NaN. Импутируем forward-fill.
  3. Точки нет в выгрузке — строки нет. Восстанавливаем календарь, ставим
     флаг was_missing_row и тоже импутируем forward-fill.

Аномалии выявляются робастно (guests > 5 × медиана по точке), заменяются
на NaN и импутируются. Флаг is_anomaly сохраняется для анализа.

Никаких операций, заглядывающих в будущее, на этом этапе нет.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_SCHEMA = {
    "date": "datetime64[ns]",
    "restaurant_id": "int64",
    "guests": "float64",     # float из-за NaN; в чистых данных — целое
    "revenue": "float64",
}

ANOMALY_FACTOR = 5.0


@dataclass(frozen=True)
class Dataset:
    """Контейнер датасета. Таргет — guests."""
    frame: pd.DataFrame
    target: str = "guests"


def load_raw(path: str | Path) -> pd.DataFrame:
    """Читает CSV и приводит к схеме RAW_SCHEMA."""
    df = pd.read_csv(path, parse_dates=["date"])
    missing = set(RAW_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"В данных нет колонок: {sorted(missing)}")
    df = df.astype(RAW_SCHEMA)
    return df.sort_values(["restaurant_id", "date"]).reset_index(drop=True)


def restore_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Достраивает пропущенные календарные дни на точку.

    Пропущенная строка — это третий тип пропуска (точки нет в выгрузке),
    а не закрытый день. Помечаем флагом was_missing_row.
    """
    df = df.copy()
    df["_present"] = True
    parts: list[pd.DataFrame] = []
    for rid, sub in df.groupby("restaurant_id", sort=True):
        full = pd.date_range(sub["date"].min(), sub["date"].max(), freq="D")
        sub = sub.set_index("date").reindex(full)
        sub["restaurant_id"] = rid
        # После reindex строки из raw имеют _present=True,
        # достроенные дни — NaN. Так отделяем одно от другого без fillna.
        sub["was_missing_row"] = sub["_present"].isna()
        sub = sub.drop(columns=["_present"])
        sub.index.name = "date"
        parts.append(sub.reset_index())
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["restaurant_id", "date"]).reset_index(drop=True)


def flag_anomalies(df: pd.DataFrame, factor: float = ANOMALY_FACTOR) -> pd.DataFrame:
    """Робастно помечает выбросы: guests > factor × медиана по точке.

    Медиана вместо среднего — устойчива к самим выбросам. Порог 5×
    оставляет новогодние всплески (×2.2 к медиане дня) как событие,
    но ловит технический POS-баг (×10+).
    """
    out = df.copy()
    median = (
        out.loc[out["guests"] > 0]
        .groupby("restaurant_id")["guests"]
        .median()
    )
    threshold = out["restaurant_id"].map(median) * factor
    out["is_anomaly"] = out["guests"] > threshold
    return out


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Импутирует пропуски и заменяет аномалии.

    Порядок важен:
      1. запоминаем, где был NaN до всего (сбой выгрузки, не восстановленные дни);
      2. аномалии → NaN (чтобы не портили лаги и rolling);
      3. forward-fill по точке (без заглядывания в будущее);
      4. revenue без истории — оценка guests × медианный чек по точке.
    """
    out = df.copy()
    out["was_nan"] = out["guests"].isna() & ~out["was_missing_row"]
    out.loc[out["is_anomaly"], "guests"] = np.nan

    out["guests"] = (
        out.groupby("restaurant_id", sort=False)["guests"]
        .transform(lambda s: s.ffill())
    )

    observed = out.loc[out["revenue"].notna() & (out["guests"] > 0)].copy()
    observed["check"] = observed["revenue"] / observed["guests"]
    median_check = observed.groupby("restaurant_id")["check"].median()
    fallback = out["guests"] * out["restaurant_id"].map(median_check)
    out["revenue"] = out["revenue"].fillna(fallback)

    # Первые дни ряда без истории ffill не заполняет — их безопасно убрать:
    # без лагов модель на них всё равно не обучится.
    out = out.dropna(subset=["guests", "revenue"])
    out["guests"] = out["guests"].astype("int64")
    return out.reset_index(drop=True)


def prepare(path: str | Path) -> Dataset:
    """Полный пайплайн: load → restore_calendar → flag → impute."""
    df = load_raw(path)
    df = restore_calendar(df)
    df = flag_anomalies(df)
    df = impute(df)
    return Dataset(frame=df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Очистка датасета посещаемости.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/guests.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/clean.csv"))
    args = parser.parse_args()

    ds = prepare(args.raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.frame.to_csv(args.out, index=False)

    print(f"OK: {args.out} — {len(ds.frame)} строк\n")
    summary = ds.frame.groupby("restaurant_id").agg(
        rows=("guests", "size"),
        imputed_nan=("was_nan", "sum"),
        missing_row=("was_missing_row", "sum"),
        anomaly=("is_anomaly", "sum"),
        zeros=("guests", lambda s: int((s == 0).sum())),
    )
    print(summary.to_string())
    print("\nПроверка целостности: NaN в guests:", ds.frame["guests"].isna().sum())