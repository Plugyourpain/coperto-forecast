"""Тесты на пайплайн подготовки данных.

Проверяем, что три типа «отсутствия данных» обрабатываются раздельно,
аномалии ловятся робастно, а календарь восстанавливается корректно.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import (
    ANOMALY_FACTOR,
    Dataset,
    flag_anomalies,
    impute,
    load_raw,
    prepare,
    restore_calendar,
)


# ── Фикстуры ──────────────────────────────────────────────────────────────

@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Минимальный чистый ряд: 2 точки × 10 дней."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for rid in (1, 2):
        for i, d in enumerate(dates):
            rows.append({
                "date": d,
                "restaurant_id": rid,
                "guests": 100 + i + 10 * rid,
                "revenue": (100 + i + 10 * rid) * 1000.0,
            })
    return pd.DataFrame(rows)


# ── load_raw ──────────────────────────────────────────────────────────────

def test_load_raw_applies_schema(tmp_path, raw_df):
    path = tmp_path / "raw.csv"
    raw_df.to_csv(path, index=False)

    df = load_raw(path)
    assert df["date"].dtype == "datetime64[ns]"
    assert df["restaurant_id"].dtype == "int64"
    assert df["guests"].dtype == "float64"
    assert df["revenue"].dtype == "float64"
    # отсортировано по (restaurant_id, date)
    assert df.equals(df.sort_values(["restaurant_id", "date"]).reset_index(drop=True))


def test_load_raw_raises_on_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2024-01-01"], "guests": [10]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="нет колонок"):
        load_raw(path)


# ── restore_calendar ──────────────────────────────────────────────────────

def test_restore_calendar_fills_gap(raw_df):
    # Вырезаем середину — 2024-01-04 .. 2024-01-06 (3 дня) у обоих ресторанов
    gap = pd.date_range("2024-01-04", "2024-01-06")
    df = raw_df[~raw_df["date"].isin(gap)].copy()

    out = restore_calendar(df)

    assert len(out) == 2 * 10                    # было 2*7 = 14, стало 20
    assert out["was_missing_row"].sum() == 6     # 3 дня × 2 ресторана
    # восстановленные строки помечены флагом
    restored = out[out["was_missing_row"]]
    assert set(restored["date"].unique()) == set(gap)
    # оригинальные строки не помечены
    original = out[~out["was_missing_row"]]
    assert len(original) == len(df)


def test_restore_calendar_no_gaps(raw_df):
    out = restore_calendar(raw_df)
    assert out["was_missing_row"].sum() == 0
    assert len(out) == len(raw_df)


# ── flag_anomalies ────────────────────────────────────────────────────────

def test_flag_anomalies_robust_to_holiday_burst(raw_df):
    """Новогодний всплеск ×2.5 не должен считаться аномалией."""
    df = raw_df.copy()
    df.loc[df["date"] == "2024-01-05", "guests"] *= 2.5
    out = flag_anomalies(df)
    assert out.loc[out["date"] == "2024-01-05", "is_anomaly"].sum() == 0


def test_flag_anomalies_catches_pos_bug(raw_df):
    """Технический выброс ×10+ должен быть помечен."""
    df = raw_df.copy()
    # Ставим выброс только у ресторана 1, чтобы ожидать ровно 1 аномалию
    mask = (df["date"] == "2024-01-05") & (df["restaurant_id"] == 1)
    df.loc[mask, "guests"] = 1500
    out = flag_anomalies(df)
    flagged = out[out["is_anomaly"]]
    assert len(flagged) == 1
    assert flagged["guests"].iloc[0] == 1500
    assert flagged["restaurant_id"].iloc[0] == 1


def test_flag_anomalies_uses_median_not_mean(raw_df):
    """Среднее бы «спрятало» выброс в свой разброс; медиана — нет."""
    df = raw_df.copy()
    df.loc[df["date"] == "2024-01-05", "guests"] = 1500
    out = flag_anomalies(df, factor=ANOMALY_FACTOR)
    assert out["is_anomaly"].sum() >= 1


# ── impute ────────────────────────────────────────────────────────────────

def test_impute_fills_nan_with_ffill(raw_df):
    """Сбой выгрузки: NaN в середине заполняется предыдущим значением."""
    df = raw_df.copy()
    df.loc[(df["restaurant_id"] == 1) & (df["date"] == "2024-01-05"),
           ["guests", "revenue"]] = np.nan
    df = restore_calendar(df)
    df = flag_anomalies(df)
    out = impute(df)

    row = out[(out["restaurant_id"] == 1) & (out["date"] == "2024-01-05")].iloc[0]
    # 2024-01-04 у ресторана 1: 100 + 3 (индекс дня) + 10 (id ресторана) = 113
    assert row["guests"] == 113
    assert row["was_nan"]


def test_impute_replaces_anomaly_with_ffill(raw_df):
    """Аномалия (POS-баг) заменяется предыдущим валидным значением."""
    df = raw_df.copy()
    df.loc[df["date"] == "2024-01-05", "guests"] = 1500
    df.loc[df["date"] == "2024-01-05", "revenue"] = 2_000_000.0
    df = restore_calendar(df)
    df = flag_anomalies(df)
    out = impute(df)

    # значение аномалии в данных должно было быть заменено на ffill от 01-04
    row = out[out["date"] == "2024-01-05"].iloc[0]
    assert row["guests"] < 500           # точно не 1500


def test_impute_produces_no_nan(raw_df):
    df = restore_calendar(raw_df)
    df = flag_anomalies(df)
    df.loc[df["date"] == "2024-01-05", "guests"] = np.nan  # искусственный NaN
    out = impute(df)
    assert out["guests"].isna().sum() == 0
    assert out["revenue"].isna().sum() == 0


def test_impute_drops_rows_without_history(raw_df):
    """Первая строка ряда после ffill остаётся NaN — должна быть удалена."""
    df = raw_df.copy()
    # обнуляем первую строку каждого ресторана (нет истории для ffill)
    first_dates = df.groupby("restaurant_id")["date"].transform("min")
    df.loc[df["date"] == first_dates, "guests"] = np.nan
    df = restore_calendar(df)
    df = flag_anomalies(df)
    out = impute(df)
    # эти строки отброшены
    assert out["guests"].isna().sum() == 0
    assert out.groupby("restaurant_id").size().min() < 10


# ── prepare ───────────────────────────────────────────────────────────────

def test_prepare_returns_dataset(tmp_path, raw_df):
    path = tmp_path / "raw.csv"
    raw_df.to_csv(path, index=False)

    ds = prepare(path)
    assert isinstance(ds, Dataset)
    assert ds.target == "guests"
    # все флаги присутствуют
    for col in ("was_missing_row", "was_nan", "is_anomaly"):
        assert col in ds.frame.columns


def test_prepare_end_to_end_on_messy_data(tmp_path, raw_df):
    """Один тест на всё: смешиваем три типа проблем и проверяем,
    что выживает ровно то, что ожидали."""
    df = raw_df.copy()

    # 1) вырезаем 2 дня у обоих ресторанов (отсутствие строк)
    gap = pd.date_range("2024-01-05", periods=2)
    df = df[~df["date"].isin(gap)]

    # 2) добавляем NaN-сбой в середине
    df.loc[(df["restaurant_id"] == 1) & (df["date"] == "2024-01-08"),
           "guests"] = np.nan

    # 3) добавляем POS-баг
    pos_mask = (df["date"] == "2024-01-07") & (df["restaurant_id"] == 1)
    df.loc[pos_mask, "guests"] = 1500

    path = tmp_path / "messy.csv"
    df.to_csv(path, index=False)

    ds = prepare(path)
    # после очистки NaN нет
    assert ds.frame["guests"].isna().sum() == 0
    assert ds.frame["revenue"].isna().sum() == 0
    # gap был восстановлен
    assert ds.frame["was_missing_row"].sum() == 4     # 2 дня × 2 ресторана
    # аномалия поймана
    assert ds.frame["is_anomaly"].sum() == 1