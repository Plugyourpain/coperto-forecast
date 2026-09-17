"""Тесты на разбиение, метрики и бейзлайн."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation import (
    Metrics,
    compare,
    evaluate,
    mae,
    mape,
    naive_baseline_predictions,
    time_split,
)


def _frame(n_days: int = 100, rid: int = 1) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return pd.DataFrame({
        "date": dates,
        "restaurant_id": rid,
        "guests": np.arange(1, n_days + 1, dtype="int64") + 50,
        "revenue": 1000.0 * (np.arange(1, n_days + 1) + 50),
    })


def test_time_split_no_overlap_and_ordered():
    df = _frame(n_days=100)
    train, valid = time_split(df, valid_weeks=2)
    assert train["date"].max() < valid["date"].min()
    assert len(valid) == 14
    assert train["date"].min() == df["date"].min()
    assert valid["date"].max() == df["date"].max()


def test_time_split_too_short_raises():
    df = _frame(n_days=20)
    with pytest.raises(ValueError):
        time_split(df, valid_weeks=4)


def test_mae_mape_simple_cases():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # |diff| = 2, 2, 3 → MAE = 7/3
    assert abs(mae(y_true, y_pred) - 7 / 3) < 1e-9
    # APE = 0.2, 0.1, 0.1 → MAPE = 0.4 / 3
    val, n, excluded = mape(y_true, y_pred)
    assert abs(val - 0.4 / 3) < 1e-9
    assert n == 3
    assert excluded == 0


def test_mape_excludes_zeros_and_reports_count():
    y_true = np.array([10.0, 0.0, 20.0, 0.0])
    y_pred = np.array([11.0, 5.0, 18.0, 1.0])
    val, n, excluded = mape(y_true, y_pred)
    assert n == 2
    assert excluded == 2
    assert abs(val - (0.1 + 0.1) / 2) < 1e-9


def test_mae_ignores_nan():
    y_true = np.array([10.0, np.nan, 20.0])
    y_pred = np.array([11.0, 12.0, 18.0])
    assert abs(mae(y_true, y_pred) - 1.5) < 1e-9


def test_evaluate_packs_metrics():
    y_true = np.array([10.0, 0.0, 20.0])
    y_pred = np.array([11.0, 1.0, 18.0])
    m = evaluate(y_true, y_pred)
    assert isinstance(m, Metrics)
    assert m.n_mae == 3
    assert m.n_mape == 2
    assert m.n_excluded_zero == 1
    assert "MAE" in m.as_dict()
    assert m.as_dict()["MAPE"].endswith("%")


def test_naive_baseline_is_lag_7_within_restaurant():
    df = _frame(n_days=30, rid=1)
    pred = naive_baseline_predictions(df)
    sorted_df = df.sort_values(["restaurant_id", "date"]).reset_index(drop=True)
    # первые 7 значений — NaN
    assert pred.iloc[:7].isna().all()
    # 8-е = 1-е
    assert pred.iloc[7] == sorted_df["guests"].iloc[0]


def test_naive_baseline_does_not_cross_restaurants():
    a = _frame(n_days=20, rid=1)
    b = _frame(n_days=20, rid=2).assign(guests=lambda d: d["guests"] * 10)
    df = pd.concat([a, b], ignore_index=True)
    pred = naive_baseline_predictions(df)

    # b_sorted сохраняет исходные индексы строк df (20..39),
    # поэтому pred.loc[...] вытащит ровно ресторан 2.
    b_sorted = df[df["restaurant_id"] == 2].sort_values("date")
    pred_b = pred.loc[b_sorted.index]

    assert pred_b.iloc[7] == b_sorted["guests"].iloc[0]
    # и первое значение — NaN (нет истории)
    assert pred_b.iloc[:7].isna().all()


def test_compare_table_contains_baseline_model_and_gain():
    b = Metrics(mae=10.0, mape=0.20, n_mae=100, n_mape=95, n_excluded_zero=5)
    m = Metrics(mae=8.0, mape=0.15, n_mae=100, n_mape=95, n_excluded_zero=5)
    table = compare(m, b)
    assert list(table.index) == ["baseline (lag_7)", "model", "улучшение"]
    assert table.loc["model", "MAE"] == 8.0
    # (10-8)/10 = +20%
    assert "+20.0%" in table.loc["улучшение", "MAE"]