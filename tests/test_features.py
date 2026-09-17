"""Тесты на функцию make_features.

Главная цель — доказать, что признаки не заглядывают в будущее
относительно прогнозируемого дня, а также не «перетекают» между точками.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import DEFAULT_HORIZON, feature_columns, make_features


def _synthetic(n_days: int = 400, restaurant_id: int = 1, seed: int = 0) -> pd.DataFrame:
    """400 дней по умолчанию — хватает, чтобы покрыть разные месяцы
    и оба сезона (зима/лето) для проверки календарных признаков."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": dates,
        "restaurant_id": restaurant_id,
        "guests": rng.integers(50, 150, size=n_days).astype("int64"),
        "revenue": rng.uniform(10_000, 30_000, size=n_days),
    })


def test_lag_matches_shifted_target():
    """lag_7 в строке D должен быть равен guests в строке D-7."""
    df = _synthetic()
    feats = make_features(df).sort_values("date").reset_index(drop=True)
    expected = df.sort_values("date")["guests"].shift(7).reset_index(drop=True)
    np.testing.assert_array_equal(feats["lag_7"].values, expected.values)


def test_perturbing_one_day_does_not_change_earlier_features():
    """Изменение guests в день D не должно влиять на признаки
    в дни < D (будущее не влияет на прошлое).
    """
    df = _synthetic()
    df2 = df.copy()
    D = pd.Timestamp("2024-02-15")
    df2.loc[df2["date"] == D, "guests"] = 99_999

    f1 = make_features(df).sort_values("date").reset_index(drop=True)
    f2 = make_features(df2).sort_values("date").reset_index(drop=True)

    early = f1["date"] < D
    pd.testing.assert_frame_equal(
        f1.loc[early].reset_index(drop=True),
        f2.loc[early].reset_index(drop=True),
    )


def test_perturbing_one_day_does_not_change_features_within_horizon():
    """Ключевой тест на утечку.

    Если признак в день D+h (h < horizon) зависит от значения в D, значит
    для реального 7-дневного прогноза его нельзя вычислить. Меняем D —
    признаки в D+1..D+horizon-1 должны остаться прежними.
    """
    df = _synthetic()
    df2 = df.copy()
    D = pd.Timestamp("2024-02-15")
    df2.loc[df2["date"] == D, "guests"] = 99_999

    f1 = make_features(df).sort_values("date").reset_index(drop=True)
    f2 = make_features(df2).sort_values("date").reset_index(drop=True)

    within = (f1["date"] > D) & (f1["date"] < D + pd.Timedelta(days=DEFAULT_HORIZON))
    assert within.any(), "тестовые данные слишком короткие"
    pd.testing.assert_frame_equal(
        f1.loc[within].reset_index(drop=True),
        f2.loc[within].reset_index(drop=True),
    )


def test_perturbing_one_day_does_change_features_at_or_after_horizon():
    """Комплементарный тест: значение в D должно повлиять на признаки
    в D+horizon (иначе лаг вообще ничего не читает).
    """
    df = _synthetic()
    df2 = df.copy()
    D = pd.Timestamp("2024-02-15")
    df2.loc[df2["date"] == D, "guests"] = 99_999

    f1 = make_features(df).sort_values("date").reset_index(drop=True)
    f2 = make_features(df2).sort_values("date").reset_index(drop=True)

    at_h = f1["date"] == D + pd.Timedelta(days=DEFAULT_HORIZON)
    assert at_h.any()
    assert not f1.loc[at_h, "lag_7"].equals(f2.loc[at_h, "lag_7"])


def test_no_cross_restaurant_leakage():
    """Лаги не должны перетекать между ресторанами."""
    a = _synthetic(n_days=30, restaurant_id=1, seed=1)
    b = _synthetic(n_days=30, restaurant_id=2, seed=2)
    df = pd.concat([a, b], ignore_index=True)
    feats = make_features(df).sort_values(["restaurant_id", "date"])

    for rid in (1, 2):
        sub = feats[feats["restaurant_id"] == rid].reset_index(drop=True)
        original = df[df["restaurant_id"] == rid].sort_values("date").reset_index(drop=True)
        # Первые 7 значений lag_7 в каждой группе — NaN
        assert sub.loc[:6, "lag_7"].isna().all()
        # 8-е значение = 1-е значение той же точки, а не соседней
        assert sub.loc[7, "lag_7"] == original.loc[0, "guests"]


def test_calendar_features_are_correct():
    df = _synthetic()
    feats = make_features(df)
    for col in ("dow", "month", "is_weekend", "is_holiday"):
        assert col in feats.columns

    # 2024-01-06 — суббота, новогодние каникулы
    row = feats[feats["date"] == pd.Timestamp("2024-01-06")].iloc[0]
    assert row["dow"] == 5
    assert row["is_weekend"] == 1
    assert row["is_holiday"] == 1

    # 2024-06-19 — среда, не праздник
    row = feats[feats["date"] == pd.Timestamp("2024-06-19")].iloc[0]
    assert row["dow"] == 2
    assert row["is_weekend"] == 0
    assert row["is_holiday"] == 0

    # 2024-03-08 — пятница, праздник, но не выходной по dow
    row = feats[feats["date"] == pd.Timestamp("2024-03-08")].iloc[0]
    assert row["dow"] == 4
    assert row["is_weekend"] == 0
    assert row["is_holiday"] == 1


def test_invalid_horizon_raises():
    df = _synthetic()
    with pytest.raises(ValueError):
        make_features(df, horizon=0)


def test_missing_target_raises():
    df = _synthetic().drop(columns=["guests"])
    with pytest.raises(ValueError):
        make_features(df, target="guests")


def test_feature_columns_excludes_service():
    df = _synthetic()
    feats = make_features(df)
    cols = feature_columns(feats)
    for service in ("date", "restaurant_id", "guests", "revenue"):
        assert service not in cols
    for expected in ("lag_7", "roll_mean_7", "dow", "is_weekend", "is_holiday"):
        assert expected in cols