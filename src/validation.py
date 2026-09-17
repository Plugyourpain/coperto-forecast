"""Разбиение по времени, метрики и наивный бейзлайн.

Правила (ТЗ):
- Только разбиение по времени, без shuffle.
- Метрики: MAE и MAPE на отложенном периоде.
- Обязателен наивный бейзлайн, посчитанный на той же выборке и по той же метрике.

Подводный камень MAPE: при guests=0 (закрытый день) метрика не определена.
Решение — исключать такие точки и явно сообщать их количество. MAE
считается по всем точкам.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET = "guests"


@dataclass(frozen=True)
class Metrics:
    """Метрики на отложенном периоде."""
    mae: float
    mape: float                  # доля, не проценты (0.12 == 12%)
    n_mae: int                   # точек, на которых считался MAE
    n_mape: int                  # точек, на которых считался MAPE (без нулей)
    n_excluded_zero: int         # сколько нулевых значений исключено из MAPE

    def as_dict(self) -> dict[str, float | int]:
        return {
            "MAE": round(self.mae, 2),
            "MAPE": f"{self.mape * 100:.2f}%",
            "N (MAE)": self.n_mae,
            "N (MAPE)": self.n_mape,
            "Исключено нулей": self.n_excluded_zero,
        }


def time_split(
    df: pd.DataFrame,
    valid_weeks: int = 6,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит датафрейм по времени: последние `valid_weeks` недель — валидация.

    Точка отсечения — максимум по дате минус valid_weeks недель.
    Строки с датой == cutoff остаются в train, чтобы длина валидации
    была ровно `valid_weeks * 7` дней.
    """
    if valid_weeks < 1:
        raise ValueError("valid_weeks должен быть >= 1")
    if date_col not in df.columns:
        raise ValueError(f"Нет колонки {date_col!r}")

    cutoff = df[date_col].max() - pd.Timedelta(weeks=valid_weeks)
    train = df[df[date_col] <= cutoff].copy()
    valid = df[df[date_col] > cutoff].copy()

    if train.empty or valid.empty:
        raise ValueError(
            f"Недостаточно истории: train={len(train)}, valid={len(valid)}. "
            f"Нужно минимум {(valid_weeks * 7) + 30} дней."
        )
    return train, valid


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean Absolute Error. Считается по всем точкам."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def mape(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> tuple[float, int, int]:
    """Mean Absolute Percentage Error.

    Возвращает (mape_доля, n_использовано, n_исключено_нулей).
    Нулевые y_true исключаются: MAPE на них не определена.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    nonzero_mask = valid_mask & (y_true != 0)
    n_excluded = int((valid_mask & (y_true == 0)).sum())

    if nonzero_mask.sum() == 0:
        return float("nan"), 0, n_excluded

    ape = np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]) / np.abs(y_true[nonzero_mask])
    return float(np.mean(ape)), int(nonzero_mask.sum()), n_excluded


def evaluate(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> Metrics:
    """Считает MAE и MAPE и упаковывает в Metrics."""
    mae_val = mae(y_true, y_pred)
    mape_val, n_mape, n_excluded = mape(y_true, y_pred)
    n_mae = int((~(np.isnan(np.asarray(y_true, dtype=float))
                   | np.isnan(np.asarray(y_pred, dtype=float)))).sum())
    return Metrics(
        mae=mae_val,
        mape=mape_val,
        n_mae=n_mae,
        n_mape=n_mape,
        n_excluded_zero=n_excluded,
    )


def naive_baseline_predictions(
    df: pd.DataFrame,
    target: str = TARGET,
    horizon: int = 7,
) -> pd.Series:
    """Наивный бейзлайн: значение того же дня недели неделю назад.

    Это `lag_7`. Требование ТЗ. Возвращает Series той же длины, что df,
    выровненный по индексу. Для первых 7 дней каждой точки — NaN.

    Реализация: shift внутри restaurant_id, чтобы не перетекать между точками.
    """
    if target not in df.columns:
        raise ValueError(f"Нет колонки {target!r}")
    if horizon != 7:
        # Обобщение: лаг равен горизонту. Для 7-дневного прогноза берём lag_7.
        pass
    sorted_df = df.sort_values(["restaurant_id", "date"])
    baseline = sorted_df.groupby("restaurant_id", sort=False)[target].shift(horizon)
    return baseline.reindex(df.index)


def compare(
    model: Metrics,
    baseline: Metrics,
) -> pd.DataFrame:
    """Табличка для README: модель vs бейзлайн, абсолютный и относительный прирост.

    Относительный прирост по MAE: (baseline.mae - model.mae) / baseline.mae.
    Положительное число означает, что модель лучше бейзлайна.
    """
    mae_gain = (baseline.mae - model.mae) / baseline.mae if baseline.mae else float("nan")
    mape_gain = (
        (baseline.mape - model.mape) / baseline.mape
        if baseline.mape and not np.isnan(baseline.mape)
        else float("nan")
    )
    return pd.DataFrame(
        {
            "MAE": [round(baseline.mae, 2), round(model.mae, 2),
                    f"{mae_gain * 100:+.1f}%"],
            "MAPE": [f"{baseline.mape * 100:.2f}%", f"{model.mape * 100:.2f}%",
                     f"{mape_gain * 100:+.1f}%"],
        },
        index=["baseline (lag_7)", "model", "улучшение"],
    )


def per_restaurant_metrics(
    df: pd.DataFrame,
    y_pred: pd.Series,
    target: str = TARGET,
    id_col: str = "restaurant_id",
) -> pd.DataFrame:
    """Метрики в разрезе точек — для README и EDA."""
    tmp = df[[id_col, target]].copy()
    tmp["_pred"] = y_pred.values
    rows = []
    for rid, sub in tmp.groupby(id_col, sort=True):
        m = evaluate(sub[target], sub["_pred"])
        rows.append({
            "restaurant_id": rid,
            "MAE": round(m.mae, 2),
            "MAPE": f"{m.mape * 100:.2f}%",
            "N (MAPE)": m.n_mape,
            "Исключено нулей": m.n_excluded_zero,
        })
    return pd.DataFrame(rows).set_index("restaurant_id")