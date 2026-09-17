"""Feature engineering для прогноза посещаемости ресторана.

Ключевое архитектурное решение: одна и та же функция `make_features`
используется и при обучении, и при инференсе. Это гарантирует, что
признаки на входе модели всегда посчитаны одинаково.

Все лаги и скользящие статистики сдвинуты минимум на `horizon` дней назад.
По умолчанию horizon=7 — ровно столько дней модель прогнозирует. Это
означает: признак в строке за день D не зависит от значений в
D-horizon+1 .. D. Иначе для 7-дневного прогноза их пришлось бы «угадывать»
рекурсивно, а ошибка бы компаундилась.

Целевая переменная в признаках напрямую не участвует — только через
лаги и скользящие со сдвигом.
"""
from __future__ import annotations

import pandas as pd

# Лаги: минимальный — 7, т.к. горизонт прогноза = 7 дней.
LAGS: tuple[int, ...] = (7, 14, 21)
ROLLING_WINDOWS: tuple[int, ...] = (7, 28)
DEFAULT_HORIZON: int = 7

# Государственные праздники РФ как (месяц, день). Фиксированные даты;
# переносы не учитываем — их влияние на дневной ряд слабое.
RU_HOLIDAYS: frozenset[tuple[int, int]] = frozenset({
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
    (2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4),
    (12, 31),
})

# Служебные колонки, которые не идут в модель как признаки.
SERVICE_COLUMNS: frozenset[str] = frozenset({
    "date", "restaurant_id", "guests", "revenue",
    "was_nan", "was_missing_row", "is_anomaly",
})


def make_features(
    df: pd.DataFrame,
    target: str = "guests",
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """Строит признаки для прогноза на `horizon` дней вперёд.

    Parameters
    ----------
    df
        Датафрейм со схемой [date, restaurant_id, guests, revenue, ...].
        Может содержать строки с будущими датами и NaN в target —
        лаги и скользящие для них будут брать только прошлое.
    target
        Имя целевой колонки (по умолчанию "guests").
    horizon
        Горизонт прогноза в днях. Все лаги и rolling сдвинуты минимум
        на `horizon`, чтобы не было утечки. Лаги короче горизонта
        молча пропускаются.

    Returns
    -------
    Копия df с добавленными признаками. Строки без достаточной истории
    сохраняются, но содержат NaN в лагах — модель их отбросит.
    """
    if target not in df.columns:
        raise ValueError(f"Нет колонки {target!r} в датафрейме")
    if horizon < 1:
        raise ValueError("horizon должен быть >= 1")

    out = df.sort_values(["restaurant_id", "date"]).copy()

    # ── Календарные признаки: известны для любой будущей даты ───────────
    dates = out["date"]
    out["dow"] = dates.dt.dayofweek.astype("int8")
    out["month"] = dates.dt.month.astype("int8")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")

    holiday_keys = {f"{m:02d}-{d:02d}" for m, d in RU_HOLIDAYS}
    out["is_holiday"] = (
        dates.dt.strftime("%m-%d").isin(holiday_keys).astype("int8")
    )

    # ── Лаги и скользящие: по каждой точке, со сдвигом >= horizon ───────
    grouped = out.groupby("restaurant_id", sort=False)[target]

    for lag in LAGS:
        if lag < horizon:
            # Лаг короче горизонта не вычислим на момент прогноза — пропускаем.
            continue
        out[f"lag_{lag}"] = grouped.shift(lag)

    for window in ROLLING_WINDOWS:
        # Сдвигаем ряд на horizon, затем считаем окно — все значения
        # внутри окна гарантированно известны в день прогноза.
        out[f"roll_mean_{window}"] = grouped.transform(
            lambda s, w=window, h=horizon: s.shift(h).rolling(w).mean()
        )
        out[f"roll_std_{window}"] = grouped.transform(
            lambda s, w=window, h=horizon: s.shift(h).rolling(w).std()
        )

    return out.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Список колонок-признаков для подачи в модель.

    Исключает дату, id, таргет, вспомогательные флаги и revenue
    (revenue коррелирует с таргетом и сам является следствием, а не
    драйвером — использовать его как признак для прогноза гостей
    означало бы частичную утечку целевой переменной).
    """
    return [c for c in df.columns if c not in SERVICE_COLUMNS]