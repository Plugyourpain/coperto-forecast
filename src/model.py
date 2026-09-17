"""Обучение, сохранение и загрузка модели прогноза гостей.

Модель — sklearn Pipeline: SimpleImputer → StandardScaler → Ridge.
Опционально сравниваем с LightGBM на тех же признаках и разбиении.

Все трансформеры (imputer, scaler) настраиваются ТОЛЬКО на обучающей
выборке — это автоматически гарантируется Pipeline. Ручная реализация
той же последовательности почти всегда приводит к утечке, поэтому
используем Pipeline, как требует ТЗ.

Модель и метаданные (feature_cols, horizon, target) сохраняются
в один joblib-файл, чтобы predict.py не переобучал модель и не
гадал, какие признаки она ожидает.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import prepare
from src.features import DEFAULT_HORIZON, feature_columns, make_features
from src.validation import (
    Metrics,
    compare,
    evaluate,
    naive_baseline_predictions,
    time_split,
)

RANDOM_STATE = 42

# Пути к артефактам — абсолютные, от корня проекта.
# Это позволяет запускать обучение и из корня (python -m src.model),
# и из ноутбуков (notebooks/02_modeling.ipynb), не меняя относительные пути.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"


# ── Pipelines ─────────────────────────────────────────────────────────────────

def build_linear_pipeline() -> Pipeline:
    """Imputer → scaler → Ridge. Ridge не принимает random_state: он детерминирован."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])


def _try_build_boosting_pipeline() -> Pipeline | None:
    """LightGBM, если установлен. Деревья не нуждаются в масштабировании."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return None

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )),
    ])


# ── Train ─────────────────────────────────────────────────────────────────────

def _prepare_features(
    raw_path: str | Path,
    horizon: int,
    valid_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], str]:
    """Готовит train/valid с признаками.

    Важно: make_features вызывается ДО time_split. Иначе лаги для первых
    дней валидации окажутся NaN — история утечёт из train в valid.
    """
    ds = prepare(raw_path)
    featurized = make_features(ds.frame, target=ds.target, horizon=horizon)
    train_df, valid_df = time_split(featurized, valid_weeks=valid_weeks)
    feat_cols = feature_columns(featurized)
    return train_df, valid_df, featurized, feat_cols, ds.target


def train(
    raw_path: str | Path = "data/raw/guests.csv",
    valid_weeks: int = 6,
    horizon: int = DEFAULT_HORIZON,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict:
    """Обучает модели, сравнивает с бейзлайном, сохраняет лучшую.

    Возвращает bundle с полями pipeline, feature_cols, horizon, target.
    """
    train_df, valid_df, featurized, feat_cols, target = _prepare_features(
        raw_path, horizon, valid_weeks
    )

    # Для обучения отбрасываем строки, где признак не посчитался (первые дни ряда).
    train_fit = train_df.dropna(subset=feat_cols + [target]).copy()
    if train_fit.empty:
        raise ValueError("Пустой train после dropna — недостаточно истории")

    X_train = train_fit[feat_cols]
    y_train = train_fit[target]

    X_valid = valid_df[feat_cols]
    y_valid = valid_df[target]

    # ── Кандидаты ────────────────────────────────────────────────────────────
    candidates: dict[str, Pipeline] = {"ridge": build_linear_pipeline()}
    boosting = _try_build_boosting_pipeline()
    if boosting is not None:
        candidates["lightgbm"] = boosting

    results: dict[str, Metrics] = {}
    fitted: dict[str, Pipeline] = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        results[name] = evaluate(y_valid, pipe.predict(X_valid))
        fitted[name] = pipe

    # Бейзлайн считаем на ПОЛНОМ featurized ряду и срезаем по валидации:
    # lag_7 для первых дней валидации должен «видеть» train, иначе
    # бейзлайн посчитается на меньшем числе строк, чем модель — сравнение
    # перестанет быть честным.
    baseline_full = featurized.groupby("restaurant_id", sort=False)[target].shift(horizon)
    baseline_pred = baseline_full.loc[valid_df.index]
    baseline_metrics = evaluate(y_valid, baseline_pred)

    # Выбираем лучшую модель по MAE на валидации
    best_name = min(results, key=lambda n: results[n].mae)
    best_pipe = fitted[best_name]
    best_metrics = results[best_name]

    # ── Сохранение ───────────────────────────────────────────────────────────
    bundle = {
        "pipeline": best_pipe,
        "feature_cols": feat_cols,
        "horizon": horizon,
        "target": target,
        "model_name": best_name,
        "valid_weeks": valid_weeks,
        "random_state": RANDOM_STATE,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    metrics_path.write_text(
        json.dumps(
            {
                "model": {best_name: best_metrics.as_dict()},
                "all_candidates": {n: m.as_dict() for n, m in results.items()},
                "baseline": baseline_metrics.as_dict(),
                "comparison": compare(best_metrics, baseline_metrics).to_dict(),
                "n_train": int(len(train_fit)),
                "n_valid": int(len(valid_df)),
                "config": {
                    "horizon": horizon,
                    "valid_weeks": valid_weeks,
                    "random_state": RANDOM_STATE,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "bundle": bundle,
        "best_name": best_name,
        "metrics": best_metrics,
        "baseline": baseline_metrics,
        "all_metrics": results,
    }


# ── Load & Predict ────────────────────────────────────────────────────────────

def load(model_path: Path = MODEL_PATH) -> dict:
    """Загружает сохранённый bundle. Кидает понятную ошибку, если файла нет."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {model_path}. Сначала обучите её: python -m src.model"
        )
    return joblib.load(model_path)


def predict_for_dates(
    bundle: dict,
    history: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    restaurant_id: int,
) -> pd.DataFrame:
    """Прогноз для одной точки на список будущих дат.

    history — очищенный df (после prepare), содержащий только эту точку
    и только даты < первой даты прогноза.

    Как работает без рекурсии: добавляем пустые строки на будущие даты
    с guests=NaN, прогоняем make_features. Для дня D признаки используют
    значения вплоть до D-horizon. При horizon=7 для D из T+1..T+7 это
    максимум T — то есть всё в прошлом относительно момента прогноза.
    """
    hist = history[history["restaurant_id"] == restaurant_id].copy()
    if hist.empty:
        raise ValueError(f"Нет истории для ресторана {restaurant_id}")
    if hist["date"].max() >= forecast_dates.min():
        raise ValueError("В history есть даты >= начала прогноза")

    future = pd.DataFrame({
        "date": forecast_dates,
        "restaurant_id": restaurant_id,
        "guests": np.nan,
        "revenue": np.nan,
    })

    combined = (
        pd.concat([hist[["date", "restaurant_id", "guests", "revenue"]], future],
                  ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    combined = make_features(combined, target=bundle["target"], horizon=bundle["horizon"])

    to_pred = combined[combined["date"].isin(forecast_dates)].copy()
    if len(to_pred) != len(forecast_dates):
        raise ValueError("Не удалось сформировать признаки на все даты прогноза")

    X = to_pred[bundle["feature_cols"]]
    y_pred = bundle["pipeline"].predict(X)
    y_pred = np.clip(np.round(y_pred), 0, None).astype(int)

    return pd.DataFrame({
        "date": to_pred["date"].dt.strftime("%Y-%m-%d").values,
        "restaurant_id": restaurant_id,
        "predicted_guests": y_pred,
    })


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Обучение модели прогноза.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/guests.csv"))
    parser.add_argument("--valid-weeks", type=int, default=6)
    args = parser.parse_args()

    result = train(raw_path=args.raw, valid_weeks=args.valid_weeks)

    print(f"\nЛучшая модель: {result['best_name']}")
    print(f"Метрики на holdout (последние {args.valid_weeks} недель):\n")
    print(compare(result["metrics"], result["baseline"]).to_string())
    print("\nВсе кандидаты:")
    for name, m in result["all_metrics"].items():
        print(f"  {name}: MAE={m.mae:.2f}, MAPE={m.mape * 100:.2f}%")
    print(f"\nМодель сохранена: {MODEL_PATH}")
    print(f"Метрики сохранены: {METRICS_PATH}")