from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor


DATA_PATH = Path("data/processed/cgm_lag_dataset.csv")
OUT_DIR = Path("reports")
RESULTS_PATH = OUT_DIR / "cgm_baseline_results.csv"

TARGETS = ["target_glucose_30min", "target_glucose_60min"]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col.startswith("glucose_lag_")
        or col.startswith("glucose_change_")
        or col.startswith("glucose_rolling_")
    ]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred),
    }


def evaluate_model(
    model_name: str,
    horizon: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    metrics = regression_metrics(y_true, y_pred)

    row = {
        "model": model_name,
        "target": horizon,
        **metrics,
    }

    return row


def build_models(feature_cols: list[str]) -> dict:
    ridge = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )

    random_forest = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    xgboost = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    return {
        "ridge": ridge,
        "random_forest": random_forest,
        "xgboost": xgboost,
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run src/features/build_cgm_lag_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH)
    feature_cols = get_feature_columns(df)

    train_df = df[df["split"] == "training"].copy()
    test_df = df[df["split"] == "testing"].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Training or testing split is empty.")

    print(f"Loaded dataset: {len(df):,} rows")
    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows:  {len(test_df):,}")
    print(f"Features: {len(feature_cols)}")
    print()

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    results = []

    for target in TARGETS:
        print("=" * 80)
        print(f"Target: {target}")
        print("=" * 80)

        y_train = train_df[target].to_numpy()
        y_test = test_df[target].to_numpy()

        # Persistence baseline: future glucose = current glucose.
        persistence_pred = test_df["glucose_lag_0min"].to_numpy()
        row = evaluate_model("persistence", target, y_test, persistence_pred)
        results.append(row)
        print(
            f"persistence | MAE={row['mae']:.3f} | "
            f"RMSE={row['rmse']:.3f} | R2={row['r2']:.3f}"
        )

        models = build_models(feature_cols)

        for model_name, model in models.items():
            print(f"Training {model_name}...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            row = evaluate_model(model_name, target, y_test, preds)
            results.append(row)

            print(
                f"{model_name} | MAE={row['mae']:.3f} | "
                f"RMSE={row['rmse']:.3f} | R2={row['r2']:.3f}"
            )

        print()

    results_df = pd.DataFrame(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    print(f"Saved results to {RESULTS_PATH}")
    print()
    print(results_df)


if __name__ == "__main__":
    main()