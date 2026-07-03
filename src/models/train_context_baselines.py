from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor


DATA_PATH = Path("data/processed/cgm_context_dataset.csv")
OUT_DIR = Path("reports")
RESULTS_PATH = OUT_DIR / "context_baseline_results.csv"

TARGETS = ["target_glucose_30min", "target_glucose_60min"]


def get_cgm_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col.startswith("glucose_lag_")
        or col.startswith("glucose_change_")
        or col.startswith("glucose_rolling_")
    ]


def get_context_feature_columns(df: pd.DataFrame) -> list[str]:
    prefixes = (
        "carbs_",
        "meal_",
        "bolus_",
        "bolus_carb_input_",
        "time_since_",
        "has_prior_",
        "basal_",
    )

    return [col for col in df.columns if col.startswith(prefixes)]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred),
    }


def evaluate_model(
    model_name: str,
    feature_set: str,
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    metrics = regression_metrics(y_true, y_pred)

    return {
        "model": model_name,
        "feature_set": feature_set,
        "target": target,
        **metrics,
    }


def build_models() -> dict:
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


def train_and_evaluate_feature_set(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    feature_set_name: str,
) -> list[dict]:
    results = []

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    print(f"Feature set: {feature_set_name}")
    print(f"Features: {len(feature_cols)}")
    print()

    for target in TARGETS:
        print("=" * 80)
        print(f"Feature set: {feature_set_name} | Target: {target}")
        print("=" * 80)

        y_train = train_df[target].to_numpy()
        y_test = test_df[target].to_numpy()

        # Persistence baseline is only meaningful as current glucose.
        persistence_pred = test_df["glucose_lag_0min"].to_numpy()
        row = evaluate_model(
            model_name="persistence",
            feature_set=feature_set_name,
            target=target,
            y_true=y_test,
            y_pred=persistence_pred,
        )
        results.append(row)

        print(
            f"persistence | MAE={row['mae']:.3f} | "
            f"RMSE={row['rmse']:.3f} | R2={row['r2']:.3f}"
        )

        models = build_models()

        for model_name, model in models.items():
            print(f"Training {model_name}...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            row = evaluate_model(
                model_name=model_name,
                feature_set=feature_set_name,
                target=target,
                y_true=y_test,
                y_pred=preds,
            )
            results.append(row)

            print(
                f"{model_name} | MAE={row['mae']:.3f} | "
                f"RMSE={row['rmse']:.3f} | R2={row['r2']:.3f}"
            )

        print()

    return results


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run src/features/build_context_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH)

    cgm_cols = get_cgm_feature_columns(df)
    context_cols = get_context_feature_columns(df)
    combined_cols = cgm_cols + context_cols

    train_df = df[df["split"] == "training"].copy()
    test_df = df[df["split"] == "testing"].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Training or testing split is empty.")

    print(f"Loaded dataset: {len(df):,} rows")
    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows:  {len(test_df):,}")
    print(f"CGM features: {len(cgm_cols)}")
    print(f"Context features: {len(context_cols)}")
    print(f"Combined features: {len(combined_cols)}")
    print()

    all_results = []

    # Re-run CGM-only from this same file to make a fair side-by-side comparison.
    all_results.extend(
        train_and_evaluate_feature_set(
            train_df=train_df,
            test_df=test_df,
            feature_cols=cgm_cols,
            feature_set_name="cgm_only",
        )
    )

    all_results.extend(
        train_and_evaluate_feature_set(
            train_df=train_df,
            test_df=test_df,
            feature_cols=combined_cols,
            feature_set_name="cgm_plus_context",
        )
    )

    results_df = pd.DataFrame(all_results)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    print(f"Saved context baseline results to {RESULTS_PATH}")
    print()
    print(results_df)


if __name__ == "__main__":
    main()