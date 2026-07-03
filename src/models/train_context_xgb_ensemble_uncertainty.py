from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


DATA_PATH = Path("data/processed/cgm_context_dataset.csv")
OUT_DIR = Path("reports")

METRICS_PATH = OUT_DIR / "context_xgb_ensemble_uncertainty_results.csv"
PREDICTIONS_PATH = OUT_DIR / "context_xgb_ensemble_uncertainty_predictions.csv"
BIN_SUMMARY_PATH = OUT_DIR / "context_xgb_ensemble_uncertainty_bin_summary.csv"

TARGETS = ["target_glucose_30min", "target_glucose_60min"]

N_MODELS = 10
BOOTSTRAP_FRAC = 0.85
BASE_RANDOM_SEED = 42


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_cols = []

    for col in df.columns:
        if (
            col.startswith("glucose_lag_")
            or col.startswith("glucose_change_")
            or col.startswith("glucose_rolling_")
            or col.startswith("carbs_")
            or col.startswith("meal_")
            or col.startswith("bolus_")
            or col.startswith("bolus_carb_input_")
            or col.startswith("time_since_")
            or col.startswith("has_prior_")
            or col.startswith("basal_")
        ):
            feature_cols.append(col)

    return feature_cols


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred),
    }


def make_xgb_model(seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
    )


def train_ensemble_for_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> tuple[pd.DataFrame, dict]:
    X_test = test_df[feature_cols]
    y_test = test_df[target].to_numpy()

    all_predictions = []

    for model_idx in range(N_MODELS):
        seed = BASE_RANDOM_SEED + model_idx

        boot_df = train_df.sample(
            frac=BOOTSTRAP_FRAC,
            replace=True,
            random_state=seed,
        )

        X_train = boot_df[feature_cols]
        y_train = boot_df[target].to_numpy()

        model = make_xgb_model(seed)

        print(f"Training context XGBoost ensemble member {model_idx + 1}/{N_MODELS} for {target}")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        all_predictions.append(preds)

    predictions = np.vstack(all_predictions)

    pred_mean = predictions.mean(axis=0)
    pred_std = predictions.std(axis=0)

    abs_error = np.abs(y_test - pred_mean)

    metrics = regression_metrics(y_test, pred_mean)
    metrics["uncertainty_error_spearman"] = pd.Series(pred_std).corr(
        pd.Series(abs_error),
        method="spearman",
    )

    pred_df = test_df[
        [
            "patient_id",
            "split",
            "source_file",
            "timestamp",
            "glucose",
        ]
    ].copy()

    pred_df["target"] = target
    pred_df["y_true"] = y_test
    pred_df["prediction_mean"] = pred_mean
    pred_df["prediction_std"] = pred_std
    pred_df["absolute_error"] = abs_error

    pred_df["uncertainty_bin"] = pd.qcut(
        pred_df["prediction_std"],
        q=5,
        labels=["very_low", "low", "medium", "high", "very_high"],
        duplicates="drop",
    )

    return pred_df, metrics


def summarize_by_uncertainty_bin(pred_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        pred_df.groupby(["target", "uncertainty_bin"], observed=True)
        .agg(
            n=("absolute_error", "size"),
            mean_uncertainty=("prediction_std", "mean"),
            mae=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            rmse=("absolute_error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        )
        .reset_index()
    )

    return summary


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run src/features/build_context_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    feature_cols = get_feature_columns(df)

    train_df = df[df["split"] == "training"].copy()
    test_df = df[df["split"] == "testing"].copy()

    print(f"Loaded context dataset: {len(df):,} rows")
    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows:  {len(test_df):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Ensemble size: {N_MODELS}")
    print()

    all_pred_dfs = []
    metric_rows = []

    for target in TARGETS:
        print("=" * 80)
        print(f"Target: {target}")
        print("=" * 80)

        pred_df, metrics = train_ensemble_for_target(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target=target,
        )

        all_pred_dfs.append(pred_df)

        metric_row = {
            "model": "context_xgboost_ensemble",
            "feature_set": "cgm_plus_context",
            "target": target,
            "n_models": N_MODELS,
            "bootstrap_frac": BOOTSTRAP_FRAC,
            **metrics,
        }
        metric_rows.append(metric_row)

        print(
            f"context_xgboost_ensemble | MAE={metrics['mae']:.3f} | "
            f"RMSE={metrics['rmse']:.3f} | R2={metrics['r2']:.3f} | "
            f"uncertainty-error Spearman={metrics['uncertainty_error_spearman']:.3f}"
        )
        print()

    predictions_df = pd.concat(all_pred_dfs, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    uncertainty_summary_df = summarize_by_uncertainty_bin(predictions_df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    metrics_df.to_csv(METRICS_PATH, index=False)
    uncertainty_summary_df.to_csv(BIN_SUMMARY_PATH, index=False)

    print(f"Saved predictions to {PREDICTIONS_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved uncertainty bin summary to {BIN_SUMMARY_PATH}")
    print()
    print("Metrics:")
    print(metrics_df)
    print()
    print("Uncertainty bin summary:")
    print(uncertainty_summary_df)


if __name__ == "__main__":
    main()