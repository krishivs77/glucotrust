from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor


DATA_PATH = Path("data/processed/cgm_lag_dataset.csv")
OUT_DIR = Path("reports")
RESULTS_PATH = OUT_DIR / "cgm_baseline_results.csv"
SPLIT_SUMMARY_PATH = OUT_DIR / "cgm_baseline_split_summary.csv"

FORECAST_HORIZONS = [30, 60]

VALIDATION_FRACTION = 0.20

# The model uses up to 120 minutes of history and predicts up to
# 60 minutes into the future. A conservative 180-minute embargo
# prevents adjacent training and validation windows from sharing
# information across the boundary.
EMBARGO_MINUTES = 180

RANDOM_STATE = 42


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_cols = [
        col
        for col in df.columns
        if col.startswith("glucose_lag_")
        or col.startswith("glucose_change_")
        or col.startswith("glucose_rolling_")
    ]

    if not feature_cols:
        raise ValueError("No CGM feature columns were found.")

    return feature_cols


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. "
            "Run src/features/build_cgm_lag_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required_cols = {
        "patient_id",
        "split",
        "timestamp",
        "glucose_lag_0min",
        "target_glucose_30min",
        "target_glucose_60min",
        "eligible_30min",
        "eligible_60min",
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns from lag dataset: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError("Dataset contains invalid timestamps.")

    numeric_cols = [
        col
        for col in df.columns
        if col.startswith("glucose_")
        or col.startswith("target_glucose_")
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    for horizon in FORECAST_HORIZONS:
        eligibility_col = f"eligible_{horizon}min"

        if df[eligibility_col].dtype != bool:
            df[eligibility_col] = (
                df[eligibility_col]
                .astype(str)
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                    }
                )
            )

        if df[eligibility_col].isna().any():
            raise ValueError(
                f"{eligibility_col} contains invalid values."
            )

        df[eligibility_col] = df[eligibility_col].astype(bool)

    duplicate_rows = df.duplicated(
        subset=["patient_id", "split", "timestamp"]
    )

    if duplicate_rows.any():
        raise ValueError(
            "Dataset contains duplicate patient/split/timestamp rows."
        )

    return df.sort_values(
        ["patient_id", "split", "timestamp"]
    ).reset_index(drop=True)


def assign_development_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the original training data chronologically per patient.

    For each patient:

    - the earliest portion becomes model training data;
    - an embargoed interval is excluded;
    - the final 20% becomes validation data.

    The original testing split is never used to define this boundary.
    """

    development = df[df["split"] == "training"].copy()

    if development.empty:
        raise ValueError("Original training split is empty.")

    embargo = pd.Timedelta(minutes=EMBARGO_MINUTES)

    train_groups = []
    validation_groups = []
    summary_rows = []

    for patient_id, group in development.groupby(
        "patient_id",
        sort=True,
    ):
        group = group.sort_values("timestamp").copy()

        unique_timestamps = (
            group["timestamp"]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        if len(unique_timestamps) < 2:
            raise ValueError(
                f"Patient {patient_id} has too few timestamps "
                "for a chronological split."
            )

        validation_start_index = int(
            np.floor(
                len(unique_timestamps)
                * (1.0 - VALIDATION_FRACTION)
            )
        )

        validation_start_index = min(
            max(validation_start_index, 1),
            len(unique_timestamps) - 1,
        )

        validation_start = unique_timestamps.iloc[
            validation_start_index
        ]

        training_end = validation_start - embargo

        patient_train = group[
            group["timestamp"] < training_end
        ].copy()

        patient_validation = group[
            group["timestamp"] >= validation_start
        ].copy()

        embargoed_rows = group[
            (group["timestamp"] >= training_end)
            & (group["timestamp"] < validation_start)
        ]

        if patient_train.empty:
            raise ValueError(
                f"Patient {patient_id} has no rows before the "
                "training/validation embargo."
            )

        if patient_validation.empty:
            raise ValueError(
                f"Patient {patient_id} has no validation rows."
            )

        patient_train["model_split"] = "train"
        patient_validation["model_split"] = "validation"

        train_groups.append(patient_train)
        validation_groups.append(patient_validation)

        summary_rows.append(
            {
                "patient_id": patient_id,
                "validation_start": validation_start,
                "training_end_before_embargo": training_end,
                "train_timeline_rows": len(patient_train),
                "embargoed_timeline_rows": len(embargoed_rows),
                "validation_timeline_rows": len(
                    patient_validation
                ),
            }
        )

    train_df = pd.concat(
        train_groups,
        ignore_index=True,
    )

    validation_df = pd.concat(
        validation_groups,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(summary_rows)

    return (
        pd.concat(
            [train_df, validation_df],
            ignore_index=True,
        ),
        summary_df,
    )


def get_horizon_dataset(
    df: pd.DataFrame,
    horizon: int,
    feature_cols: list[str],
) -> pd.DataFrame:
    target_col = f"target_glucose_{horizon}min"
    eligibility_col = f"eligible_{horizon}min"

    horizon_df = df[
        df[eligibility_col]
    ].copy()

    required_model_cols = feature_cols + [target_col]

    complete_rows = horizon_df[
        required_model_cols
    ].notna().all(axis=1)

    if not complete_rows.all():
        unexpected_missing = int(
            (~complete_rows).sum()
        )

        raise ValueError(
            f"Found {unexpected_missing:,} rows marked eligible "
            f"for {horizon} minutes that still contain missing "
            "features or targets."
        )

    return horizon_df


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "rmse": float(
            mean_squared_error(y_true, y_pred) ** 0.5
        ),
        "r2": float(
            r2_score(y_true, y_pred)
        ),
    }


def evaluate_predictions(
    model_name: str,
    horizon: int,
    evaluation_split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    metrics = regression_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )

    return {
        "model": model_name,
        "horizon_minutes": horizon,
        "evaluation_split": evaluation_split,
        "n_rows": len(y_true),
        **metrics,
    }


def build_models() -> dict:
    ridge = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(alpha=1.0),
            ),
        ]
    )

    random_forest = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    xgboost = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return {
        "ridge": ridge,
        "random_forest": random_forest,
        "xgboost": xgboost,
    }


def print_result(row: dict) -> None:
    print(
        f"{row['model']:>14} | "
        f"{row['evaluation_split']:>10} | "
        f"N={row['n_rows']:>6,} | "
        f"MAE={row['mae']:.3f} | "
        f"RMSE={row['rmse']:.3f} | "
        f"R2={row['r2']:.3f}"
    )


def main() -> None:
    df = load_dataset()
    feature_cols = get_feature_columns(df)

    development_df, split_summary_df = (
        assign_development_splits(df)
    )

    test_df = df[
        df["split"] == "testing"
    ].copy()

    if test_df.empty:
        raise ValueError("Original testing split is empty.")

    train_timeline = development_df[
        development_df["model_split"] == "train"
    ]

    validation_timeline = development_df[
        development_df["model_split"] == "validation"
    ]

    print(f"Loaded dataset: {len(df):,} timeline rows")
    print(
        "Development training timeline rows: "
        f"{len(train_timeline):,}"
    )
    print(
        "Development validation timeline rows: "
        f"{len(validation_timeline):,}"
    )
    print(
        "Original held-out testing timeline rows: "
        f"{len(test_df):,}"
    )
    print(
        f"Validation fraction: {VALIDATION_FRACTION:.0%}"
    )
    print(
        f"Boundary embargo: {EMBARGO_MINUTES} minutes"
    )
    print(f"Features: {len(feature_cols)}")
    print()

    results = []

    for horizon in FORECAST_HORIZONS:
        target_col = (
            f"target_glucose_{horizon}min"
        )

        train_df = get_horizon_dataset(
            df=train_timeline,
            horizon=horizon,
            feature_cols=feature_cols,
        )

        validation_df = get_horizon_dataset(
            df=validation_timeline,
            horizon=horizon,
            feature_cols=feature_cols,
        )

        horizon_test_df = get_horizon_dataset(
            df=test_df,
            horizon=horizon,
            feature_cols=feature_cols,
        )

        if (
            train_df.empty
            or validation_df.empty
            or horizon_test_df.empty
        ):
            raise ValueError(
                f"An eligible split is empty for the "
                f"{horizon}-minute horizon."
            )

        X_train = train_df[feature_cols]
        y_train = train_df[target_col].to_numpy()

        evaluation_sets = {
            "validation": (
                validation_df[feature_cols],
                validation_df[target_col].to_numpy(),
                validation_df[
                    "glucose_lag_0min"
                ].to_numpy(),
            ),
            "test": (
                horizon_test_df[feature_cols],
                horizon_test_df[
                    target_col
                ].to_numpy(),
                horizon_test_df[
                    "glucose_lag_0min"
                ].to_numpy(),
            ),
        }

        print("=" * 88)
        print(
            f"{horizon}-minute forecast"
        )
        print("=" * 88)
        print(
            f"Train rows:      {len(train_df):,}"
        )
        print(
            f"Validation rows: {len(validation_df):,}"
        )
        print(
            f"Test rows:       {len(horizon_test_df):,}"
        )
        print()

        for (
            evaluation_split,
            (
                _X_evaluation,
                y_evaluation,
                persistence_predictions,
            ),
        ) in evaluation_sets.items():
            row = evaluate_predictions(
                model_name="persistence",
                horizon=horizon,
                evaluation_split=evaluation_split,
                y_true=y_evaluation,
                y_pred=persistence_predictions,
            )

            results.append(row)
            print_result(row)

        models = build_models()

        for model_name, model in models.items():
            print()
            print(
                f"Training {model_name} "
                f"for {horizon}-minute forecasting..."
            )

            model.fit(
                X_train,
                y_train,
            )

            for (
                evaluation_split,
                (
                    X_evaluation,
                    y_evaluation,
                    _persistence_predictions,
                ),
            ) in evaluation_sets.items():
                predictions = model.predict(
                    X_evaluation
                )

                row = evaluate_predictions(
                    model_name=model_name,
                    horizon=horizon,
                    evaluation_split=evaluation_split,
                    y_true=y_evaluation,
                    y_pred=predictions,
                )

                results.append(row)
                print_result(row)

        print()

    results_df = pd.DataFrame(results).sort_values(
        [
            "horizon_minutes",
            "evaluation_split",
            "mae",
        ]
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    split_summary_df.to_csv(
        SPLIT_SUMMARY_PATH,
        index=False,
    )

    print(f"Saved results to {RESULTS_PATH}")
    print(
        "Saved split summary to "
        f"{SPLIT_SUMMARY_PATH}"
    )
    print()
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()