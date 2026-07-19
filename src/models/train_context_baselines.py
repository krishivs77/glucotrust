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
SPLIT_SUMMARY_PATH = OUT_DIR / "context_baseline_split_summary.csv"

FORECAST_HORIZONS = [30, 60]

VALIDATION_FRACTION = 0.20
EMBARGO_MINUTES = 180


def get_cgm_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = [
        column
        for column in df.columns
        if (
            column.startswith("glucose_lag_")
            or column.startswith("glucose_change_")
            or column.startswith("glucose_rolling_")
        )
    ]

    if not feature_columns:
        raise ValueError("No CGM feature columns were found.")

    return feature_columns


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

    feature_columns = [
        column
        for column in df.columns
        if column.startswith(prefixes)
    ]

    if not feature_columns:
        raise ValueError("No context feature columns were found.")

    return feature_columns


def parse_boolean_column(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    if series.dtype == bool:
        return series

    parsed = (
        series.astype(str)
        .str.strip()
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

    if parsed.isna().any():
        raise ValueError(
            f"{column_name} contains invalid Boolean values."
        )

    return parsed.astype(bool)


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. "
            "Run src/features/build_context_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "patient_id",
        "split",
        "timestamp",
        "glucose_lag_0min",
        "target_glucose_30min",
        "target_glucose_60min",
        "eligible_30min",
        "eligible_60min",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Context dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Context dataset contains invalid timestamps."
        )

    cgm_columns = get_cgm_feature_columns(df)
    context_columns = get_context_feature_columns(df)

    numeric_columns = list(
        dict.fromkeys(
            cgm_columns
            + context_columns
            + [
                "target_glucose_30min",
                "target_glucose_60min",
            ]
        )
    )

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    for horizon in FORECAST_HORIZONS:
        eligibility_column = f"eligible_{horizon}min"

        df[eligibility_column] = parse_boolean_column(
            df[eligibility_column],
            eligibility_column,
        )

    duplicate_rows = df.duplicated(
        subset=[
            "patient_id",
            "split",
            "timestamp",
        ],
        keep=False,
    )

    if duplicate_rows.any():
        raise ValueError(
            "Context dataset contains duplicate "
            "patient/split/timestamp rows."
        )

    return df.sort_values(
        [
            "patient_id",
            "split",
            "timestamp",
        ]
    ).reset_index(drop=True)


def assign_development_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development_df = df[
        df["split"] == "training"
    ].copy()

    if development_df.empty:
        raise ValueError(
            "Original training split is empty."
        )

    embargo = pd.Timedelta(
        minutes=EMBARGO_MINUTES
    )

    training_groups = []
    validation_groups = []
    summary_rows = []

    for patient_id, patient_df in development_df.groupby(
        "patient_id",
        sort=True,
    ):
        patient_df = patient_df.sort_values(
            "timestamp"
        ).copy()

        unique_timestamps = (
            patient_df["timestamp"]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        if len(unique_timestamps) < 2:
            raise ValueError(
                f"Patient {patient_id} has too few timestamps "
                "for a chronological validation split."
            )

        validation_index = int(
            np.floor(
                len(unique_timestamps)
                * (1.0 - VALIDATION_FRACTION)
            )
        )

        validation_index = min(
            max(validation_index, 1),
            len(unique_timestamps) - 1,
        )

        validation_start = unique_timestamps.iloc[
            validation_index
        ]

        training_end = validation_start - embargo

        patient_training = patient_df[
            patient_df["timestamp"] < training_end
        ].copy()

        patient_validation = patient_df[
            patient_df["timestamp"] >= validation_start
        ].copy()

        patient_embargo = patient_df[
            (
                patient_df["timestamp"] >= training_end
            )
            & (
                patient_df["timestamp"] < validation_start
            )
        ]

        if patient_training.empty:
            raise ValueError(
                f"Patient {patient_id} has no rows before "
                "the training/validation embargo."
            )

        if patient_validation.empty:
            raise ValueError(
                f"Patient {patient_id} has no validation rows."
            )

        patient_training["model_split"] = "train"
        patient_validation["model_split"] = "validation"

        training_groups.append(patient_training)
        validation_groups.append(patient_validation)

        summary_rows.append(
            {
                "patient_id": patient_id,
                "validation_start": validation_start,
                "training_end_before_embargo": training_end,
                "train_timeline_rows": len(patient_training),
                "embargoed_timeline_rows": len(patient_embargo),
                "validation_timeline_rows": len(patient_validation),
            }
        )

    split_df = pd.concat(
        training_groups + validation_groups,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(summary_rows)

    return split_df, summary_df


def get_horizon_dataset(
    df: pd.DataFrame,
    horizon: int,
    feature_columns: list[str],
) -> pd.DataFrame:
    target_column = f"target_glucose_{horizon}min"
    eligibility_column = f"eligible_{horizon}min"

    horizon_df = df[
        df[eligibility_column]
    ].copy()

    required_model_columns = feature_columns + [
        target_column
    ]

    complete_rows = horizon_df[
        required_model_columns
    ].notna().all(axis=1)

    if not complete_rows.all():
        invalid_count = int(
            (~complete_rows).sum()
        )

        raise ValueError(
            f"{invalid_count:,} rows marked eligible for "
            f"{horizon}-minute forecasting contain missing "
            "features or targets."
        )

    return horizon_df


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            mean_squared_error(
                y_true,
                y_pred,
            )
            ** 0.5
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
    }


def build_models() -> dict[str, object]:
    return {
        "ridge": Pipeline(
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
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_predictions(
    model_name: str,
    feature_set: str,
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
        "feature_set": feature_set,
        "horizon_minutes": horizon,
        "target": f"target_glucose_{horizon}min",
        "evaluation_split": evaluation_split,
        "n_rows": len(y_true),
        **metrics,
    }


def train_and_evaluate_feature_set(
    train_timeline: pd.DataFrame,
    validation_timeline: pd.DataFrame,
    test_timeline: pd.DataFrame,
    feature_columns: list[str],
    feature_set_name: str,
) -> list[dict]:
    result_rows = []

    print("=" * 88)
    print(f"Feature set: {feature_set_name}")
    print(f"Features: {len(feature_columns)}")
    print("=" * 88)
    print()

    for horizon in FORECAST_HORIZONS:
        target_column = f"target_glucose_{horizon}min"

        train_df = get_horizon_dataset(
            df=train_timeline,
            horizon=horizon,
            feature_columns=feature_columns,
        )

        validation_df = get_horizon_dataset(
            df=validation_timeline,
            horizon=horizon,
            feature_columns=feature_columns,
        )

        test_df = get_horizon_dataset(
            df=test_timeline,
            horizon=horizon,
            feature_columns=feature_columns,
        )

        if (
            train_df.empty
            or validation_df.empty
            or test_df.empty
        ):
            raise ValueError(
                "An eligible split is empty for "
                f"{feature_set_name}, {horizon}-minute forecasting."
            )

        evaluation_frames = {
            "validation": validation_df,
            "test": test_df,
        }

        print("-" * 88)
        print(
            f"{feature_set_name} | "
            f"{horizon}-minute forecast"
        )
        print("-" * 88)
        print(
            f"Train rows:      {len(train_df):,}"
        )
        print(
            f"Validation rows: {len(validation_df):,}"
        )
        print(
            f"Test rows:       {len(test_df):,}"
        )
        print()

        for split_name, evaluation_df in evaluation_frames.items():
            persistence_prediction = evaluation_df[
                "glucose_lag_0min"
            ].to_numpy()

            row = evaluate_predictions(
                model_name="persistence",
                feature_set=feature_set_name,
                horizon=horizon,
                evaluation_split=split_name,
                y_true=evaluation_df[target_column].to_numpy(),
                y_pred=persistence_prediction,
            )

            result_rows.append(row)

            print(
                f"persistence | {split_name:>10} | "
                f"MAE={row['mae']:.3f} | "
                f"RMSE={row['rmse']:.3f} | "
                f"R2={row['r2']:.3f}"
            )

        models = build_models()

        X_train = train_df[
            feature_columns
        ]

        y_train = train_df[
            target_column
        ].to_numpy()

        for model_name, model in models.items():
            print(
                f"Training {model_name}..."
            )

            model.fit(
                X_train,
                y_train,
            )

            for split_name, evaluation_df in evaluation_frames.items():
                predictions = model.predict(
                    evaluation_df[
                        feature_columns
                    ]
                )

                row = evaluate_predictions(
                    model_name=model_name,
                    feature_set=feature_set_name,
                    horizon=horizon,
                    evaluation_split=split_name,
                    y_true=evaluation_df[
                        target_column
                    ].to_numpy(),
                    y_pred=predictions,
                )

                result_rows.append(row)

                print(
                    f"{model_name:>13} | "
                    f"{split_name:>10} | "
                    f"MAE={row['mae']:.3f} | "
                    f"RMSE={row['rmse']:.3f} | "
                    f"R2={row['r2']:.3f}"
                )

        print()

    return result_rows


def main() -> None:
    df = load_dataset()

    cgm_columns = get_cgm_feature_columns(df)
    context_columns = get_context_feature_columns(df)
    combined_columns = cgm_columns + context_columns

    development_df, split_summary_df = (
        assign_development_splits(df)
    )

    train_timeline = development_df[
        development_df["model_split"] == "train"
    ].copy()

    validation_timeline = development_df[
        development_df["model_split"] == "validation"
    ].copy()

    test_timeline = df[
        df["split"] == "testing"
    ].copy()

    if test_timeline.empty:
        raise ValueError(
            "Original held-out testing split is empty."
        )

    print(
        f"Loaded context dataset: {len(df):,} timeline rows"
    )
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
        f"{len(test_timeline):,}"
    )
    print(
        f"Validation fraction: {VALIDATION_FRACTION:.0%}"
    )
    print(
        f"Boundary embargo: {EMBARGO_MINUTES} minutes"
    )
    print(
        f"CGM features: {len(cgm_columns)}"
    )
    print(
        f"Context features: {len(context_columns)}"
    )
    print(
        f"Combined features: {len(combined_columns)}"
    )
    print()

    all_results = []

    all_results.extend(
        train_and_evaluate_feature_set(
            train_timeline=train_timeline,
            validation_timeline=validation_timeline,
            test_timeline=test_timeline,
            feature_columns=cgm_columns,
            feature_set_name="cgm_only",
        )
    )

    all_results.extend(
        train_and_evaluate_feature_set(
            train_timeline=train_timeline,
            validation_timeline=validation_timeline,
            test_timeline=test_timeline,
            feature_columns=combined_columns,
            feature_set_name="cgm_plus_context",
        )
    )

    results_df = pd.DataFrame(
        all_results
    ).sort_values(
        [
            "horizon_minutes",
            "evaluation_split",
            "feature_set",
            "model",
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

    print(
        "Saved repaired context baseline results to "
        f"{RESULTS_PATH}"
    )
    print(
        "Saved context baseline split summary to "
        f"{SPLIT_SUMMARY_PATH}"
    )
    print()
    print(
        results_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()