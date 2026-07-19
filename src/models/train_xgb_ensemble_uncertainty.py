from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor


DATA_PATH = Path("data/processed/cgm_lag_dataset.csv")
OUT_DIR = Path("reports")

METRICS_PATH = (
    OUT_DIR / "xgb_ensemble_uncertainty_results.csv"
)
PREDICTIONS_PATH = (
    OUT_DIR / "xgb_ensemble_uncertainty_predictions.csv"
)
BIN_SUMMARY_PATH = (
    OUT_DIR / "xgb_ensemble_uncertainty_bin_summary.csv"
)
SPLIT_SUMMARY_PATH = (
    OUT_DIR / "xgb_ensemble_uncertainty_split_summary.csv"
)

FORECAST_HORIZONS = [30, 60]

VALIDATION_FRACTION = 0.20
EMBARGO_MINUTES = 180

N_MODELS = 10
BOOTSTRAP_FRAC = 0.85
BASE_RANDOM_SEED = 42


def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    feature_cols = [
        col
        for col in df.columns
        if col.startswith("glucose_lag_")
        or col.startswith("glucose_change_")
        or col.startswith("glucose_rolling_")
    ]

    if not feature_cols:
        raise ValueError(
            "No CGM feature columns were found."
        )

    return feature_cols


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
            f"{column_name} contains invalid values."
        )

    return parsed.astype(bool)


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
        "source_file",
        "timestamp",
        "glucose_observed",
        "glucose_lag_0min",
        "target_glucose_30min",
        "target_glucose_60min",
        "eligible_30min",
        "eligible_60min",
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Dataset contains invalid timestamps."
        )

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

        df[eligibility_col] = parse_boolean_column(
            df[eligibility_col],
            eligibility_col,
        )

    duplicate_rows = df.duplicated(
        subset=[
            "patient_id",
            "split",
            "timestamp",
        ]
    )

    if duplicate_rows.any():
        raise ValueError(
            "Dataset contains duplicate "
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
    development = df[
        df["split"] == "training"
    ].copy()

    if development.empty:
        raise ValueError(
            "Original training split is empty."
        )

    embargo = pd.Timedelta(
        minutes=EMBARGO_MINUTES
    )

    train_groups = []
    validation_groups = []
    summary_rows = []

    for patient_id, group in development.groupby(
        "patient_id",
        sort=True,
    ):
        group = group.sort_values(
            "timestamp"
        ).copy()

        timestamps = (
            group["timestamp"]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        if len(timestamps) < 2:
            raise ValueError(
                f"Patient {patient_id} has too few "
                "timestamps for splitting."
            )

        validation_index = int(
            np.floor(
                len(timestamps)
                * (1.0 - VALIDATION_FRACTION)
            )
        )

        validation_index = min(
            max(validation_index, 1),
            len(timestamps) - 1,
        )

        validation_start = timestamps.iloc[
            validation_index
        ]

        training_end = (
            validation_start - embargo
        )

        patient_train = group[
            group["timestamp"] < training_end
        ].copy()

        patient_validation = group[
            group["timestamp"] >= validation_start
        ].copy()

        patient_embargo = group[
            (
                group["timestamp"]
                >= training_end
            )
            & (
                group["timestamp"]
                < validation_start
            )
        ]

        if patient_train.empty:
            raise ValueError(
                f"Patient {patient_id} has no "
                "training rows before the embargo."
            )

        if patient_validation.empty:
            raise ValueError(
                f"Patient {patient_id} has no "
                "validation rows."
            )

        patient_train[
            "model_split"
        ] = "train"

        patient_validation[
            "model_split"
        ] = "validation"

        train_groups.append(patient_train)
        validation_groups.append(
            patient_validation
        )

        summary_rows.append(
            {
                "patient_id": patient_id,
                "validation_start": (
                    validation_start
                ),
                "training_end_before_embargo": (
                    training_end
                ),
                "train_timeline_rows": len(
                    patient_train
                ),
                "embargoed_timeline_rows": len(
                    patient_embargo
                ),
                "validation_timeline_rows": len(
                    patient_validation
                ),
            }
        )

    development_split = pd.concat(
        train_groups + validation_groups,
        ignore_index=True,
    )

    split_summary = pd.DataFrame(
        summary_rows
    )

    return development_split, split_summary


def get_horizon_dataset(
    df: pd.DataFrame,
    horizon: int,
    feature_cols: list[str],
) -> pd.DataFrame:
    target_col = (
        f"target_glucose_{horizon}min"
    )
    eligibility_col = (
        f"eligible_{horizon}min"
    )

    horizon_df = df[
        df[eligibility_col]
    ].copy()

    required_cols = (
        feature_cols + [target_col]
    )

    complete_rows = horizon_df[
        required_cols
    ].notna().all(axis=1)

    if not complete_rows.all():
        n_bad = int(
            (~complete_rows).sum()
        )

        raise ValueError(
            f"{n_bad:,} rows marked eligible "
            f"for {horizon} minutes contain "
            "missing model inputs or targets."
        )

    return horizon_df


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    absolute_error = np.abs(
        y_true - y_pred
    )

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
        "median_absolute_error": float(
            np.median(absolute_error)
        ),
    }


def make_xgb_model(
    seed: int,
) -> XGBRegressor:
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


def build_prediction_frame(
    evaluation_df: pd.DataFrame,
    evaluation_split: str,
    horizon: int,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
) -> pd.DataFrame:
    target_col = (
        f"target_glucose_{horizon}min"
    )

    y_true = evaluation_df[
        target_col
    ].to_numpy()

    pred_df = evaluation_df[
        [
            "patient_id",
            "split",
            "source_file",
            "timestamp",
            "glucose_observed",
            "glucose_lag_0min",
        ]
    ].copy()

    pred_df[
        "evaluation_split"
    ] = evaluation_split

    pred_df[
        "horizon_minutes"
    ] = horizon

    pred_df["target"] = target_col
    pred_df["y_true"] = y_true
    pred_df["prediction_mean"] = pred_mean
    pred_df["prediction_std"] = pred_std

    pred_df["absolute_error"] = np.abs(
        y_true - pred_mean
    )

    pred_df["squared_error"] = np.square(
        y_true - pred_mean
    )

    return pred_df


def train_ensemble_for_horizon(
    train_df: pd.DataFrame,
    evaluation_frames: dict[str, pd.DataFrame],
    feature_cols: list[str],
    horizon: int,
) -> tuple[list[pd.DataFrame], list[dict]]:
    target_col = (
        f"target_glucose_{horizon}min"
    )

    ensemble_predictions = {
        split_name: []
        for split_name in evaluation_frames
    }

    for model_idx in range(N_MODELS):
        seed = (
            BASE_RANDOM_SEED + model_idx
        )

        bootstrap_df = train_df.sample(
            frac=BOOTSTRAP_FRAC,
            replace=True,
            random_state=seed,
        )

        X_train = bootstrap_df[
            feature_cols
        ]

        y_train = bootstrap_df[
            target_col
        ].to_numpy()

        model = make_xgb_model(seed)

        print(
            "Training XGBoost ensemble member "
            f"{model_idx + 1}/{N_MODELS} "
            f"for {horizon}-minute forecasting"
        )

        model.fit(
            X_train,
            y_train,
        )

        for (
            split_name,
            evaluation_df,
        ) in evaluation_frames.items():
            predictions = model.predict(
                evaluation_df[feature_cols]
            )

            ensemble_predictions[
                split_name
            ].append(predictions)

    prediction_frames = []
    metric_rows = []

    for (
        split_name,
        evaluation_df,
    ) in evaluation_frames.items():
        prediction_matrix = np.vstack(
            ensemble_predictions[split_name]
        )

        pred_mean = prediction_matrix.mean(
            axis=0
        )

        pred_std = prediction_matrix.std(
            axis=0
        )

        target_values = evaluation_df[
            target_col
        ].to_numpy()

        absolute_error = np.abs(
            target_values - pred_mean
        )

        metrics = regression_metrics(
            y_true=target_values,
            y_pred=pred_mean,
        )

        uncertainty_error_spearman = (
            pd.Series(pred_std).corr(
                pd.Series(absolute_error),
                method="spearman",
            )
        )

        metric_rows.append(
            {
                "model": "xgboost_ensemble",
                "horizon_minutes": horizon,
                "target": target_col,
                "evaluation_split": split_name,
                "n_rows": len(
                    evaluation_df
                ),
                "n_models": N_MODELS,
                "bootstrap_frac": (
                    BOOTSTRAP_FRAC
                ),
                **metrics,
                "uncertainty_error_spearman": (
                    uncertainty_error_spearman
                ),
            }
        )

        prediction_frames.append(
            build_prediction_frame(
                evaluation_df=(
                    evaluation_df
                ),
                evaluation_split=(
                    split_name
                ),
                horizon=horizon,
                pred_mean=pred_mean,
                pred_std=pred_std,
            )
        )

    return prediction_frames, metric_rows


def add_uncertainty_bins(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    predictions_df = predictions_df.copy()

    predictions_df[
        "uncertainty_bin"
    ] = pd.NA

    group_cols = [
        "horizon_minutes",
        "evaluation_split",
    ]

    labels = [
        "very_low",
        "low",
        "medium",
        "high",
        "very_high",
    ]

    for _, group in predictions_df.groupby(
        group_cols,
        sort=True,
    ):
        bins = pd.qcut(
            group["prediction_std"],
            q=5,
            labels=labels,
            duplicates="drop",
        )

        predictions_df.loc[
            group.index,
            "uncertainty_bin",
        ] = bins.astype(str)

    return predictions_df


def summarize_by_uncertainty_bin(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    return (
        predictions_df.groupby(
            [
                "horizon_minutes",
                "target",
                "evaluation_split",
                "uncertainty_bin",
            ],
            observed=True,
        )
        .agg(
            n=("absolute_error", "size"),
            mean_uncertainty=(
                "prediction_std",
                "mean",
            ),
            minimum_uncertainty=(
                "prediction_std",
                "min",
            ),
            maximum_uncertainty=(
                "prediction_std",
                "max",
            ),
            mae=(
                "absolute_error",
                "mean",
            ),
            median_absolute_error=(
                "absolute_error",
                "median",
            ),
            rmse=(
                "squared_error",
                lambda values: float(
                    np.sqrt(values.mean())
                ),
            ),
        )
        .reset_index()
    )


def print_metric_row(
    row: dict,
) -> None:
    print(
        f"{row['evaluation_split']:>10} | "
        f"N={row['n_rows']:>6,} | "
        f"MAE={row['mae']:.3f} | "
        f"RMSE={row['rmse']:.3f} | "
        f"R2={row['r2']:.3f} | "
        "uncertainty-error Spearman="
        f"{row['uncertainty_error_spearman']:.3f}"
    )


def main() -> None:
    df = load_dataset()
    feature_cols = get_feature_columns(df)

    development_df, split_summary_df = (
        assign_development_splits(df)
    )

    train_timeline = development_df[
        development_df["model_split"]
        == "train"
    ].copy()

    validation_timeline = development_df[
        development_df["model_split"]
        == "validation"
    ].copy()

    test_timeline = df[
        df["split"] == "testing"
    ].copy()

    if test_timeline.empty:
        raise ValueError(
            "Original testing split is empty."
        )

    print(
        f"Loaded dataset: {len(df):,} timeline rows"
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
        f"Validation fraction: "
        f"{VALIDATION_FRACTION:.0%}"
    )
    print(
        f"Boundary embargo: "
        f"{EMBARGO_MINUTES} minutes"
    )
    print(
        f"Features: {len(feature_cols)}"
    )
    print(
        f"Ensemble size: {N_MODELS}"
    )
    print()

    all_prediction_frames = []
    all_metric_rows = []

    for horizon in FORECAST_HORIZONS:
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

        test_df = get_horizon_dataset(
            df=test_timeline,
            horizon=horizon,
            feature_cols=feature_cols,
        )

        if (
            train_df.empty
            or validation_df.empty
            or test_df.empty
        ):
            raise ValueError(
                f"An eligible split is empty for "
                f"the {horizon}-minute horizon."
            )

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
            f"Test rows:       {len(test_df):,}"
        )
        print()

        prediction_frames, metric_rows = (
            train_ensemble_for_horizon(
                train_df=train_df,
                evaluation_frames={
                    "validation": validation_df,
                    "test": test_df,
                },
                feature_cols=feature_cols,
                horizon=horizon,
            )
        )

        all_prediction_frames.extend(
            prediction_frames
        )
        all_metric_rows.extend(
            metric_rows
        )

        print()

        for row in metric_rows:
            print_metric_row(row)

        print()

    predictions_df = pd.concat(
        all_prediction_frames,
        ignore_index=True,
    )

    predictions_df = add_uncertainty_bins(
        predictions_df
    )

    metrics_df = pd.DataFrame(
        all_metric_rows
    ).sort_values(
        [
            "horizon_minutes",
            "evaluation_split",
        ]
    )

    uncertainty_summary_df = (
        summarize_by_uncertainty_bin(
            predictions_df
        )
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    uncertainty_summary_df.to_csv(
        BIN_SUMMARY_PATH,
        index=False,
    )

    split_summary_df.to_csv(
        SPLIT_SUMMARY_PATH,
        index=False,
    )

    print(
        f"Saved predictions to "
        f"{PREDICTIONS_PATH}"
    )
    print(
        f"Saved metrics to "
        f"{METRICS_PATH}"
    )
    print(
        f"Saved uncertainty-bin summary to "
        f"{BIN_SUMMARY_PATH}"
    )
    print(
        f"Saved split summary to "
        f"{SPLIT_SUMMARY_PATH}"
    )

    print()
    print("Metrics:")
    print(
        metrics_df.to_string(index=False)
    )

    print()
    print("Uncertainty-bin summary:")
    print(
        uncertainty_summary_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()