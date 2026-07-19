from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS_PATH = Path(
    "reports/xgb_ensemble_uncertainty_predictions.csv"
)
OUT_PATH = Path(
    "reports/selective_prediction_results.csv"
)
THRESHOLDS_PATH = Path(
    "reports/selective_prediction_thresholds.csv"
)

COVERAGE_LEVELS = [
    1.00,
    0.90,
    0.80,
    0.70,
    0.60,
    0.50,
    0.40,
    0.30,
]


def compute_rmse(
    squared_errors: pd.Series,
) -> float:
    return float(
        np.sqrt(
            squared_errors.mean()
        )
    )


def validate_predictions(
    pred_df: pd.DataFrame,
) -> None:
    required_cols = {
        "horizon_minutes",
        "target",
        "evaluation_split",
        "prediction_std",
        "absolute_error",
        "squared_error",
    }

    missing = required_cols - set(
        pred_df.columns
    )

    if missing:
        raise ValueError(
            "Missing required columns from "
            f"predictions file: {missing}"
        )

    expected_splits = {
        "validation",
        "test",
    }

    found_splits = set(
        pred_df["evaluation_split"]
        .dropna()
        .unique()
    )

    missing_splits = (
        expected_splits - found_splits
    )

    if missing_splits:
        raise ValueError(
            "Predictions file is missing "
            f"evaluation splits: {missing_splits}"
        )

    numeric_cols = [
        "horizon_minutes",
        "prediction_std",
        "absolute_error",
        "squared_error",
    ]

    for col in numeric_cols:
        pred_df[col] = pd.to_numeric(
            pred_df[col],
            errors="coerce",
        )

    if pred_df[numeric_cols].isna().any().any():
        raise ValueError(
            "Predictions file contains invalid "
            "numeric values."
        )

    if (
        pred_df["prediction_std"] < 0
    ).any():
        raise ValueError(
            "Prediction uncertainty contains "
            "negative values."
        )

    if (
        pred_df["absolute_error"] < 0
    ).any():
        raise ValueError(
            "Absolute error contains "
            "negative values."
        )

    if (
        pred_df["squared_error"] < 0
    ).any():
        raise ValueError(
            "Squared error contains "
            "negative values."
        )


def derive_validation_thresholds(
    validation_df: pd.DataFrame,
    horizon: int,
    target: str,
) -> pd.DataFrame:
    if validation_df.empty:
        raise ValueError(
            "Validation prediction frame is empty "
            f"for horizon {horizon}."
        )

    uncertainty = (
        validation_df["prediction_std"]
        .sort_values()
        .reset_index(drop=True)
    )

    rows = []

    for requested_coverage in COVERAGE_LEVELS:
        n_keep = int(
            np.ceil(
                len(uncertainty)
                * requested_coverage
            )
        )

        n_keep = min(
            max(n_keep, 1),
            len(uncertainty),
        )

        if requested_coverage == 1.0:
            threshold = float("inf")
        else:
            threshold = float(
                uncertainty.iloc[n_keep - 1]
            )

        validation_kept = validation_df[
            validation_df["prediction_std"]
            <= threshold
        ].copy()

        achieved_validation_coverage = (
            len(validation_kept)
            / len(validation_df)
        )

        rows.append(
            {
                "horizon_minutes": horizon,
                "target": target,
                "requested_coverage": (
                    requested_coverage
                ),
                "uncertainty_threshold": threshold,
                "validation_total_predictions": (
                    len(validation_df)
                ),
                "validation_kept_predictions": (
                    len(validation_kept)
                ),
                "achieved_validation_coverage": (
                    achieved_validation_coverage
                ),
            }
        )

    return pd.DataFrame(rows)


def evaluate_threshold_on_split(
    evaluation_df: pd.DataFrame,
    evaluation_split: str,
    threshold_row: pd.Series,
) -> dict:
    threshold = float(
        threshold_row[
            "uncertainty_threshold"
        ]
    )

    kept = evaluation_df[
        evaluation_df["prediction_std"]
        <= threshold
    ].copy()

    rejected = evaluation_df[
        evaluation_df["prediction_std"]
        > threshold
    ].copy()

    total_predictions = len(
        evaluation_df
    )

    kept_predictions = len(kept)
    rejected_predictions = len(rejected)

    achieved_coverage = (
        kept_predictions
        / total_predictions
        if total_predictions
        else np.nan
    )

    if kept.empty:
        mae = np.nan
        median_absolute_error = np.nan
        rmse = np.nan
        mean_uncertainty_kept = np.nan
        max_uncertainty_kept = np.nan
    else:
        mae = float(
            kept["absolute_error"].mean()
        )

        median_absolute_error = float(
            kept["absolute_error"].median()
        )

        rmse = compute_rmse(
            kept["squared_error"]
        )

        mean_uncertainty_kept = float(
            kept["prediction_std"].mean()
        )

        max_uncertainty_kept = float(
            kept["prediction_std"].max()
        )

    return {
        "horizon_minutes": int(
            threshold_row["horizon_minutes"]
        ),
        "target": threshold_row["target"],
        "evaluation_split": evaluation_split,
        "requested_coverage": float(
            threshold_row[
                "requested_coverage"
            ]
        ),
        "uncertainty_threshold": threshold,
        "total_predictions": (
            total_predictions
        ),
        "kept_predictions": (
            kept_predictions
        ),
        "rejected_predictions": (
            rejected_predictions
        ),
        "achieved_coverage": (
            achieved_coverage
        ),
        "mean_uncertainty_kept": (
            mean_uncertainty_kept
        ),
        "max_uncertainty_kept": (
            max_uncertainty_kept
        ),
        "mae": mae,
        "median_absolute_error": (
            median_absolute_error
        ),
        "rmse": rmse,
    }


def run_selective_prediction_for_horizon(
    pred_df: pd.DataFrame,
    horizon: int,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon_df = pred_df[
        (
            pred_df["horizon_minutes"]
            == horizon
        )
        & (
            pred_df["target"]
            == target
        )
    ].copy()

    validation_df = horizon_df[
        horizon_df["evaluation_split"]
        == "validation"
    ].copy()

    test_df = horizon_df[
        horizon_df["evaluation_split"]
        == "test"
    ].copy()

    if validation_df.empty:
        raise ValueError(
            "No validation predictions found "
            f"for {horizon}-minute horizon."
        )

    if test_df.empty:
        raise ValueError(
            "No test predictions found "
            f"for {horizon}-minute horizon."
        )

    thresholds_df = derive_validation_thresholds(
        validation_df=validation_df,
        horizon=horizon,
        target=target,
    )

    result_rows = []

    for _, threshold_row in (
        thresholds_df.iterrows()
    ):
        validation_result = (
            evaluate_threshold_on_split(
                evaluation_df=validation_df,
                evaluation_split="validation",
                threshold_row=threshold_row,
            )
        )

        test_result = (
            evaluate_threshold_on_split(
                evaluation_df=test_df,
                evaluation_split="test",
                threshold_row=threshold_row,
            )
        )

        result_rows.extend(
            [
                validation_result,
                test_result,
            ]
        )

    return (
        pd.DataFrame(result_rows),
        thresholds_df,
    )


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PREDICTIONS_PATH}. "
            "Run "
            "src/models/"
            "train_xgb_ensemble_uncertainty.py "
            "first."
        )

    pred_df = pd.read_csv(
        PREDICTIONS_PATH
    )

    validate_predictions(pred_df)

    all_results = []
    all_thresholds = []

    horizon_targets = (
        pred_df[
            [
                "horizon_minutes",
                "target",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "horizon_minutes",
                "target",
            ]
        )
    )

    for _, row in (
        horizon_targets.iterrows()
    ):
        horizon = int(
            row["horizon_minutes"]
        )

        target = row["target"]

        print(
            "Running validation-calibrated "
            "selective prediction for "
            f"{horizon}-minute forecasting"
        )

        results_df, thresholds_df = (
            run_selective_prediction_for_horizon(
                pred_df=pred_df,
                horizon=horizon,
                target=target,
            )
        )

        all_results.append(
            results_df
        )

        all_thresholds.append(
            thresholds_df
        )

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    thresholds = pd.concat(
        all_thresholds,
        ignore_index=True,
    )

    results = results.sort_values(
        [
            "horizon_minutes",
            "requested_coverage",
            "evaluation_split",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    thresholds = thresholds.sort_values(
        [
            "horizon_minutes",
            "requested_coverage",
        ],
        ascending=[
            True,
            False,
        ],
    )

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUT_PATH,
        index=False,
    )

    thresholds.to_csv(
        THRESHOLDS_PATH,
        index=False,
    )

    print()
    print(
        "Saved selective prediction results "
        f"to {OUT_PATH}"
    )
    print(
        "Saved validation-derived thresholds "
        f"to {THRESHOLDS_PATH}"
    )

    print()
    print("Selective prediction results:")
    print(
        results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()