from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS_PATH = Path("reports/xgb_ensemble_uncertainty_predictions.csv")
OUT_PATH = Path("reports/selective_prediction_results.csv")

COVERAGE_LEVELS = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]


def compute_rmse(errors: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def selective_prediction_for_target(pred_df: pd.DataFrame, target: str) -> pd.DataFrame:
    target_df = pred_df[pred_df["target"] == target].copy()

    if target_df.empty:
        raise ValueError(f"No predictions found for target: {target}")

    # Lower prediction_std = more confident.
    target_df = target_df.sort_values("prediction_std", ascending=True).reset_index(drop=True)

    rows = []

    for coverage in COVERAGE_LEVELS:
        n_keep = int(round(len(target_df) * coverage))
        n_keep = max(n_keep, 1)

        kept = target_df.iloc[:n_keep].copy()
        rejected = target_df.iloc[n_keep:].copy()

        row = {
            "target": target,
            "coverage": coverage,
            "kept_predictions": len(kept),
            "rejected_predictions": len(rejected),
            "mean_uncertainty_kept": kept["prediction_std"].mean(),
            "max_uncertainty_kept": kept["prediction_std"].max(),
            "mae": kept["absolute_error"].mean(),
            "median_absolute_error": kept["absolute_error"].median(),
            "rmse": compute_rmse(kept["absolute_error"]),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PREDICTIONS_PATH}. Run src/models/train_xgb_ensemble_uncertainty.py first."
        )

    pred_df = pd.read_csv(PREDICTIONS_PATH)

    required_cols = {
        "target",
        "prediction_std",
        "absolute_error",
    }
    missing = required_cols - set(pred_df.columns)
    if missing:
        raise ValueError(f"Missing required columns from predictions file: {missing}")

    all_results = []

    for target in sorted(pred_df["target"].unique()):
        print(f"Running selective prediction analysis for {target}")
        all_results.append(selective_prediction_for_target(pred_df, target))

    results = pd.concat(all_results, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_PATH, index=False)

    print()
    print(f"Saved selective prediction results to {OUT_PATH}")
    print()
    print(results)


if __name__ == "__main__":
    main()