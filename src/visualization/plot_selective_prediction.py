from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("reports/selective_prediction_results.csv")
FIGURES_DIR = Path("reports/figures")
OUT_PATH = FIGURES_DIR / "selective_prediction_test_rmse.png"

HORIZON_LABELS = {
    30: "30-minute forecast",
    60: "60-minute forecast",
}


def load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RESULTS_PATH}. "
            "Run src/evaluation/selective_prediction.py first."
        )

    results = pd.read_csv(RESULTS_PATH)

    required_columns = {
        "horizon_minutes",
        "evaluation_split",
        "requested_coverage",
        "achieved_coverage",
        "rmse",
    }

    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            "Selective prediction results are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    test_results = results[
        results["evaluation_split"] == "test"
    ].copy()

    if test_results.empty:
        raise ValueError(
            "Selective prediction results contain no held-out test rows."
        )

    unknown_horizons = (
        set(test_results["horizon_minutes"].unique())
        - set(HORIZON_LABELS)
    )

    if unknown_horizons:
        raise ValueError(
            "Unexpected forecast horizons found: "
            f"{sorted(unknown_horizons)}"
        )

    duplicate_rows = test_results.duplicated(
        subset=[
            "horizon_minutes",
            "requested_coverage",
        ],
        keep=False,
    )

    if duplicate_rows.any():
        raise ValueError(
            "Multiple held-out test rows were found for the same "
            "forecast horizon and requested coverage."
        )

    return test_results


def main() -> None:
    results = load_results()

    fig, ax = plt.subplots(figsize=(8, 5))

    for horizon in sorted(HORIZON_LABELS):
        horizon_results = results[
            results["horizon_minutes"] == horizon
        ].sort_values(
            "achieved_coverage",
            ascending=False,
        )

        if horizon_results.empty:
            raise ValueError(
                f"No held-out test results found for the "
                f"{horizon}-minute forecast."
            )

        ax.plot(
            horizon_results["achieved_coverage"] * 100,
            horizon_results["rmse"],
            marker="o",
            linewidth=2,
            label=HORIZON_LABELS[horizon],
        )

    ax.set_title(
        "Selective prediction performance on the held-out test set"
    )
    ax.set_xlabel("Achieved coverage retained (%)")
    ax.set_ylabel("RMSE on retained predictions (mg/dL)")
    ax.grid(alpha=0.3)
    ax.legend(title="Forecast horizon")

    ax.invert_xaxis()

    plt.tight_layout()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()