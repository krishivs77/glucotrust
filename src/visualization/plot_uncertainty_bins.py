from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_PATH = Path(
    "reports/xgb_ensemble_uncertainty_bin_summary.csv"
)
FIGURES_DIR = Path("reports/figures")
OUT_PATH = (
    FIGURES_DIR
    / "xgb_uncertainty_bins_test_mae.png"
)

HORIZON_LABELS = {
    30: "30-minute forecast",
    60: "60-minute forecast",
}

UNCERTAINTY_ORDER = [
    "very_low",
    "low",
    "medium",
    "high",
    "very_high",
]

UNCERTAINTY_LABELS = {
    "very_low": "Very low",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "very_high": "Very high",
}


def load_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SUMMARY_PATH}. "
            "Run "
            "src/models/train_xgb_ensemble_uncertainty.py "
            "first."
        )

    summary = pd.read_csv(SUMMARY_PATH)

    required_columns = {
        "horizon_minutes",
        "evaluation_split",
        "uncertainty_bin",
        "mae",
    }

    missing_columns = (
        required_columns - set(summary.columns)
    )

    if missing_columns:
        raise ValueError(
            "Uncertainty-bin summary is missing "
            "required columns: "
            f"{sorted(missing_columns)}"
        )

    test_summary = summary[
        summary["evaluation_split"] == "test"
    ].copy()

    if test_summary.empty:
        raise ValueError(
            "Uncertainty-bin summary contains no "
            "held-out test rows."
        )

    unknown_horizons = (
        set(
            test_summary[
                "horizon_minutes"
            ].unique()
        )
        - set(HORIZON_LABELS)
    )

    if unknown_horizons:
        raise ValueError(
            "Unexpected forecast horizons found: "
            f"{sorted(unknown_horizons)}"
        )

    unknown_bins = (
        set(
            test_summary[
                "uncertainty_bin"
            ].unique()
        )
        - set(UNCERTAINTY_ORDER)
    )

    if unknown_bins:
        raise ValueError(
            "Unexpected uncertainty bins found: "
            f"{sorted(unknown_bins)}"
        )

    duplicate_rows = test_summary.duplicated(
        subset=[
            "horizon_minutes",
            "uncertainty_bin",
        ],
        keep=False,
    )

    if duplicate_rows.any():
        raise ValueError(
            "Multiple held-out test rows were found "
            "for the same forecast horizon and "
            "uncertainty bin."
        )

    expected_pairs = {
        (horizon, uncertainty_bin)
        for horizon in HORIZON_LABELS
        for uncertainty_bin in UNCERTAINTY_ORDER
    }

    observed_pairs = set(
        zip(
            test_summary["horizon_minutes"],
            test_summary["uncertainty_bin"],
        )
    )

    missing_pairs = expected_pairs - observed_pairs

    if missing_pairs:
        raise ValueError(
            "Missing held-out test uncertainty bins: "
            f"{sorted(missing_pairs)}"
        )

    return test_summary


def main() -> None:
    summary = load_summary()

    summary["uncertainty_bin"] = pd.Categorical(
        summary["uncertainty_bin"],
        categories=UNCERTAINTY_ORDER,
        ordered=True,
    )

    summary["uncertainty_label"] = (
        summary["uncertainty_bin"]
        .astype("object")
        .map(UNCERTAINTY_LABELS)
    )

    summary["horizon_label"] = (
        summary["horizon_minutes"]
        .map(HORIZON_LABELS)
    )

    label_order = [
        UNCERTAINTY_LABELS[uncertainty_bin]
        for uncertainty_bin in UNCERTAINTY_ORDER
    ]

    horizon_order = [
        HORIZON_LABELS[horizon]
        for horizon in sorted(HORIZON_LABELS)
    ]

    pivot = (
        summary.pivot(
            index="uncertainty_label",
            columns="horizon_label",
            values="mae",
        )
        .loc[
            label_order,
            horizon_order,
        ]
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(9, 5),
    )

    ax.set_title(
        "Forecast error rises with ensemble uncertainty "
        "on the held-out test set"
    )
    ax.set_xlabel("Ensemble uncertainty bin")
    ax.set_ylabel("Test MAE (mg/dL)")
    ax.legend(title="Forecast horizon")
    ax.grid(axis="y", alpha=0.3)

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
            padding=3,
            fontsize=9,
        )

    plt.xticks(rotation=0)
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