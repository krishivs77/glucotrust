from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("reports/cgm_baseline_results.csv")
FIGURES_DIR = Path("reports/figures")
OUT_PATH = FIGURES_DIR / "cgm_baseline_test_mae.png"


MODEL_LABELS = {
    "persistence": "Persistence",
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

HORIZON_LABELS = {
    30: "30-minute",
    60: "60-minute",
}

MODEL_ORDER = [
    "Persistence",
    "Ridge",
    "Random Forest",
    "XGBoost",
]

HORIZON_ORDER = [
    "30-minute",
    "60-minute",
]


def validate_results(results: pd.DataFrame) -> None:
    required_columns = {
        "model",
        "horizon_minutes",
        "evaluation_split",
        "mae",
    }

    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            "Baseline results are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    results["horizon_minutes"] = pd.to_numeric(
        results["horizon_minutes"],
        errors="coerce",
    )

    results["mae"] = pd.to_numeric(
        results["mae"],
        errors="coerce",
    )

    if results["horizon_minutes"].isna().any():
        raise ValueError(
            "Baseline results contain invalid horizon values."
        )

    if results["mae"].isna().any():
        raise ValueError(
            "Baseline results contain invalid MAE values."
        )

    if (results["mae"] < 0).any():
        raise ValueError(
            "Baseline results contain negative MAE values."
        )


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RESULTS_PATH}. "
            "Run src/models/train_cgm_baselines.py first."
        )

    results = pd.read_csv(RESULTS_PATH)
    validate_results(results)

    test_results = results[
        results["evaluation_split"] == "test"
    ].copy()

    if test_results.empty:
        raise ValueError(
            "No held-out test rows were found in the baseline results."
        )

    test_results["model_label"] = test_results["model"].map(
        MODEL_LABELS
    )

    test_results["horizon_label"] = test_results[
        "horizon_minutes"
    ].map(HORIZON_LABELS)

    unmapped_models = test_results.loc[
        test_results["model_label"].isna(),
        "model",
    ].unique()

    if len(unmapped_models) > 0:
        raise ValueError(
            "Unrecognized baseline models: "
            f"{sorted(unmapped_models)}"
        )

    unmapped_horizons = test_results.loc[
        test_results["horizon_label"].isna(),
        "horizon_minutes",
    ].unique()

    if len(unmapped_horizons) > 0:
        raise ValueError(
            "Unrecognized forecast horizons: "
            f"{sorted(unmapped_horizons)}"
        )

    duplicate_rows = test_results.duplicated(
        subset=[
            "model_label",
            "horizon_label",
        ],
        keep=False,
    )

    if duplicate_rows.any():
        duplicates = test_results.loc[
            duplicate_rows,
            [
                "model",
                "horizon_minutes",
                "evaluation_split",
            ],
        ]

        raise ValueError(
            "Multiple test rows were found for the same model and "
            f"forecast horizon:\n{duplicates.to_string(index=False)}"
        )

    pivot = test_results.pivot(
        index="model_label",
        columns="horizon_label",
        values="mae",
    )

    missing_models = [
        model
        for model in MODEL_ORDER
        if model not in pivot.index
    ]

    if missing_models:
        raise ValueError(
            "Test results are missing baseline models: "
            f"{missing_models}"
        )

    missing_horizons = [
        horizon
        for horizon in HORIZON_ORDER
        if horizon not in pivot.columns
    ]

    if missing_horizons:
        raise ValueError(
            "Test results are missing forecast horizons: "
            f"{missing_horizons}"
        )

    pivot = pivot.loc[
        MODEL_ORDER,
        HORIZON_ORDER,
    ]

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6),
        width=0.78,
    )

    ax.set_title(
        "CGM-only baseline performance on the held-out test set"
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean absolute error (mg/dL)")
    ax.legend(
        title="Forecast horizon",
        frameon=False,
    )
    ax.grid(
        axis="y",
        alpha=0.3,
    )
    ax.set_axisbelow(True)

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.1f",
            padding=3,
            fontsize=9,
        )

    maximum_mae = float(
        pivot.to_numpy().max()
    )

    ax.set_ylim(
        0,
        maximum_mae * 1.18,
    )

    plt.xticks(
        rotation=0,
    )
    plt.tight_layout()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print(
        "Saved held-out baseline MAE figure to "
        f"{OUT_PATH}"
    )


if __name__ == "__main__":
    main()