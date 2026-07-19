from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CGM_ONLY_PATH = Path("reports/selective_prediction_results.csv")
CONTEXT_PATH = Path("reports/context_selective_prediction_results.csv")
FIGURES_DIR = Path("reports/figures")

FEATURE_SET_LABELS = {
    "cgm_only": "CGM only",
    "cgm_plus_context": "CGM + meal/insulin",
}

HORIZON_LABELS = {
    30: "30-minute forecast",
    60: "60-minute forecast",
}


def load_results() -> pd.DataFrame:
    if not CGM_ONLY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CGM_ONLY_PATH}. "
            "Run src/evaluation/selective_prediction.py first."
        )

    if not CONTEXT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CONTEXT_PATH}. "
            "Run src/evaluation/context_selective_prediction.py first."
        )

    cgm_only = pd.read_csv(CGM_ONLY_PATH)
    context = pd.read_csv(CONTEXT_PATH)

    required_columns = {
        "horizon_minutes",
        "evaluation_split",
        "requested_coverage",
        "achieved_coverage",
        "rmse",
    }

    for path, frame in (
        (CGM_ONLY_PATH, cgm_only),
        (CONTEXT_PATH, context),
    ):
        missing_columns = required_columns - set(frame.columns)

        if missing_columns:
            raise ValueError(
                f"{path} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

    cgm_only = cgm_only.copy()
    context = context.copy()

    cgm_only["feature_set"] = "cgm_only"

    if "feature_set" not in context.columns:
        context["feature_set"] = "cgm_plus_context"

    test_results = pd.concat(
        [
            cgm_only,
            context,
        ],
        ignore_index=True,
    )

    test_results = test_results[
        test_results["evaluation_split"] == "test"
    ].copy()

    if test_results.empty:
        raise ValueError(
            "No held-out test rows were found."
        )

    unknown_feature_sets = (
        set(test_results["feature_set"].unique())
        - set(FEATURE_SET_LABELS)
    )

    if unknown_feature_sets:
        raise ValueError(
            "Unexpected feature sets found: "
            f"{sorted(unknown_feature_sets)}"
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
            "feature_set",
            "horizon_minutes",
            "requested_coverage",
        ],
        keep=False,
    )

    if duplicate_rows.any():
        raise ValueError(
            "Multiple test rows were found for the same "
            "feature set, horizon, and requested coverage."
        )

    return test_results


def plot_horizon(
    results: pd.DataFrame,
    horizon: int,
    out_path: Path,
) -> None:
    horizon_results = results[
        results["horizon_minutes"] == horizon
    ].copy()

    if horizon_results.empty:
        raise ValueError(
            f"No test results found for the "
            f"{horizon}-minute forecast."
        )

    fig, ax = plt.subplots(figsize=(8, 5))

    feature_set_order = [
        "cgm_only",
        "cgm_plus_context",
    ]

    for feature_set in feature_set_order:
        feature_results = horizon_results[
            horizon_results["feature_set"] == feature_set
        ].sort_values(
            "achieved_coverage",
            ascending=False,
        )

        if feature_results.empty:
            raise ValueError(
                f"No {feature_set} test rows found for the "
                f"{horizon}-minute forecast."
            )

        ax.plot(
            feature_results["achieved_coverage"] * 100,
            feature_results["rmse"],
            marker="o",
            linewidth=2,
            label=FEATURE_SET_LABELS[feature_set],
        )

    ax.set_title(
        "Selective prediction with and without context "
        f"({HORIZON_LABELS[horizon]})"
    )
    ax.set_xlabel("Predictions retained (%)")
    ax.set_ylabel("RMSE on retained predictions (mg/dL)")
    ax.grid(alpha=0.3)
    ax.legend(title="Feature set")
    ax.invert_xaxis()

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved figure to {out_path}")


def main() -> None:
    results = load_results()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_horizon(
        results=results,
        horizon=30,
        out_path=(
            FIGURES_DIR
            / "selective_prediction_context_comparison_30min.png"
        ),
    )

    plot_horizon(
        results=results,
        horizon=60,
        out_path=(
            FIGURES_DIR
            / "selective_prediction_context_comparison_60min.png"
        ),
    )


if __name__ == "__main__":
    main()