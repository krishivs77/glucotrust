from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CGM_ONLY_PATH = Path("reports/selective_prediction_results.csv")
CONTEXT_PATH = Path("reports/context_selective_prediction_results.csv")
FIGURES_DIR = Path("reports/figures")

TARGET_LABELS = {
    "target_glucose_30min": "30-min forecast",
    "target_glucose_60min": "60-min forecast",
}

FEATURE_SET_LABELS = {
    "cgm_only": "CGM only",
    "cgm_plus_context": "CGM + meal/insulin",
}


def load_results() -> pd.DataFrame:
    if not CGM_ONLY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CGM_ONLY_PATH}. Run src/evaluation/selective_prediction.py first."
        )

    if not CONTEXT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CONTEXT_PATH}. Run src/evaluation/context_selective_prediction.py first."
        )

    cgm_only = pd.read_csv(CGM_ONLY_PATH)
    context = pd.read_csv(CONTEXT_PATH)

    if "feature_set" not in cgm_only.columns:
        cgm_only["feature_set"] = "cgm_only"

    if "feature_set" not in context.columns:
        context["feature_set"] = "cgm_plus_context"

    return pd.concat([cgm_only, context], ignore_index=True)


def plot_target(results: pd.DataFrame, target: str, out_path: Path) -> None:
    target_df = results[results["target"] == target].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    for feature_set, group in target_df.groupby("feature_set"):
        group = group.sort_values("coverage")
        label = FEATURE_SET_LABELS.get(feature_set, feature_set)

        ax.plot(
            group["coverage"] * 100,
            group["rmse"],
            marker="o",
            label=label,
        )

    ax.set_title(f"Selective prediction reliability: {TARGET_LABELS[target]}")
    ax.set_xlabel("Coverage kept (%)")
    ax.set_ylabel("RMSE on kept predictions (mg/dL)")
    ax.grid(alpha=0.3)
    ax.legend(title="Feature set")
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved figure to {out_path}")


def main() -> None:
    results = load_results()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_target(
        results=results,
        target="target_glucose_30min",
        out_path=FIGURES_DIR / "selective_prediction_context_comparison_30min.png",
    )

    plot_target(
        results=results,
        target="target_glucose_60min",
        out_path=FIGURES_DIR / "selective_prediction_context_comparison_60min.png",
    )


if __name__ == "__main__":
    main()