from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_PATH = Path("reports/xgb_ensemble_uncertainty_bin_summary.csv")
FIGURES_DIR = Path("reports/figures")
OUT_PATH = FIGURES_DIR / "xgb_uncertainty_bins_mae.png"


TARGET_LABELS = {
    "target_glucose_30min": "30-min forecast",
    "target_glucose_60min": "60-min forecast",
}

UNCERTAINTY_ORDER = ["very_low", "low", "medium", "high", "very_high"]


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SUMMARY_PATH}. Run src/models/train_xgb_ensemble_uncertainty.py first."
        )

    summary = pd.read_csv(SUMMARY_PATH)

    summary["target_label"] = summary["target"].map(TARGET_LABELS)
    summary["uncertainty_bin"] = pd.Categorical(
        summary["uncertainty_bin"],
        categories=UNCERTAINTY_ORDER,
        ordered=True,
    )

    pivot = summary.pivot(
        index="uncertainty_bin",
        columns="target_label",
        values="mae",
    ).loc[UNCERTAINTY_ORDER]

    ax = pivot.plot(kind="bar", figsize=(9, 5))

    ax.set_title("Forecast error increases with ensemble uncertainty")
    ax.set_xlabel("Ensemble uncertainty bin")
    ax.set_ylabel("MAE (mg/dL)")
    ax.legend(title="Target")
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=0)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=200)
    plt.close()

    print(f"Saved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()