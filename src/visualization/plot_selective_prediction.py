from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("reports/selective_prediction_results.csv")
FIGURES_DIR = Path("reports/figures")
OUT_PATH = FIGURES_DIR / "selective_prediction_rmse.png"


TARGET_LABELS = {
    "target_glucose_30min": "30-min forecast",
    "target_glucose_60min": "60-min forecast",
}


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RESULTS_PATH}. Run src/evaluation/selective_prediction.py first."
        )

    results = pd.read_csv(RESULTS_PATH)
    results["target_label"] = results["target"].map(TARGET_LABELS)

    fig, ax = plt.subplots(figsize=(8, 5))

    for target_label, group in results.groupby("target_label"):
        group = group.sort_values("coverage")
        ax.plot(
            group["coverage"] * 100,
            group["rmse"],
            marker="o",
            label=target_label,
        )

    ax.set_title("Selective prediction improves reliability")
    ax.set_xlabel("Coverage kept (%)")
    ax.set_ylabel("RMSE on kept predictions (mg/dL)")
    ax.grid(alpha=0.3)
    ax.legend(title="Target")

    # Show 100% on the left or right? Here, decreasing coverage left-to-right is intuitive:
    # lower coverage = stricter confidence threshold.
    ax.invert_xaxis()

    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=200)
    plt.close()

    print(f"Saved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()