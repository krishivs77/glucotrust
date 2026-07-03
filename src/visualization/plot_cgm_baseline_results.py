from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("reports/cgm_baseline_results.csv")
FIGURES_DIR = Path("reports/figures")
OUT_PATH = FIGURES_DIR / "cgm_baseline_mae.png"


MODEL_LABELS = {
    "persistence": "Persistence",
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

TARGET_LABELS = {
    "target_glucose_30min": "30-min forecast",
    "target_glucose_60min": "60-min forecast",
}


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RESULTS_PATH}. Run src/models/train_cgm_baselines.py first."
        )

    results = pd.read_csv(RESULTS_PATH)

    results["model_label"] = results["model"].map(MODEL_LABELS)
    results["target_label"] = results["target"].map(TARGET_LABELS)

    pivot = results.pivot(index="model_label", columns="target_label", values="mae")

    # Keep model order stable.
    model_order = ["Persistence", "Ridge", "Random Forest", "XGBoost"]
    pivot = pivot.loc[model_order]

    ax = pivot.plot(kind="bar", figsize=(9, 5))

    ax.set_title("CGM-only baseline forecasting error")
    ax.set_xlabel("Model")
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