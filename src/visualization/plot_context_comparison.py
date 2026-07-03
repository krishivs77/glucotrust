from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("reports/context_baseline_results.csv")
FIGURES_DIR = Path("reports/figures")
OUT_PATH = FIGURES_DIR / "context_vs_cgm_mae.png"


MODEL_LABELS = {
    "persistence": "Persistence",
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

FEATURE_SET_LABELS = {
    "cgm_only": "CGM only",
    "cgm_plus_context": "CGM + meal/insulin",
}

TARGET_LABELS = {
    "target_glucose_30min": "30-min forecast",
    "target_glucose_60min": "60-min forecast",
}


def make_target_plot(results: pd.DataFrame, target: str, out_path: Path) -> None:
    target_df = results[results["target"] == target].copy()

    target_df["model_label"] = target_df["model"].map(MODEL_LABELS)
    target_df["feature_set_label"] = target_df["feature_set"].map(FEATURE_SET_LABELS)

    model_order = ["Persistence", "Ridge", "Random Forest", "XGBoost"]

    pivot = target_df.pivot(
        index="model_label",
        columns="feature_set_label",
        values="mae",
    ).loc[model_order]

    ax = pivot.plot(kind="bar", figsize=(9, 5))

    ax.set_title(f"CGM-only vs context features: {TARGET_LABELS[target]}")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE (mg/dL)")
    ax.legend(title="Feature set")
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved figure to {out_path}")


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RESULTS_PATH}. Run src/models/train_context_baselines.py first."
        )

    results = pd.read_csv(RESULTS_PATH)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    make_target_plot(
        results,
        target="target_glucose_30min",
        out_path=FIGURES_DIR / "context_vs_cgm_mae_30min.png",
    )

    make_target_plot(
        results,
        target="target_glucose_60min",
        out_path=FIGURES_DIR / "context_vs_cgm_mae_60min.png",
    )


if __name__ == "__main__":
    main()