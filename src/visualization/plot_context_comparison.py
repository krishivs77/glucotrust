from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("reports/context_baseline_results.csv")
FIGURES_DIR = Path("reports/figures")

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

HORIZON_LABELS = {
    30: "30-minute forecast",
    60: "60-minute forecast",
}


def make_horizon_plot(
    results: pd.DataFrame,
    horizon: int,
    out_path: Path,
) -> None:
    horizon_df = results[
        (results["evaluation_split"] == "test")
        & (results["horizon_minutes"] == horizon)
    ].copy()

    required = {
        "model",
        "feature_set",
        "mae",
    }

    missing = required - set(horizon_df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    horizon_df["model_label"] = (
        horizon_df["model"]
        .map(MODEL_LABELS)
    )

    horizon_df["feature_set_label"] = (
        horizon_df["feature_set"]
        .map(FEATURE_SET_LABELS)
    )

    model_order = [
        "Persistence",
        "Ridge",
        "Random Forest",
        "XGBoost",
    ]

    pivot = (
        horizon_df.pivot(
            index="model_label",
            columns="feature_set_label",
            values="mae",
        )
        .loc[model_order]
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(9, 5),
    )

    ax.set_title(
        f"CGM-only vs context features ({HORIZON_LABELS[horizon]})"
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Test MAE (mg/dL)")
    ax.legend(title="Feature set")
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

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved {out_path}")


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RESULTS_PATH}"
        )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.read_csv(RESULTS_PATH)

    make_horizon_plot(
        results,
        horizon=30,
        out_path=FIGURES_DIR / "context_vs_cgm_mae_30min.png",
    )

    make_horizon_plot(
        results,
        horizon=60,
        out_path=FIGURES_DIR / "context_vs_cgm_mae_60min.png",
    )


if __name__ == "__main__":
    main()