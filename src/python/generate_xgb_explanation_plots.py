"""Generate report-ready XGBoost importance, ICE, and CP plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
FIGURE_DIR = ROOT / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

from model_open_journeys import make_training_snapshots


LEAKY_FEATURES = {
    "n_order_shipped",
    "n_place_order_web",
    "n_place_order_phone",
    "n_place_downpayment",
    "n_account_downpaymentreceived",
    "n_account_downpaymentcleared",
    "n_customer_requested_catalog_(digital)",
    "last_is_place_order_web",
    "last_is_place_order_phone",
    "last_is_place_downpayment",
}


def display_name(feature: str) -> str:
    return feature.replace("_", " ").replace("n unique events", "number of unique events")


def main() -> None:
    events = pd.read_csv(
        DATA_DIR / "dat_train1.csv",
        usecols=["id", "event_name", "event_timestamp"],
        parse_dates=["event_timestamp"],
        nrows=300_000,
    )
    events = events.drop_duplicates(["id", "event_name", "event_timestamp"]).copy()
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"], utc=True)

    features, outcome, _ = make_training_snapshots(events, max_training_rows=20_000)
    features = features.drop(
        columns=[column for column in LEAKY_FEATURES if column in features],
        errors="ignore",
    )

    # Match the journey-modeling workflow's roughly 5% success prevalence.
    failures = outcome.index[outcome == 0]
    successes = outcome.index[outcome == 1]
    rng = np.random.default_rng(42)
    kept_successes = rng.choice(successes, size=min(len(successes), len(failures) // 19), replace=False)
    kept_rows = failures.union(pd.Index(kept_successes))
    features = features.loc[kept_rows]
    outcome = outcome.loc[kept_rows]

    x_train, x_valid, y_train, _ = train_test_split(
        features,
        outcome,
        test_size=0.25,
        stratify=outcome,
        random_state=42,
    )

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        booster="gbtree",
        tree_method="hist",
        n_jobs=8,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(x_train, y_train)

    importances = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)
    plotted_feature = next(feature for feature in importances.index if features[feature].nunique() > 5)
    plotted_label = display_name(plotted_feature)

    plt.style.use("seaborn-v0_8-whitegrid")
    top_importances = importances.head(12).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    top_importances.plot.barh(ax=ax, color="#277da1")
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Gain-based importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "xgb_feature_importance.png", dpi=180)
    plt.close(fig)

    values = np.unique(
        np.quantile(features[plotted_feature], np.linspace(0, 1, 12)).round(6)
    )
    ice_sample = x_valid.sample(n=min(100, len(x_valid)), random_state=42)
    ice_predictions = []
    for value in values:
        modified = ice_sample.copy()
        modified[plotted_feature] = value
        ice_predictions.append(model.predict_proba(modified)[:, 1])
    ice_predictions = np.asarray(ice_predictions).T

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for curve in ice_predictions:
        ax.plot(values, curve, color="#90caf9", alpha=0.22, linewidth=0.8)
    ax.plot(
        values,
        ice_predictions.mean(axis=0),
        color="#d1495b",
        linewidth=2.5,
        label="Average effect",
    )
    ax.set(
        title=f"ICE Profiles for {plotted_label}",
        xlabel=plotted_label.capitalize(),
        ylabel="Predicted probability of shipment",
        ylim=(-0.01, 1.01),
    )
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "xgb_ice_plot.png", dpi=180)
    plt.close(fig)

    valid_predictions = pd.Series(model.predict_proba(x_valid)[:, 1], index=x_valid.index)
    profile_targets = {"Higher-probability journey": 0.75, "Lower-probability journey": 0.10}
    selected_profiles = []
    for label, target in profile_targets.items():
        row_id = (valid_predictions - target).abs().idxmin()
        row = x_valid.loc[[row_id]]
        selected_profiles.append(
            {
                "label": label,
                "row": row,
                "baseline_probability": valid_predictions.loc[row_id],
                "baseline_value": row.iloc[0][plotted_feature],
            }
        )
    cp_values = np.unique(
        np.concatenate([values, [profile["baseline_value"] for profile in selected_profiles]])
    )

    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = ["#277da1", "#f8961e"]
    profile_summary = []
    for profile, color in zip(selected_profiles, colors):
        label = profile["label"]
        row = profile["row"]
        baseline_probability = profile["baseline_probability"]
        baseline_value = profile["baseline_value"]

        repeated = pd.concat([row] * len(cp_values), ignore_index=True)
        repeated[plotted_feature] = cp_values
        probabilities = model.predict_proba(repeated)[:, 1]

        ax.plot(
            cp_values,
            probabilities,
            marker="o",
            markersize=3.5,
            color=color,
            linewidth=2,
            label=f"{label} (baseline {baseline_probability:.2f})",
        )
        ax.scatter(
            baseline_value,
            baseline_probability,
            color=color,
            edgecolor="black",
            s=65,
            zorder=3,
        )
        profile_summary.append(
            {
                "profile": label,
                "baseline_probability": baseline_probability,
                "baseline_feature_value": baseline_value,
            }
        )

    ax.set(
        title=f"Ceteris Paribus Profiles for {plotted_label}",
        xlabel=plotted_label.capitalize(),
        ylabel="Predicted probability of shipment",
        ylim=(-0.01, 1.01),
    )
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "xgb_cp_profiles.png", dpi=180)
    plt.close(fig)

    pd.DataFrame(profile_summary).to_csv(FIGURE_DIR / "xgb_cp_profile_summary.csv", index=False)
    print(f"Plotted feature: {plotted_feature}")
    print(importances.head(10).to_string())
    print(pd.DataFrame(profile_summary).to_string(index=False))


if __name__ == "__main__":
    main()
