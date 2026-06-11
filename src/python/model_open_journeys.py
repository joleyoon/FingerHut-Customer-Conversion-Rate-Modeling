from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "dat_train1.csv"
OPEN_PATH = ROOT / "data" / "raw" / "open_journeys1_flattened_all0.csv"
PREDICTION_PATH = ROOT / "results" / "predictions" / "open_journey_predictions.csv"
IMPORTANCE_PATH = ROOT / "reports" / "figures" / "model_variable_importance.png"

SELECTED_EVENTS = [
    "browse_products",
    "view_cart",
    "add_to_cart",
    "begin_checkout",
    "place_order_web",
    "place_order_phone",
    "place_downpayment",
    "application_web_view",
    "application_web_submit",
    "application_web_approved",
    "promotion_created",
    "campaignemail_clicked",
    "campaign_click",
]


def build_features(events: pd.DataFrame) -> pd.DataFrame:
    """Create journey-level features from events visible at snapshot time."""
    events = events.sort_values(["id", "event_timestamp"]).copy()

    grouped = events.groupby("id")
    features = grouped.agg(
        n_events=("event_name", "size"),
        n_unique_events=("event_name", "nunique"),
        first_event_time=("event_timestamp", "min"),
        last_event_time=("event_timestamp", "max"),
    )
    features["journey_duration_days"] = (
        features["last_event_time"] - features["first_event_time"]
    ).dt.total_seconds() / (60 * 60 * 24)

    event_counts = (
        events[events["event_name"].isin(SELECTED_EVENTS)]
        .groupby(["id", "event_name"])
        .size()
        .unstack(fill_value=0)
    )
    event_counts = event_counts.reindex(columns=SELECTED_EVENTS, fill_value=0)
    event_counts = event_counts.add_prefix("n_")

    last_events = grouped.tail(1).set_index("id")["event_name"]
    for event_name in SELECTED_EVENTS:
        features[f"last_is_{event_name}"] = (last_events == event_name).astype(int)

    features = features.join(event_counts, how="left").fillna(0)
    features = features.drop(columns=["first_event_time", "last_event_time"])
    return features


def make_training_snapshots(df: pd.DataFrame, max_training_rows: int = 200_000):
    """Return point-in-time training features and labels.

    Positive examples use events before the first order_shipped event.
    Negative examples use old, non-shipped journeys that later became inactive.
    The final 60 days are held out as the current-open window.
    """
    cutoff_date = df["event_timestamp"].max() - pd.Timedelta(days=60)

    first_ship = (
        df.loc[df["event_name"] == "order_shipped"]
        .groupby("id")["event_timestamp"]
        .min()
        .rename("first_ship_time")
    )
    last_event_time = df.groupby("id")["event_timestamp"].max().rename("last_event_time")
    journey_status = pd.concat([first_ship, last_event_time], axis=1)

    positive_ids = journey_status.index[journey_status["first_ship_time"].notna()]
    negative_ids = journey_status.index[
        journey_status["first_ship_time"].isna()
        & (journey_status["last_event_time"] < cutoff_date)
    ]

    labels = pd.concat(
        [
            pd.Series(1, index=positive_ids, name="will_ship"),
            pd.Series(0, index=negative_ids, name="will_ship"),
        ]
    )
    if len(labels) > max_training_rows:
        labels = (
            labels.groupby(labels)
            .sample(n=min(max_training_rows // 2, labels.value_counts().min()), random_state=123)
            .sort_index()
        )

    selected_ids = labels.index
    selected_events = df[df["id"].isin(selected_ids)].merge(
        first_ship, left_on="id", right_index=True, how="left"
    )
    snapshot_events = selected_events[
        selected_events["first_ship_time"].isna()
        | (selected_events["event_timestamp"] < selected_events["first_ship_time"])
    ].drop(columns=["first_ship_time"])

    features = build_features(snapshot_events)
    labels = labels.reindex(features.index)

    return features, labels, cutoff_date


def main():
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        TRAIN_PATH,
        usecols=["id", "event_name", "event_timestamp"],
        parse_dates=["event_timestamp"],
    )
    df = df.drop_duplicates(subset=["id", "event_name", "event_timestamp"]).copy()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)

    X, y, cutoff_date = make_training_snapshots(df)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=10,
        max_features="sqrt",
        n_jobs=-1,
        oob_score=True,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    print(f"Open-window cutoff date: {cutoff_date}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_valid):,}")
    print(f"OOB accuracy: {rf.oob_score_:.4f}")
    print(f"Validation accuracy: {rf.score(X_valid, y_valid):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_valid, rf.predict(X_valid)))
    print("Classification report:")
    print(classification_report(y_valid, rf.predict(X_valid)))

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(20)
    ax = importances.plot(kind="barh", figsize=(9, 7), title="Random Forest Variable Importance")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PATH, dpi=150)
    plt.close()

    open_journeys = pd.read_csv(OPEN_PATH)
    open_events = df[df["id"].isin(open_journeys["id"])]
    X_open = build_features(open_events).reindex(open_journeys["id"]).fillna(0)
    X_open = X_open.reindex(columns=X.columns, fill_value=0)

    open_journeys["ship_probability"] = rf.predict_proba(X_open)[:, 1]
    open_journeys["predicted_order_shipped"] = (open_journeys["ship_probability"] >= 0.5).astype(int)
    open_journeys.to_csv(PREDICTION_PATH, index=False)
    print(f"Saved predictions to {PREDICTION_PATH}")
    print(f"Saved variable importance plot to {IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
