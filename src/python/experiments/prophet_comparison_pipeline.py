from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc
import math
import os
import warnings

import numpy as np
import pandas as pd
import polars as pl
from prophet import Prophet
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 90
EXPECTED_SUBMISSION_ROWS = 123_467
NEGATIVE_TO_POSITIVE_RATIO = 19
DEFAULT_MAX_BALANCED_ROWS = 250_000

PROPHET_FEATURE_COLUMNS = [
    "prophet_orders_yhat",
    "prophet_orders_trend",
    "prophet_orders_weekly",
    "prophet_orders_yearly",
    "prophet_orders_holidays",
]

LEAKERS = [
    "first_action_ts",
    "last_action_ts",
    "n_order_shipped",
    "n_place_order_web",
    "n_place_order_phone",
    "n_place_downpayment",
    "n_account_downpaymentreceived",
    "n_account_downpaymentcleared",
    "n_customer_requested_catalog_(digital)",
]

XGB_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.08,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "booster": "gbtree",
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": 8,
    "random_state": RANDOM_STATE,
}


@dataclass(frozen=True)
class ComparisonPaths:
    project_root: Path
    source_submission_dir: Path
    output_dir: Path
    source_data_dir: Path
    output_data_dir: Path
    train_features: Path
    test_features: Path
    test_template: Path
    dt_clean: Path
    user_outcomes: Path
    daily_counts: Path
    source_prophet_forecast: Path
    output_prophet_forecast: Path
    prophet_summary: Path
    submission_without_prophet: Path
    submission_with_prophet: Path
    comparison_metrics: Path
    prediction_comparison: Path


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "src").exists() and (candidate / "README.md").exists():
            return candidate
    return Path(__file__).resolve().parents[3]


def parse_max_rows(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "full", "all"}:
        return None
    return int(normalized.replace(",", ""))


def get_paths(
    source_submission_dir: Path | None = None,
    output_dir: Path | None = None,
) -> ComparisonPaths:
    project_root = find_project_root()
    source_dir = source_submission_dir or project_root / "data" / "raw"
    out_dir = output_dir or project_root / "results" / "experiments" / "prophet-comparison"
    source_data_dir = project_root / "data" / "processed"
    output_data_dir = out_dir / "data"
    output_data_dir.mkdir(parents=True, exist_ok=True)

    return ComparisonPaths(
        project_root=project_root,
        source_submission_dir=source_dir,
        output_dir=out_dir,
        source_data_dir=source_data_dir,
        output_data_dir=output_data_dir,
        train_features=source_data_dir / "train_features_kcut_sample.parquet",
        test_features=source_data_dir / "test_features_open_journeys2.parquet",
        test_template=source_dir / "open_journeys2_flattened_all0.csv",
        dt_clean=source_data_dir / "dt_clean.parquet",
        user_outcomes=source_data_dir / "user_outcomes.parquet",
        daily_counts=source_data_dir / "order_shipped_ts.csv",
        source_prophet_forecast=source_data_dir / "prophet_daily_order_forecast.csv",
        output_prophet_forecast=output_data_dir / "prophet_daily_order_forecast.csv",
        prophet_summary=output_data_dir / "prophet_training_summary.csv",
        submission_without_prophet=out_dir / "submission_without_prophet.csv",
        submission_with_prophet=out_dir / "submission_with_prophet.csv",
        comparison_metrics=out_dir / "comparison_metrics.csv",
        prediction_comparison=out_dir / "prediction_comparison.csv",
    )


def require_inputs(paths: ComparisonPaths) -> None:
    required = [
        paths.train_features,
        paths.test_features,
        paths.test_template,
        paths.dt_clean,
        paths.user_outcomes,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required cached inputs:\n{missing_text}")


def normalize_day(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None).dt.floor("D")


def load_balanced_train(paths: ComparisonPaths, max_balanced_rows: int | None) -> pd.DataFrame:
    train_pl = pl.read_parquet(paths.train_features)
    failures = train_pl.filter(pl.col("final_outcome") == "failure")
    successes = train_pl.filter(pl.col("final_outcome") == "success")

    full_successes = min(failures.height // NEGATIVE_TO_POSITIVE_RATIO, successes.height)
    full_failures = failures.height
    full_rows = full_failures + full_successes

    if max_balanced_rows is None:
        n_failures = full_failures
        n_successes = full_successes
    else:
        target_rows = min(int(max_balanced_rows), full_rows)
        success_share = full_successes / full_rows
        n_successes = max(1, int(round(target_rows * success_share)))
        n_failures = target_rows - n_successes
        n_successes = min(n_successes, successes.height)
        n_failures = min(n_failures, failures.height)

    sampled = pl.concat(
        [
            failures.sample(n=n_failures, seed=RANDOM_STATE),
            successes.sample(n=n_successes, seed=RANDOM_STATE),
        ],
        how="vertical",
    ).sample(fraction=1.0, shuffle=True, seed=RANDOM_STATE)

    train = sampled.to_pandas()
    train["id"] = train["id"].astype(str)
    train["final_outcome"] = pd.Categorical(
        train["final_outcome"], categories=["success", "failure"]
    )
    print(
        "Balanced train rows:",
        len(train),
        train["final_outcome"].value_counts().to_dict(),
    )
    return train


def load_test(paths: ComparisonPaths) -> tuple[pd.DataFrame, pd.Series]:
    test = pl.read_parquet(paths.test_features).to_pandas()
    test["id"] = test["id"].astype(str)

    submission_ids = pd.read_csv(paths.test_template, usecols=["id"])["id"].astype(str)
    if len(submission_ids) != EXPECTED_SUBMISSION_ROWS:
        raise ValueError(
            f"{paths.test_template} has {len(submission_ids)} rows; "
            f"expected {EXPECTED_SUBMISSION_ROWS}."
        )

    test = pd.DataFrame({"id": submission_ids}).merge(test, on="id", how="left")
    if len(test) != EXPECTED_SUBMISSION_ROWS:
        raise ValueError(f"Test merge produced {len(test)} rows.")

    test["final_outcome"] = test["final_outcome"].fillna("incomplete")
    test["final_outcome"] = pd.Categorical(
        test["final_outcome"], categories=["success", "failure"]
    )
    return test, submission_ids


def get_success_journey_cutoff_days(paths: ComparisonPaths, default: int = 71) -> int:
    try:
        success_ids = (
            pl.scan_parquet(paths.user_outcomes)
            .filter(pl.col("final_outcome") == "success")
            .select("id")
        )
        p80 = (
            pl.scan_parquet(paths.dt_clean)
            .join(success_ids, on="id", how="inner")
            .group_by("id")
            .agg(
                pl.min("event_timestamp").alias("first_ts"),
                pl.max("event_timestamp").alias("last_ts"),
            )
            .with_columns(
                (pl.col("last_ts") - pl.col("first_ts"))
                .dt.total_days()
                .alias("journey_length_days")
            )
            .select(pl.col("journey_length_days").quantile(0.80).alias("p80_days"))
            .collect()
            .item()
        )
        if pd.notna(p80) and np.isfinite(p80):
            return int(round(float(p80)))
    except Exception as exc:
        print(f"Using default Prophet start cutoff ({default} days): {exc}")
    return default


def load_daily_counts(paths: ComparisonPaths) -> pd.DataFrame:
    if paths.daily_counts.exists():
        daily = pd.read_csv(paths.daily_counts)
        if {"date", "n_order_shipped"}.issubset(daily.columns):
            daily = daily.rename(columns={"date": "ds", "n_order_shipped": "y"})
        elif not {"ds", "y"}.issubset(daily.columns):
            raise ValueError(f"Could not find date/count columns in {paths.daily_counts}")
        daily = daily[["ds", "y"]]
        daily["ds"] = pd.to_datetime(daily["ds"])
        return daily.sort_values("ds").reset_index(drop=True)

    daily = (
        pl.scan_parquet(paths.dt_clean)
        .with_columns(pl.col("event_timestamp").dt.date().alias("ds"))
        .group_by("ds")
        .agg((pl.col("event_name") == "order_shipped").sum().alias("y"))
        .sort("ds")
        .collect()
        .to_pandas()
    )
    daily["ds"] = pd.to_datetime(daily["ds"])
    return daily


def select_prophet_columns(forecast: pd.DataFrame) -> pd.DataFrame:
    forecast = forecast.copy()
    forecast["ds"] = pd.to_datetime(forecast["ds"]).dt.tz_localize(None)
    for col in ["weekly", "yearly", "holidays"]:
        if col not in forecast.columns:
            forecast[col] = 0.0

    selected = forecast[["ds", "yhat", "trend", "weekly", "yearly", "holidays"]]
    return selected.rename(
        columns={
            "ds": "prophet_ds",
            "yhat": "prophet_orders_yhat",
            "trend": "prophet_orders_trend",
            "weekly": "prophet_orders_weekly",
            "yearly": "prophet_orders_yearly",
            "holidays": "prophet_orders_holidays",
        }
    )


def fit_or_load_prophet_forecast(
    paths: ComparisonPaths,
    frames: list[pd.DataFrame],
    rebuild_prophet: bool,
) -> pd.DataFrame:
    needed_days = pd.concat(
        [normalize_day(frame["last_action_ts"]) for frame in frames],
        ignore_index=True,
    ).dropna()
    forecast_end = max(pd.Timestamp("2024-12-01"), needed_days.max())

    for forecast_path in [paths.output_prophet_forecast, paths.source_prophet_forecast]:
        if not rebuild_prophet and forecast_path.exists():
            forecast = pd.read_csv(forecast_path)
            forecast["ds"] = pd.to_datetime(forecast["ds"])
            if forecast["ds"].max() >= forecast_end:
                print("Loaded Prophet forecast:", forecast_path)
                return select_prophet_columns(forecast)
            print("Existing Prophet forecast is too short; refitting.")

    daily = load_daily_counts(paths)
    cutoff_days = get_success_journey_cutoff_days(paths, default=71)
    start_cutoff = daily["ds"].min() + pd.Timedelta(days=cutoff_days)
    prophet_train = daily[daily["ds"] > start_cutoff].copy()
    prophet_train = prophet_train[prophet_train["ds"].dt.dayofweek < 5].copy()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    try:
        model.add_country_holidays(country_name="US")
    except Exception as exc:
        print("Skipping Prophet US holidays:", exc)

    model.fit(prophet_train)
    future = pd.DataFrame(
        {"ds": pd.date_range(prophet_train["ds"].min(), forecast_end, freq="D")}
    )
    forecast = model.predict(future)
    if "holidays" not in forecast.columns:
        forecast["holidays"] = 0.0

    forecast.to_csv(paths.output_prophet_forecast, index=False)
    pd.DataFrame(
        [
            {"key": "success_journey_length_p80_days", "value": cutoff_days},
            {"key": "prophet_start_cutoff", "value": start_cutoff.date().isoformat()},
            {"key": "daily_train_rows_weekdays_only", "value": len(prophet_train)},
            {"key": "forecast_end", "value": forecast_end.date().isoformat()},
        ]
    ).to_csv(paths.prophet_summary, index=False)
    print("Fit Prophet forecast:", paths.output_prophet_forecast)
    return select_prophet_columns(forecast)


def add_prophet_features(frame: pd.DataFrame, prophet_forecast: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["prophet_ds"] = normalize_day(out["last_action_ts"])
    out = out.merge(prophet_forecast, on="prophet_ds", how="left")
    medians = prophet_forecast[PROPHET_FEATURE_COLUMNS].median(numeric_only=True)
    out[PROPHET_FEATURE_COLUMNS] = (
        out[PROPHET_FEATURE_COLUMNS].fillna(medians).fillna(0)
    )
    return out


def make_xy(
    frame: pd.DataFrame,
    include_prophet: bool,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    y = (frame["final_outcome"] == "success").astype(int)
    groups = frame["id"].astype(str)

    prophet_drop_cols = set() if include_prophet else set(PROPHET_FEATURE_COLUMNS)
    drop_cols = {
        "final_outcome",
        "id",
        "snapshot_id",
        "prophet_ds",
        *LEAKERS,
        *prophet_drop_cols,
    }
    if feature_columns is None:
        feature_columns = [col for col in frame.columns if col not in drop_cols]

    x = frame.reindex(columns=feature_columns, fill_value=0).copy()
    for col in x.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        x[col] = x[col].view("int64") / 1e9
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0)
    return x, y, groups, feature_columns


def make_model() -> XGBClassifier:
    return XGBClassifier(**XGB_PARAMS)


def evaluate_predictions(y_true: pd.Series, prob: np.ndarray) -> dict[str, float]:
    return {
        "valid_brier": brier_score_loss(y_true, prob),
        "valid_prauc": average_precision_score(y_true, prob),
        "valid_prediction_mean": float(np.mean(prob)),
    }


def write_submission(
    submission_ids: pd.Series,
    prob: np.ndarray,
    path: Path,
) -> pd.DataFrame:
    submission = pd.DataFrame(
        {
            "id": submission_ids.astype(str),
            "order_shipped": np.clip(np.asarray(prob, dtype=float), 0, 1),
        }
    )
    if len(submission) != EXPECTED_SUBMISSION_ROWS:
        raise ValueError(
            f"Submission has {len(submission)} rows; expected {EXPECTED_SUBMISSION_ROWS}."
        )
    submission.to_csv(path, index=False)
    return submission


def run_variant(
    label: str,
    include_prophet: bool,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    submission_ids: pd.Series,
    split_indices: tuple[np.ndarray, np.ndarray],
    output_path: Path,
) -> tuple[dict[str, float | int | str | bool], pd.DataFrame]:
    x, y, _, feature_columns = make_xy(train_frame, include_prophet=include_prophet)
    x_test, _, _, _ = make_xy(
        test_frame,
        include_prophet=include_prophet,
        feature_columns=feature_columns,
    )
    train_idx, valid_idx = split_indices

    validation_model = make_model()
    validation_model.fit(x.iloc[train_idx], y.iloc[train_idx])
    valid_prob = validation_model.predict_proba(x.iloc[valid_idx])[:, 1]
    metrics = evaluate_predictions(y.iloc[valid_idx], valid_prob)

    final_model = make_model()
    final_model.fit(x, y)
    test_prob = final_model.predict_proba(x_test)[:, 1]
    submission = write_submission(submission_ids, test_prob, output_path)

    metrics.update(
        {
            "variant": label,
            "include_prophet": include_prophet,
            "n_features": len(feature_columns),
            "train_rows": len(train_frame),
            "submission_mean": float(submission["order_shipped"].mean()),
            "submission_std": float(submission["order_shipped"].std()),
            "submission_min": float(submission["order_shipped"].min()),
            "submission_p50": float(submission["order_shipped"].median()),
            "submission_max": float(submission["order_shipped"].max()),
            "output_path": str(output_path),
        }
    )
    return metrics, submission


def compare_submissions(
    without_prophet: pd.DataFrame,
    with_prophet: pd.DataFrame,
    paths: ComparisonPaths,
) -> pd.DataFrame:
    comparison = without_prophet.rename(
        columns={"order_shipped": "without_prophet"}
    ).merge(
        with_prophet.rename(columns={"order_shipped": "with_prophet"}),
        on="id",
        how="inner",
    )
    comparison["delta_with_minus_without"] = (
        comparison["with_prophet"] - comparison["without_prophet"]
    )
    comparison.to_csv(paths.prediction_comparison, index=False)
    return comparison


def run_comparison(
    max_balanced_rows: int | str | None = "env",
    rebuild_prophet: bool | None = None,
    source_submission_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if max_balanced_rows == "env":
        max_balanced_rows = parse_max_rows(
            os.getenv("MAX_BALANCED_ROWS", str(DEFAULT_MAX_BALANCED_ROWS))
        )
    else:
        max_balanced_rows = parse_max_rows(max_balanced_rows)

    if rebuild_prophet is None:
        rebuild_prophet = os.getenv("REBUILD_PROPHET", "0") == "1"

    if output_dir is None and os.getenv("OUTPUT_DIR"):
        output_dir = Path(os.environ["OUTPUT_DIR"])

    paths = get_paths(source_submission_dir=source_submission_dir, output_dir=output_dir)
    require_inputs(paths)

    print("Project root:", paths.project_root)
    print("Source submission dir:", paths.source_submission_dir)
    print("Output dir:", paths.output_dir)
    print("Max balanced rows:", max_balanced_rows)

    train = load_balanced_train(paths, max_balanced_rows=max_balanced_rows)
    test, submission_ids = load_test(paths)

    prophet_forecast = fit_or_load_prophet_forecast(
        paths,
        frames=[train, test],
        rebuild_prophet=bool(rebuild_prophet),
    )
    train = add_prophet_features(train, prophet_forecast)
    test = add_prophet_features(test, prophet_forecast)
    print("Prophet features:", PROPHET_FEATURE_COLUMNS)

    y = (train["final_outcome"] == "success").astype(int)
    groups = train["id"].astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    split_indices = next(splitter.split(np.zeros(len(train)), y, groups))

    metrics_without, submission_without = run_variant(
        label="without_prophet",
        include_prophet=False,
        train_frame=train,
        test_frame=test,
        submission_ids=submission_ids,
        split_indices=split_indices,
        output_path=paths.submission_without_prophet,
    )
    metrics_with, submission_with = run_variant(
        label="with_prophet",
        include_prophet=True,
        train_frame=train,
        test_frame=test,
        submission_ids=submission_ids,
        split_indices=split_indices,
        output_path=paths.submission_with_prophet,
    )

    metrics = pd.DataFrame([metrics_without, metrics_with])
    prediction_comparison = compare_submissions(submission_without, submission_with, paths)

    delta_summary = prediction_comparison["delta_with_minus_without"].describe().to_frame("delta")
    correlation = prediction_comparison[["without_prophet", "with_prophet"]].corr().iloc[0, 1]
    metrics["prediction_correlation"] = correlation
    metrics.to_csv(paths.comparison_metrics, index=False)

    print("Wrote:", paths.submission_without_prophet)
    print("Wrote:", paths.submission_with_prophet)
    print("Wrote:", paths.comparison_metrics)
    print("Wrote:", paths.prediction_comparison)

    del train, test
    gc.collect()

    return {
        "metrics": metrics,
        "prediction_comparison": prediction_comparison,
        "delta_summary": delta_summary,
        "output_paths": {
            "without_prophet": paths.submission_without_prophet,
            "with_prophet": paths.submission_with_prophet,
            "metrics": paths.comparison_metrics,
            "prediction_comparison": paths.prediction_comparison,
        },
    }


if __name__ == "__main__":
    results = run_comparison()
    print(results["metrics"].to_string(index=False))
    print(results["delta_summary"].to_string())
