from __future__ import annotations

from pathlib import Path
import gc
import json
import math
import random
import warnings

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
EXPECTED_SUBMISSION_ROWS = 123_467
MAX_LEN = 80
MAX_TRAIN_SNAPSHOTS = 120_000
BATCH_SIZE = 1024
EPOCHS = 4
LEARNING_RATE = 1e-3

STATIC_COLS = [
    "journey_length_s",
    "days_inactive",
    "total_actions",
    "mean_gap_sec",
    "median_gap_sec",
    "max_gap_sec",
]


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "src").exists() and (candidate / "README.md").exists():
            return candidate
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = find_project_root()
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "results" / "experiments" / "lstm"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

DT_CLEAN_PATH = DATA_DIR / "dt_clean.parquet"
TRAIN_FEATURES_PATH = DATA_DIR / "train_features_kcut_sample.parquet"
TEST_FEATURES_PATH = DATA_DIR / "test_features_open_journeys2.parquet"
TEST_EVENTS_PATH = RAW_DATA_DIR / "open_journeys2.csv"
TEST_TEMPLATE_PATH = RAW_DATA_DIR / "open_journeys2_flattened_all0.csv"

VOCAB_PATH = ARTIFACT_DIR / "rnn_event_vocab.json"
TRAIN_SEQUENCE_PATH = ARTIFACT_DIR / f"rnn_train_sequences_len{MAX_LEN}_n{MAX_TRAIN_SNAPSHOTS}.npz"
TEST_SEQUENCE_PATH = ARTIFACT_DIR / f"rnn_test_sequences_len{MAX_LEN}.npz"
MODEL_PATH = ARTIFACT_DIR / "lstm_model.pt"
SUBMISSION_PATH = ARTIFACT_DIR / "submission.csv"


def scan_events_csv(path: Path) -> pl.LazyFrame:
    lf = pl.scan_csv(path, infer_schema_length=20_000)
    schema = dict(lf.collect_schema())
    if schema.get("event_timestamp") == pl.String:
        lf = lf.with_columns(
            pl.col("event_timestamp")
            .str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ", time_zone="UTC", strict=False)
            .alias("event_timestamp")
        )
    return lf


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_event_vocab() -> dict[str, int]:
    if VOCAB_PATH.exists():
        return json.loads(VOCAB_PATH.read_text())

    train_events = (
        pl.scan_parquet(DT_CLEAN_PATH)
        .select("event_name")
        .unique()
        .collect()
        .get_column("event_name")
        .to_list()
    )
    test_events = (
        scan_events_csv(TEST_EVENTS_PATH)
        .select("event_name")
        .unique()
        .collect()
        .get_column("event_name")
        .to_list()
    )
    event_names = sorted(set(train_events) | set(test_events))
    vocab = {event_name: idx + 1 for idx, event_name in enumerate(event_names)}
    VOCAB_PATH.write_text(json.dumps(vocab, indent=2))
    return vocab


def sample_training_snapshots() -> pd.DataFrame:
    cols = ["id", "snapshot_id", "last_action_ts", "final_outcome", *STATIC_COLS]
    train_flat = pl.read_parquet(TRAIN_FEATURES_PATH, columns=cols)

    failures = train_flat.filter(pl.col("final_outcome") == "failure")
    successes = train_flat.filter(pl.col("final_outcome") == "success")
    target_successes = min(math.floor(failures.height / 19), successes.height)

    success_share = target_successes / (failures.height + target_successes)
    n_success = max(1, int(round(MAX_TRAIN_SNAPSHOTS * success_share)))
    n_failure = MAX_TRAIN_SNAPSHOTS - n_success

    sampled = pl.concat(
        [
            failures.sample(n=min(n_failure, failures.height), seed=RANDOM_STATE),
            successes.sample(n=min(n_success, successes.height), seed=RANDOM_STATE),
        ],
        how="vertical",
    ).sample(fraction=1.0, shuffle=True, seed=RANDOM_STATE)

    df = sampled.to_pandas()
    df["id"] = df["id"].astype(str)
    df["snapshot_id"] = df["snapshot_id"].astype(str)
    df["label"] = (df["final_outcome"] == "success").astype("float32")
    return df


def load_test_snapshots() -> pd.DataFrame:
    template = pd.read_csv(TEST_TEMPLATE_PATH, usecols=["id"])
    if len(template) != EXPECTED_SUBMISSION_ROWS:
        raise ValueError(
            f"{TEST_TEMPLATE_PATH} has {len(template)} rows; expected {EXPECTED_SUBMISSION_ROWS}."
        )

    test_flat = pl.read_parquet(TEST_FEATURES_PATH).to_pandas()
    test_flat["id"] = test_flat["id"].astype(str)
    test = template.astype({"id": str}).merge(test_flat, on="id", how="left")
    if len(test) != EXPECTED_SUBMISSION_ROWS:
        raise ValueError(f"Template merge produced {len(test)} rows.")

    test["snapshot_id"] = test["id"].astype(str)
    test["label"] = 0.0
    return test


def normalize_static(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    train_static = train_df[STATIC_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    test_static = test_df[STATIC_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)

    means = train_static.mean()
    stds = train_static.std().replace(0, 1).fillna(1)

    train_arr = ((train_static - means) / stds).astype("float32").to_numpy()
    test_arr = ((test_static - means) / stds).astype("float32").to_numpy()

    stats = {
        "means": means.to_dict(),
        "stds": stds.to_dict(),
        "static_cols": STATIC_COLS,
    }
    return train_arr, test_arr, stats


def collect_train_sequence_events(snapshots: pd.DataFrame) -> pd.DataFrame:
    snapshot_lf = pl.from_pandas(
        snapshots[["snapshot_id", "id", "last_action_ts"]].rename(columns={"last_action_ts": "cutoff_ts"})
    ).lazy()

    events = (
        pl.scan_parquet(DT_CLEAN_PATH)
        .select(["id", "event_timestamp", "event_name"])
        .join(snapshot_lf, on="id", how="inner")
        .filter(pl.col("event_timestamp") <= pl.col("cutoff_ts"))
        .sort(["snapshot_id", "event_timestamp"])
        .with_columns(
            pl.len().over("snapshot_id").alias("_n_events"),
            pl.col("event_timestamp").cum_count().over("snapshot_id").alias("_pos"),
        )
        .filter(pl.col("_pos") > (pl.col("_n_events") - MAX_LEN))
        .select(["snapshot_id", "event_timestamp", "event_name", "cutoff_ts"])
        .collect()
    )
    return events.to_pandas()


def collect_test_sequence_events(snapshots: pd.DataFrame) -> pd.DataFrame:
    snapshot_lf = pl.from_pandas(
        snapshots[["snapshot_id", "id", "last_action_ts"]].rename(columns={"last_action_ts": "cutoff_ts"})
    ).lazy()

    events_lf = (
        scan_events_csv(TEST_EVENTS_PATH)
        .select(["id", "event_timestamp", "event_name"])
        .unique(subset=["id", "event_timestamp", "event_name"], keep="first")
    )
    events = (
        events_lf
        .join(snapshot_lf, on="id", how="inner")
        .filter(pl.col("event_timestamp") <= pl.col("cutoff_ts"))
        .sort(["snapshot_id", "event_timestamp"])
        .with_columns(
            pl.len().over("snapshot_id").alias("_n_events"),
            pl.col("event_timestamp").cum_count().over("snapshot_id").alias("_pos"),
        )
        .filter(pl.col("_pos") > (pl.col("_n_events") - MAX_LEN))
        .select(["snapshot_id", "event_timestamp", "event_name", "cutoff_ts"])
        .collect()
    )
    return events.to_pandas()


def events_to_arrays(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    vocab: dict[str, int],
    static_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(snapshots)
    event_ids = np.zeros((n, MAX_LEN), dtype=np.int64)
    numeric = np.zeros((n, MAX_LEN, 2), dtype=np.float32)
    lengths = np.ones(n, dtype=np.int64)

    snapshot_to_idx = {sid: idx for idx, sid in enumerate(snapshots["snapshot_id"].astype(str))}
    labels = snapshots["label"].astype("float32").to_numpy()

    events = events.copy()
    events["snapshot_id"] = events["snapshot_id"].astype(str)
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"], utc=True)
    events["cutoff_ts"] = pd.to_datetime(events["cutoff_ts"], utc=True)

    for snapshot_id, group in events.groupby("snapshot_id", sort=False):
        idx = snapshot_to_idx.get(snapshot_id)
        if idx is None:
            continue
        group = group.sort_values("event_timestamp").tail(MAX_LEN)
        length = len(group)
        if length == 0:
            continue

        start = MAX_LEN - length
        ts = group["event_timestamp"].astype("int64").to_numpy() / 1e9
        cutoff = group["cutoff_ts"].iloc[-1].value / 1e9
        event_ids[idx, start:] = [vocab.get(name, 0) for name in group["event_name"]]

        delta_hours = np.diff(ts, prepend=ts[0]) / 3600.0
        hours_to_cutoff = np.maximum((cutoff - ts) / 3600.0, 0)
        numeric[idx, start:, 0] = np.log1p(np.maximum(delta_hours, 0))
        numeric[idx, start:, 1] = np.log1p(hours_to_cutoff)
        lengths[idx] = length

    return event_ids, numeric, static_arr.astype("float32"), lengths, labels


def build_or_load_sequences() -> tuple[dict, dict, dict[str, int]]:
    vocab = make_event_vocab()

    if TRAIN_SEQUENCE_PATH.exists() and TEST_SEQUENCE_PATH.exists():
        train_npz = dict(np.load(TRAIN_SEQUENCE_PATH, allow_pickle=True))
        test_npz = dict(np.load(TEST_SEQUENCE_PATH, allow_pickle=True))
        return train_npz, test_npz, vocab

    train_snapshots = sample_training_snapshots()
    test_snapshots = load_test_snapshots()
    train_static, test_static, static_stats = normalize_static(train_snapshots, test_snapshots)

    print("Training snapshots:", train_snapshots["final_outcome"].value_counts().to_dict())
    print("Collecting train sequence events...")
    train_events = collect_train_sequence_events(train_snapshots)
    print("Train sequence event rows:", len(train_events))

    print("Collecting test sequence events...")
    test_events = collect_test_sequence_events(test_snapshots)
    print("Test sequence event rows:", len(test_events))

    train_arrays = events_to_arrays(train_snapshots, train_events, vocab, train_static)
    test_arrays = events_to_arrays(test_snapshots, test_events, vocab, test_static)

    np.savez_compressed(
        TRAIN_SEQUENCE_PATH,
        event_ids=train_arrays[0],
        numeric=train_arrays[1],
        static=train_arrays[2],
        lengths=train_arrays[3],
        y=train_arrays[4],
        groups=train_snapshots["id"].astype(str).to_numpy(),
        static_stats=np.array(json.dumps(static_stats)),
    )
    np.savez_compressed(
        TEST_SEQUENCE_PATH,
        event_ids=test_arrays[0],
        numeric=test_arrays[1],
        static=test_arrays[2],
        lengths=test_arrays[3],
        y=test_arrays[4],
        ids=test_snapshots["id"].astype(str).to_numpy(),
    )

    del train_events, test_events
    gc.collect()

    return dict(np.load(TRAIN_SEQUENCE_PATH, allow_pickle=True)), dict(np.load(TEST_SEQUENCE_PATH, allow_pickle=True)), vocab


class JourneySequenceDataset(Dataset):
    def __init__(self, arrays: dict, indices: np.ndarray | None = None):
        self.event_ids = arrays["event_ids"]
        self.numeric = arrays["numeric"]
        self.static = arrays["static"]
        self.lengths = arrays["lengths"]
        self.y = arrays["y"]
        self.indices = np.arange(len(self.y)) if indices is None else indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = self.indices[item]
        return (
            torch.as_tensor(self.event_ids[idx], dtype=torch.long),
            torch.as_tensor(self.numeric[idx], dtype=torch.float32),
            torch.as_tensor(self.static[idx], dtype=torch.float32),
            torch.as_tensor(self.lengths[idx], dtype=torch.long),
            torch.as_tensor(self.y[idx], dtype=torch.float32),
        )


class JourneyLSTM(nn.Module):
    def __init__(self, n_events: int, n_static: int):
        super().__init__()
        self.embedding = nn.Embedding(n_events + 1, 24, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=26,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.static_net = nn.Sequential(
            nn.Linear(n_static, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 2 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, event_ids, numeric, static, lengths):
        emb = self.embedding(event_ids)
        x = torch.cat([emb, numeric], dim=-1)
        packed = pack_padded_sequence(
            x,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        seq_repr = torch.cat([hidden[-2], hidden[-1]], dim=1)
        static_repr = self.static_net(static)
        return self.head(torch.cat([seq_repr, static_repr], dim=1)).squeeze(1)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for event_ids, numeric, static, lengths, _ in loader:
            logits = model(
                event_ids.to(device),
                numeric.to(device),
                static.to(device),
                lengths.to(device),
            )
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(preds)


def train_lstm(train_arrays: dict, vocab: dict[str, int]) -> JourneyLSTM:
    y = train_arrays["y"].astype(int)
    groups = train_arrays["groups"].astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, valid_idx = next(splitter.split(np.zeros(len(y)), y, groups))

    train_loader = DataLoader(
        JourneySequenceDataset(train_arrays, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        JourneySequenceDataset(train_arrays, valid_idx),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
    )

    device = get_device()
    model = JourneyLSTM(n_events=len(vocab), n_static=train_arrays["static"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_prauc = -np.inf
    best_state = None

    print("Device:", device)
    print("Train/valid rows:", len(train_idx), len(valid_idx))
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for event_ids, numeric, static, lengths, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                event_ids.to(device),
                numeric.to(device),
                static.to(device),
                lengths.to(device),
            )
            loss = criterion(logits, labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        valid_prob = predict(model, valid_loader, device)
        valid_y = y[valid_idx]
        prauc = average_precision_score(valid_y, valid_prob)
        brier = brier_score_loss(valid_y, valid_prob)
        print(
            f"epoch={epoch} loss={np.mean(losses):.5f} "
            f"valid_prauc={prauc:.5f} valid_brier={brier:.5f}"
        )
        if prauc > best_prauc:
            best_prauc = prauc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "max_len": MAX_LEN,
            "static_cols": STATIC_COLS,
            "best_valid_prauc": best_prauc,
        },
        MODEL_PATH,
    )
    return model


def write_submission(model: JourneyLSTM, test_arrays: dict) -> pd.DataFrame:
    device = get_device()
    model = model.to(device)
    test_loader = DataLoader(
        JourneySequenceDataset(test_arrays),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
    )
    prob = predict(model, test_loader, device)

    submission = pd.DataFrame(
        {
            "id": test_arrays["ids"].astype(str),
            "order_shipped": np.clip(prob, 0, 1),
        }
    )
    template = pd.read_csv(TEST_TEMPLATE_PATH, usecols=["id"]).astype({"id": str})
    submission = template.merge(submission, on="id", how="left")
    submission["order_shipped"] = submission["order_shipped"].fillna(submission["order_shipped"].median())

    if len(submission) != EXPECTED_SUBMISSION_ROWS:
        raise ValueError(f"Submission has {len(submission)} rows; expected {EXPECTED_SUBMISSION_ROWS}.")
    if not submission["id"].astype(str).equals(template["id"].astype(str)):
        raise ValueError("Submission id order does not match template.")

    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Wrote:", SUBMISSION_PATH)
    print("Rows:", len(submission))
    print("Probability range:", submission["order_shipped"].min(), submission["order_shipped"].max())
    return submission


def main() -> pd.DataFrame:
    seed_everything()
    print("Submission dir:", SUBMISSION_DIR)
    print("Max train snapshots:", MAX_TRAIN_SNAPSHOTS)
    print("Max sequence length:", MAX_LEN)

    train_arrays, test_arrays, vocab = build_or_load_sequences()
    model = train_lstm(train_arrays, vocab)
    return write_submission(model, test_arrays)


if __name__ == "__main__":
    main()
