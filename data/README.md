# Data

The original event data is not committed because it is large and was provided
through the course competition. The only tracked data file is
`event_definitions.csv`, which maps event names to journey stages.

## Expected Layout

```text
data/
├── event_definitions.csv
├── raw/
│   ├── dat_train1.csv
│   ├── dat_train2.csv
│   ├── open_journeys1_flattened_all0.csv
│   ├── open_journeys2.csv
│   └── open_journeys2_flattened_all0.csv
└── processed/
    ├── dt_clean.parquet
    ├── dt_clean2.parquet
    ├── user_outcomes.parquet
    ├── user_outcomes2.parquet
    ├── train_features_kcut_sample.parquet
    ├── test_features.parquet
    └── test_features_open_journeys2.parquet
```

The R workflows in `src/r/data-preparation` create most processed files. File
names differ slightly between iterations because the course released data in
multiple rounds.

## Journey Labels

- **Successful:** the journey eventually contains `order_shipped`.
- **Unsuccessful:** no shipment occurs and the journey is inactive for at least
  60 days before the cutoff.
- **Ongoing:** no shipment occurs and the latest activity is within the final
  60-day window.
