# Source Workflows

The curated source is organized by workflow rather than contributor.

## R Workflow

Run the Quarto files from the repository root so their `data/`, `reports/`, and
`results/` paths resolve correctly.

1. `r/data-preparation/01_data_cleaning.qmd`: remove duplicates, label journey
   outcomes, and write cleaned event data.
2. `r/data-preparation/02_sampling_flattening.qmd`: sample point-in-time journey
   snapshots and create model features.
3. `r/modeling/03_modelling.qmd`: compare random forest and XGBoost models.
4. `r/modeling/04_nn_prep.qmd` and `05_nn_fit.qmd`: prepare and fit an LSTM
   sequence model.
5. `r/forecasting/prophet.qmd`: forecast aggregate shipped-order volume.

Required R packages include `arrow`, `collapse`, `data.table`, `future`,
`ggplot2`, `mlr3`, `mlr3learners`, `mlr3tuning`, `patchwork`, `prophet`, and
`torch`.

## Python Workflow

- `python/model_open_journeys.py`: random forest baseline and open-journey
  predictions.
- `python/generate_xgb_explanation_plots.py`: report-ready XGBoost importance,
  ICE, and local-profile plots.
- `python/experiments/`: Prophet-feature comparison and sequence-model
  experiments.

Install dependencies with `pip install -r requirements.txt`. Generated
predictions and experiment artifacts are written under `results/`, which is
ignored by git.
