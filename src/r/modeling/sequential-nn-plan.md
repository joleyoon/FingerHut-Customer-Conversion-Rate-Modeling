# Plan: Sequence Modeling for User Journeys (RNN / Transformer)

This plan outlines how to incorporate deep sequence models—such as **RNNs (LSTMs)** or **Transformers (BERT-like)**—to model the exact sequential path of actions (the "journey") that each user takes before making a purchase.

---

## 1. Sequence Representation of Journeys
Currently, the journey data is flattened by taking counts of each event (`n_*`) and calculating global summary stats (e.g., `journey_length_s`, `days_inactive`). This aggregates out the order and temporal transitions of actions.

To feed user journeys into deep sequential models, we represent each user $u$ as a sequence of events and elapsed time gaps up to their cutoff time:
1. **Event Tokens**: Map each unique event name to a unique integer ID (e.g., `1: browse_products`, `2: add_to_cart`, `3: begin_checkout`, etc., reserving `0` for padding).
2. **Sequence Padding/Truncation**: Pad or truncate each sequence to a maximum sequence length $M$ (e.g., $M = 100$).
3. **Time Gaps (Optional but recommended)**: Calculate the time difference $\Delta t_i$ between consecutive events, log-transform it to handle extreme variance, and pass it as a continuous sequence feature along with the event tokens.

---

## 2. Proposed Model Architectures

### Architecture A: Recurrent Neural Network (LSTM / GRU)
An LSTM is highly effective for sequence classification and represents a robust baseline for sequential path data.
* **Input**: Sequence of event IDs of size $(B, M)$ and time gaps of size $(B, M, 1)$.
* **Embedding Layer**: Maps event IDs to dense embedding vectors of size $D$ (e.g., 32 or 64).
* **Feature Fusion**: Concatenate event embeddings with the continuous time-gap feature.
* **LSTM Layer**: Pass the sequence through a unidirectional or bidirectional LSTM (e.g., 64 hidden units).
* **Classification Head**: Extract the final hidden state (or apply global average/max pooling over sequence steps) and pass through a Dense layer with a Sigmoid activation to predict the probability of success.

### Architecture B: Transformer Encoder (BERT-like)
A Transformer uses self-attention to capture interactions between actions regardless of their distance in the sequence.
* **Embedding Layer**: Event token embeddings + learnable positional embeddings.
* **Transformer Blocks**: 2–4 layers of Multi-Head Self-Attention (with padding masks to ignore padded time steps).
* **Pooling / Classification**: Prepend a special `[CLS]` token to the sequence (like BERT) or apply mean pooling across the non-padded tokens, followed by a linear classification layer.

---

## 3. Implementation Stack Options

Since the project uses both **R** (in Quarto files) and **Python** (via Joel's scripts), we can implement this in either ecosystem:

### Option 1: Python + PyTorch (Highly Recommended)
* **Why**: PyTorch has the most mature ecosystem for building sequence models, custom datasets, and attention masks.
* **Integration**: The sequence experiments in `src/python/experiments/` read the Parquet data, train the PyTorch models, and save predictions and model artifacts under `results/experiments/`.

### Option 2: R + `torch`
* **Why**: Keeps all of Charlie's work inside R Quarto files using Posit's native `torch` package.
* **Integration**: We can write a new Quarto document `04_sequence_modelling.qmd` using `library(torch)` to build the custom Dataset, LSTM/Transformer modules, and training loop.

---

## 4. Concrete Step-by-Step Implementation Steps

### Step 1: Sequential Data Preparation
1. Create an event-to-index vocabulary mapping from `dt_clean2.parquet`.
2. Extract the ordered sequence of events for each user `id` up to their `cutoff_time`.
3. Compute the time differences between events for each user.
4. Save the sequences as padded arrays (tensors) along with the `final_outcome` labels.

### Step 2: Model and Dataset Definition
* Define the PyTorch/R-torch Dataset class that handles loading the sequences, applying padding masks, and delivering batches.
* Implement either the **LSTM** or **Transformer** network module with dropout to prevent overfitting.

### Step 3: Model Training with Validation
* Split users into train and validation sets using the same stratified group split used in the baseline.
* Train using Binary Cross Entropy Loss and Adam optimizer.
* Since the dataset is extremely large (~1.57 million users), we can either use the balanced downsampled set (5% success) or train on the full set with weighted loss.

### Step 4: Model Evaluation and Inference
* Evaluate the sequence model using PR-AUC and Brier score.
* Generate purchase probabilities for the test set.
* **Ensembling**: Compare and average predictions with the tuned XGBoost model from `03_modelling.qmd` for a powerful ensemble model.

---

## 5. Next Action
Please choose your preferred implementation path to proceed:
- **Option A**: Implement using **Python and PyTorch** (creating a Python modeling script).
- **Option B**: Implement using **R and the `torch` package** (creating a Quarto document).
