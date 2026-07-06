# FraudGuard

Credit-card fraud detection on the [ULB dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions, 0.17% fraud). The interesting part of this problem isn't
the model — it's dealing with extreme class imbalance and picking a decision
threshold that reflects the asymmetric cost of missing fraud vs. blocking a
legitimate customer.

## Results

Held-out test set (71,201 transactions):

| Model | ROC-AUC | PR-AUC | F1 @ 0.5 |
|---|---|---|---|
| Logistic regression (class-weighted) | 0.976 | 0.680 | 0.114 |
| LightGBM | 0.979 | 0.845 | 0.858 |

The gap between ROC-AUC and PR-AUC is the whole story: with 0.17% positives,
ROC-AUC looks great for almost anything, while PR-AUC actually separates the
models. The logistic baseline's F1 collapses at the default threshold because
class weighting shifts its probabilities — which is why the app tunes the
threshold against a cost matrix instead of using 0.5.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Drop the Kaggle CSV at `data/raw/creditcard.csv`. Without it, training and the
app fall back to a small synthetic sample so everything still runs — but the
numbers above come from the real data.

## Usage

```bash
python train.py --algorithm lightgbm            # train + save metrics
python train.py --algorithm xgboost --tune      # Optuna search (20 trials)
python train.py --algorithm logistic_regression --calibrate --chronological
streamlit run app.py                            # scoring dashboard
streamlit run streamlit_app.py                  # full EDA/benchmark dashboard
pytest -q                                       # smoke test
```

`--chronological` splits by transaction time instead of randomly. Random
splits leak future transactions into training on this dataset, so the
time-based number is the honest one.

## Layout

```
src/
  data.py       loading + feature engineering (Amount log-transform, time-of-day)
  model.py      LR / XGBoost / LightGBM backends, calibration, chrono split
  evaluate.py   metrics report + persistence
  core.py       scaler and metric helpers
train.py        CLI entrypoint
app.py          light dashboard (deploys on Streamlit Cloud free tier)
streamlit_app.py  heavier dashboard with EDA and model comparison
notebooks/      end-to-end walkthrough
```

## Known limitations

- The dataset covers two days of European transactions; nothing here is
  validated for drift over longer horizons.
- Features V1–V28 are PCA-anonymised, so per-feature interpretability is
  limited to "component 14 matters", not *why*.
- No online/streaming scoring — batch only.
