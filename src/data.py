"""data.py - synthetic fallback and real dataset loader with feature engineering."""
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ["feat_%02d" % i for i in range(12)]


def make_synthetic(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    d = len(FEATURES)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d) * (rng.random(d) < 0.5)
    logits = X @ w + 0.6 * X[:, 0] * X[:, 1] - 1.4
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    return {"X": X, "y": y, "features": FEATURES}


def load_real_creditcard():
    import pandas as pd
    import urllib.request

    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "creditcard.csv"

    if not csv_path.exists():
        print(f"Downloading real dataset from {url} ...")
        urllib.request.urlretrieve(url, csv_path)

    df = pd.read_csv(csv_path)
    target = "Class"
    num = df.drop(columns=[target]).select_dtypes("number")
    return {"X": num.to_numpy(), "y": df[target].astype(int).to_numpy(), "features": list(num.columns)}


def engineer_features(df, time_col="Time", amount_col="Amount"):
    """Apply feature engineering used by top Kaggle notebooks."""
    dfe = df.copy()

    # 1. Log transform amount
    dfe["log_amount"] = np.log1p(dfe[amount_col] + 1e-6)

    # 2. Time-based features
    if time_col in dfe.columns:
        hour = (dfe[time_col] // 3600) % 24
        dfe["hour"] = hour
        dfe["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        dfe["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        dfe["day_segment"] = pd.cut(
            hour, bins=[0, 6, 12, 18, 24],
            labels=["Night", "Morning", "Afternoon", "Evening"]
        )
        # Weekend proxy (last 16h of the 48h dataset are "weekend-like")
        total_seconds = dfe[time_col]
        dfe["is_weekend"] = (total_seconds > (48 - 16) * 3600).astype(int)

    # 3. Interaction features (common in top Kaggle notebooks)
    for col in ["V14", "V4", "V12", "V10", "V11"]:
        if col in dfe.columns:
            dfe[f"{col}_x_Amount"] = dfe[col] * dfe[amount_col]

    # 4. Velocity: deviation from median amount by hour
    if "hour" in dfe.columns:
        medians = dfe.groupby("hour")[amount_col].transform("median")
        dfe["amount_deviation"] = dfe[amount_col] - medians

    # 5. Z-score outliers for key features (from ispromadhka's notebook)
    for col in ["V14", "V12", "V10"]:
        if col in dfe.columns:
            mu, sd = dfe[col].mean(), dfe[col].std()
            dfe[f"{col}_zscore"] = (dfe[col] - mu) / sd

    return dfe


def load_real(csv_name, target):
    import pandas as pd
    df = pd.read_csv(Path("data/raw") / csv_name)
    num = df.drop(columns=[target]).select_dtypes("number")
    return {"X": num.to_numpy(), "y": df[target].astype(int).to_numpy(), "features": list(num.columns)}


if __name__ == "__main__":
    d = load_real_creditcard()
    print("Real X", d["X"].shape, "pos", int(d["y"].sum()))
