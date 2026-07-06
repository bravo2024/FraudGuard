"""Deploy-friendly Streamlit dashboard for FraudGuard."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import make_synthetic
from src.model import fit_and_evaluate, predict_proba

RAW_CSV = ROOT / "data" / "raw" / "creditcard.csv"


st.set_page_config(
    page_title="FraudGuard | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)


st.markdown(
    """
    <style>
    .hero {
        padding: 1.25rem 1.4rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e40af 55%, #0369a1 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 { margin-bottom: 0.25rem; }
    .small-muted { color: #64748b; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt_metric(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


@st.cache_data(show_spinner=False)
def load_demo_data(use_real_file: bool, sample_rows: int, seed: int) -> dict:
    if use_real_file and RAW_CSV.exists():
        df = pd.read_csv(RAW_CSV)
        if len(df) > sample_rows:
            fraud = df[df["Class"] == 1]
            legitimate = df[df["Class"] == 0]
            fraud_count = min(len(fraud), max(20, sample_rows // 12))
            fraud_sample = fraud.sample(n=fraud_count, random_state=seed)
            legitimate_count = min(len(legitimate), sample_rows - len(fraud_sample))
            legitimate_sample = legitimate.sample(n=legitimate_count, random_state=seed)
            df = (
                pd.concat([fraud_sample, legitimate_sample], ignore_index=True)
                .sample(frac=1, random_state=seed)
                .reset_index(drop=True)
            )

        feature_frame = df.drop(columns=["Class"]).select_dtypes("number")
        return {
            "X": feature_frame.to_numpy(),
            "y": df["Class"].astype(int).to_numpy(),
            "features": list(feature_frame.columns),
            "frame": pd.concat([feature_frame, df["Class"].astype(int)], axis=1),
            "source": "ULB credit-card fraud sample from data/raw/creditcard.csv",
        }

    data = make_synthetic(n=sample_rows, seed=seed)
    frame = pd.DataFrame(data["X"], columns=data["features"])
    frame["Class"] = data["y"]
    data["frame"] = frame
    data["source"] = "Synthetic imbalanced fraud demo; no external data needed"
    return data


@st.cache_resource(show_spinner=False)
def train_cached(use_real_file: bool, sample_rows: int, seed: int) -> tuple[dict, dict]:
    data = load_demo_data(use_real_file, sample_rows, seed)
    return fit_and_evaluate(data, algorithm="logistic_regression")


def threshold_sweep(metrics: dict, false_negative_cost: int, false_positive_cost: int) -> pd.DataFrame:
    y_true = np.asarray(metrics.get("y_true", []), dtype=int)
    y_proba = np.asarray(metrics.get("y_proba", []), dtype=float)
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        predicted = (y_proba >= threshold).astype(int)
        tp = int(((predicted == 1) & (y_true == 1)).sum())
        fp = int(((predicted == 1) & (y_true == 0)).sum())
        tn = int(((predicted == 0) & (y_true == 0)).sum())
        fn = int(((predicted == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "business_cost": fn * false_negative_cost + fp * false_positive_cost,
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    return pd.DataFrame(rows)


st.markdown(
    """
    <div class="hero">
      <h1>FraudGuard</h1>
      <p>Real-time card-fraud scoring with class imbalance, threshold tuning, and business-cost framing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Run Settings")
    use_real_file = st.checkbox(
        "Use local creditcard.csv",
        value=RAW_CSV.exists(),
        help="If missing, the app automatically falls back to synthetic data for cloud deployment.",
    )
    sample_rows = st.slider("Rows used for demo training", 2_000, 60_000, 12_000, step=2_000)
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1)
    st.divider()
    st.caption("Deploy command")
    st.code("streamlit run app.py", language="bash")
    if RAW_CSV.exists():
        st.success("Local real CSV found")
    else:
        st.info("No local CSV found; synthetic mode is deploy-safe")

data = load_demo_data(use_real_file, sample_rows, seed)

with st.spinner("Training a cost-sensitive logistic baseline..."):
    model, metrics = train_cached(use_real_file, sample_rows, seed)

st.caption(data["source"])
metric_cols = st.columns(5)
metric_cols[0].metric("Rows", f"{len(data['y']):,}")
metric_cols[1].metric("Fraud Rate", f"{np.mean(data['y']):.2%}")
metric_cols[2].metric("ROC-AUC", fmt_metric(metrics["roc_auc"]))
metric_cols[3].metric("PR-AUC", fmt_metric(metrics["pr_auc"]))
metric_cols[4].metric("F1", fmt_metric(metrics["f1"]))

tab_business, tab_model, tab_threshold, tab_score, tab_deploy = st.tabs(
    ["Business Case", "Model Lab", "Threshold Simulator", "Score a Case", "Deploy"]
)

with tab_business:
    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("What makes this problem hard")
        st.markdown(
            """
            - Handles **extreme class imbalance**, the central challenge in fraud ML.
            - Optimizes for **business cost**, not vanity accuracy.
            - Uses a simple, explainable baseline before heavier models.
            - Ships a separate full notebook-style app in `streamlit_app.py` for deeper EDA/XAI storytelling.
            """
        )
    with right:
        class_counts = data["frame"]["Class"].value_counts().rename(index={0: "Legitimate", 1: "Fraud"})
        st.bar_chart(class_counts)
        st.caption("Fraud detection is won in the minority class, not overall accuracy.")

with tab_model:
    st.subheader("Model performance")
    show_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {"y_true", "y_proba"}
    }
    st.json(show_metrics)

    confusion = pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"],
    )
    st.dataframe(confusion, use_container_width=True)

    st.subheader("Dataset preview")
    st.dataframe(data["frame"].head(25), use_container_width=True)

with tab_threshold:
    st.subheader("Cost-sensitive threshold tuning")
    cost_left, cost_right = st.columns(2)
    false_negative_cost = cost_left.number_input(
        "Cost of missing fraud",
        min_value=1,
        value=100,
        step=10,
    )
    false_positive_cost = cost_right.number_input(
        "Cost of blocking a good transaction",
        min_value=1,
        value=10,
        step=5,
    )
    sweep = threshold_sweep(metrics, false_negative_cost, false_positive_cost)
    best = sweep.loc[sweep["business_cost"].idxmin()]
    best_cols = st.columns(4)
    best_cols[0].metric("Best Threshold", f"{best['threshold']:.2f}")
    best_cols[1].metric("Minimum Cost", f"${best['business_cost']:,.0f}")
    best_cols[2].metric("Recall", f"{best['recall']:.2%}")
    best_cols[3].metric("Precision", f"{best['precision']:.2%}")
    st.line_chart(sweep.set_index("threshold")[["business_cost"]])
    st.dataframe(sweep.sort_values("business_cost").head(10), use_container_width=True)

with tab_score:
    st.subheader("Score a transaction-like case")
    frame = data["frame"].drop(columns=["Class"])
    selected_row = st.slider("Pick an observed row as a what-if case", 0, len(frame) - 1, 0)
    candidate = frame.iloc[[selected_row]].copy()

    if "Amount" in candidate.columns:
        multiplier = st.slider("Amount multiplier", 0.2, 5.0, 1.0, step=0.1)
        candidate["Amount"] = candidate["Amount"] * multiplier
    else:
        multiplier = st.slider("Feature shock multiplier", 0.2, 5.0, 1.0, step=0.1)
        first_feature = candidate.columns[0]
        candidate[first_feature] = candidate[first_feature] * multiplier

    fraud_probability = float(predict_proba(model, candidate.to_numpy())[0])
    st.metric("Fraud Probability", f"{fraud_probability:.2%}")
    st.progress(min(max(fraud_probability, 0.0), 1.0))
    st.dataframe(candidate, use_container_width=True)

with tab_deploy:
    st.subheader("Deployment checklist")
    st.markdown(
        """
        1. Push this `FraudGuard/FraudGuard` folder to GitHub.
        2. On Streamlit Community Cloud, choose `app.py` for the fast demo.
        3. Keep `data/raw/` out of Git for small deployments; this app falls back gracefully.
        4. Use `streamlit_app.py` when you want the full notebook-mirroring EDA/XAI version.
        """
    )
    st.code(
        "pip install -r requirements.txt\n"
        "streamlit run app.py\n"
        "# optional deep-dive app:\n"
        "streamlit run streamlit_app.py",
        language="bash",
    )
