import sys, os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath("src"))

st.set_page_config(page_title="FraudGuard", page_icon="", layout="wide")
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore")

st.markdown("""
<style>
    .big { font-size: 2.2rem; font-weight: 700; color: #1f2937; }
    .sub { font-size: 1.1rem; color: #6b7280; margin-bottom: 16px; }
    .step-header { font-size: 1.5rem; font-weight: 600; color: #111827; border-bottom: 2px solid #3b82f6; padding-bottom: 4px; margin-bottom: 12px; }
    .metric-card { background: #f9fafb; border-radius: 10px; padding: 16px; border-left: 4px solid #3b82f6; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    from src.data import load_real_creditcard
    return load_real_creditcard()


@st.cache_data
def load_metrics():
    files = {
        "Logistic Regression": Path("models/metrics_lr.json"),
        "LightGBM": Path("models/metrics_lgb.json"),
        "XGBoost": Path("models/metrics_xgboost.json"),
    }
    return {name: json.load(open(p)) for name, p in files.items() if p.exists()}


@st.cache_resource
def train_models():
    from src.model import fit_and_evaluate
    from src.persist import save_model
    from src.evaluate import save_metrics
    d = load_data()
    results = {}
    with st.status("Training Logistic Regression...") as s:
        m, met = fit_and_evaluate(d, algorithm="logistic_regression")
        save_model(m, "models/model_lr.pkl")
        save_metrics(met, "models/metrics_lr.json")
        s.success(f"LR done | PR-AUC: {met['pr_auc']:.4f}")
        results["Logistic Regression"] = (m, met)
    with st.status("Training XGBoost (scale_pos_weight=577)...") as s:
        m, met = fit_and_evaluate(d, algorithm="xgboost")
        save_model(m, "models/model_xgboost.pkl")
        save_metrics(met, "models/metrics_xgboost.json")
        s.success(f"XGB done | PR-AUC: {met['pr_auc']:.4f}")
        results["XGBoost"] = (m, met)
    with st.status("Training LightGBM...") as s:
        m, met = fit_and_evaluate(d, algorithm="lightgbm")
        save_model(m, "models/model_lgb.pkl")
        save_metrics(met, "models/metrics_lgb.json")
        s.success(f"LGB done | PR-AUC: {met['pr_auc']:.4f}")
        results["LightGBM"] = (m, met)
    return results


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("# FraudGuard")
st.sidebar.caption("AI-Driven Credit Card Fraud Detection")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Notebook Steps",
    [
        "1. Business Context",
        "2. Data Loading & Inspection",
        "3. Exploratory Data Analysis",
        "4. Feature Engineering",
        "5. Model Training & Evaluation",
        "6. Cost-Sensitive Threshold",
        "7. SHAP Explainability",
    ],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "This app mirrors the **FraudGuard_Master.ipynb** notebook. "
    "Each step corresponds to a section in the notebook."
)

# =========================================================================
# PAGE 1 — Business Context & Problem Framing
# =========================================================================
if page == "1. Business Context":
    st.markdown("<div class='big'>1. Business Context & Problem Framing</div>", unsafe_allow_html=True)
    st.markdown(
        "**Objective:** Build an end-to-end, production-ready machine learning pipeline for credit card fraud detection, "
        "targeting entry-level AI/ML roles in the banking sector."
    )
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### The Problem")
        st.markdown(
            "Credit card fraud is a multi-billion dollar problem for financial institutions. "
            "A key challenge is the **extreme class imbalance** — fraudulent transactions typically "
            "account for less than 0.2% of all transactions."
        )
        st.latex(r"\text{Imbalance Ratio (IR)} = \frac{N_{\text{majority}}}{N_{\text{minority}}} \approx 577:1")
    with col2:
        st.markdown("### Business Costs")
        st.markdown("Optimizing for accuracy is **dangerous** in fraud detection. We optimize for business cost:")
        st.markdown("- **False Negative (FN):** Missing fraud = **$100 per transaction**")
        st.markdown("- **False Positive (FP):** Blocking legitimate = **$10 per transaction**")
        st.markdown("- **Goal:** Minimize total cost, not just maximize accuracy")
    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown(
        "**ULB Credit Card Fraud Detection** — 284,807 transactions, 492 frauds (0.172%), "
        "September 2013 by European cardholders over 2 days. Features V1–V28 are PCA-transformed "
        "for confidentiality, plus `Time` (seconds elapsed) and `Amount` (transaction value)."
    )
    with st.expander("View key business metrics"):
        st.markdown("""
        | Metric | Value |
        |---|---|
        | Total transactions | 284,807 |
        | Fraudulent transactions | 492 |
        | Fraud rate | 0.172% |
        | Imbalance ratio | ~577:1 |
        """)

# =========================================================================
# PAGE 2 — Data Loading & Initial Inspection
# =========================================================================
elif page == "2. Data Loading & Inspection":
    st.markdown("<div class='big'>2. Data Loading & Initial Inspection</div>", unsafe_allow_html=True)
    with st.spinner("Loading dataset..."):
        d = load_data()
    df = pd.DataFrame(d["X"], columns=d["features"])
    df["Class"] = d["y"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Shape", f"{df.shape[0]:,} rows × {df.shape[1]} cols")
    col2.metric("Fraud Rate", f"{df['Class'].mean():.4%}")
    col3.metric("Fraud Count", f"{int(df['Class'].sum()):,}")
    st.markdown("### First 5 rows")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown("### Data Summary")
    st.dataframe(df.describe(), use_container_width=True)

# =========================================================================
# PAGE 3 — Exploratory Data Analysis
# =========================================================================
elif page == "3. Exploratory Data Analysis":
    st.markdown("<div class='big'>3. Exploratory Data Analysis</div>", unsafe_allow_html=True)
    st.markdown("Understanding the data is crucial before modeling. We analyze the class imbalance, "
                "feature correlations, and the temporal nature of fraud.")
    d = load_data()
    df = pd.DataFrame(d["X"], columns=d["features"])
    df["Class"] = d["y"]
    tab1, tab2, tab3, tab4 = st.tabs([
        "Class Imbalance & Time Analysis",
        "Transaction Amount Analysis",
        "Correlation Analysis",
        "Cyclical Time Encoding",
    ])
    # --- Tab 1: Class Imbalance & Time Analysis ---
    with tab1:
        st.markdown("#### Class Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        counts = df["Class"].value_counts()
        sns.barplot(x=counts.index, y=counts.values, ax=axes[0], palette=["#3498db", "#e74c3c"])
        axes[0].set_xticklabels(["Normal (0)", "Fraud (1)"])
        axes[0].set_title("Transaction Class Distribution")
        for i, v in enumerate(counts.values):
            axes[0].text(i, v + 500, str(v), ha="center", va="bottom")
        # Fraud rate by hour
        df["Hour"] = (df["Time"] // 3600) % 24
        fraud_by_hour = df.groupby("Hour")["Class"].mean()
        sns.barplot(x=fraud_by_hour.index, y=fraud_by_hour.values, ax=axes[1], color="#e74c3c")
        axes[1].set_title("Fraud Rate by Hour of Day")
        axes[1].set_xlabel("Hour (0–23)")
        axes[1].set_ylabel("Fraud Rate")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Fraud rates spike during off-hours (late night / early morning), a common behavioral pattern.")
    # --- Tab 2: Transaction Amount Analysis ---
    with tab2:
        st.markdown("#### Transaction Amount Distribution (Log Scale)")
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        sns.histplot(
            df[df["Class"] == 0]["Amount"], bins=50, ax=axes[0],
            label="Normal", color="#3498db", log_scale=True, stat="density", alpha=0.6,
        )
        sns.histplot(
            df[df["Class"] == 1]["Amount"], bins=50, ax=axes[0],
            label="Fraud", color="#e74c3c", log_scale=True, stat="density", alpha=0.6,
        )
        axes[0].set_title("Transaction Amount Distribution (Log Scale)")
        axes[0].set_xlabel("Amount (USD)")
        axes[0].legend()
        # Box plot of key features
        sample_df = df.sample(n=20000, random_state=42)
        melted = pd.melt(sample_df, id_vars=["Class"], value_vars=["V14", "V4", "V12", "V10"])
        sns.boxplot(x="variable", y="value", hue="Class", data=melted, ax=axes[1],
                    palette=["#3498db", "#e74c3c"])
        axes[1].set_title("Distribution of Key PCA Components by Class")
        plt.tight_layout()
        st.pyplot(fig)
    # --- Tab 3: Correlation Analysis ---
    with tab3:
        st.markdown("#### Feature Correlation Matrix")
        with st.spinner("Computing correlation matrix on a 50k sample..."):
            sample = df.sample(n=50000, random_state=42)
            corr = sample.corr()
        fig, ax = plt.subplots(figsize=(20, 15))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)
        st.caption("V1–V28 are PCA components, so they are mostly uncorrelated. "
                   "`Amount` and `Time` show low correlation with other features.")
    # --- Tab 4: Cyclical Time Encoding ---
    with tab4:
        st.markdown("#### Cyclical Encoding of Time (from top Kaggle notebooks)")
        st.markdown(
            "Treating hour as a linear feature (0→23) falsely implies that hour 23 is far from hour 0. "
            "**Cyclical encoding** maps hour onto a unit circle using sin/cos, preserving circular continuity."
        )
        st.latex(r"\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right)")
        st.latex(r"\text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)")
        hours = np.arange(0, 24)
        hour_sin = np.sin(2 * np.pi * hours / 24)
        hour_cos = np.cos(2 * np.pi * hours / 24)
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        axes[0].scatter(hour_sin, hour_cos, c=hours, cmap="viridis", s=200, edgecolor="black")
        for h, s, c in zip(hours, hour_sin, hour_cos):
            axes[0].annotate(str(int(h)), (s, c), textcoords="offset points", xytext=(5, 5), fontsize=9)
        axes[0].set_xlabel("sin(2πh/24)")
        axes[0].set_ylabel("cos(2πh/24)")
        axes[0].set_title("Cyclical Encoding: Hour Mapped to Unit Circle")
        axes[0].set_aspect("equal")
        axes[0].grid(True, alpha=0.3)
        # Fraud rate by cyclical encoding
        df_hour = df.copy()
        df_hour["hour"] = (df_hour["Time"] // 3600) % 24
        df_hour["hour_sin"] = np.sin(2 * np.pi * df_hour["hour"] / 24)
        df_hour["hour_cos"] = np.cos(2 * np.pi * df_hour["hour"] / 24)
        fraud_by_hour = df_hour.groupby("hour")["Class"].mean().values
        sc = axes[1].scatter(
            df_hour["hour_sin"], df_hour["hour_cos"],
            c=df_hour["Class"], cmap="coolwarm", alpha=0.1, s=1
        )
        axes[1].set_xlabel("sin(2πh/24)")
        axes[1].set_ylabel("cos(2πh/24)")
        axes[1].set_title("All Transactions in Cyclical Space (red=fraud)")
        plt.colorbar(sc, ax=axes[1], label="Class")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Cyclical encoding ensures hour 23 and hour 0 are adjacent, which linear encoding fails to capture.")

# =========================================================================
# PAGE 4 — Feature Engineering
# =========================================================================
elif page == "4. Feature Engineering":
    st.markdown("<div class='big'>4. Advanced Feature Engineering</div>", unsafe_allow_html=True)
    st.markdown("Raw features are rarely sufficient. We derive new features that capture behavioral patterns "
                "and non-linear relationships.")
    d = load_data()
    df = pd.DataFrame(d["X"], columns=d["features"])
    df["Class"] = d["y"]
    # Apply advanced feature engineering (mirroring top Kaggle notebooks)
    from src.data import engineer_features
    df_fe = engineer_features(df)
    for col in df_fe.columns:
        df[col] = df_fe[col]
    feature_cols = [c for c in df.columns if c not in ["Class", "Time", "day_segment"]]
    new_features = [c for c in feature_cols if c not in d["features"]]
    st.metric("Total Features after Engineering", len(feature_cols))
    st.markdown("### New Features Created")
    st.code("\n".join(f"  • {f}" for f in new_features))
    with st.expander("Show engineered dataset sample"):
        st.dataframe(df[feature_cols + ["Class"]].head(), use_container_width=True)
    st.session_state["engineered_df"] = df
    st.session_state["feature_cols"] = feature_cols

# =========================================================================
# PAGE 5 — Model Training & Evaluation
# =========================================================================
elif page == "5. Model Training & Evaluation":
    st.markdown("<div class='big'>5. Model Training & Evaluation</div>", unsafe_allow_html=True)
    st.markdown(
        "We establish a baseline with **Logistic Regression**, then benchmark against "
        "**LightGBM** which natively handles non-linearities. We use `class_weight='balanced'` "
        "to mitigate the extreme class imbalance."
    )
    st.info(
        "**Why stratified split?** In imbalanced data, a random split might place all fraud cases "
        "in one fold. Stratification preserves the class distribution across train and test sets."
    )
    col_left, col_right = st.columns([2, 1])
    with col_left:
        if st.button("Run Full Benchmark (LR + XGBoost + LightGBM)", type="primary"):
            results = train_models()
            st.success("Benchmarking complete! See results below.")
    with col_right:
        if st.button("Train with 5-fold CV", help="Train with cross-validation for more robust metrics"):
            with st.spinner("Training XGBoost with 5-fold CV..."):
                from sklearn.model_selection import StratifiedKFold
                from sklearn.metrics import average_precision_score
                from src.core import Standardizer
                from src.evaluate import save_metrics
                from src.persist import save_model
                import xgboost as xgb
                d = load_data()
                X, y = d["X"], d["y"]
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                pr_aucs, models = [], []
                for train_idx, val_idx in skf.split(X, y):
                    sc = Standardizer().fit(X[train_idx])
                    m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, scale_pos_weight=577, use_label_encoder=False, eval_metric="logloss", random_state=42)
                    m.fit(sc.transform(X[train_idx]), y[train_idx])
                    proba = m.predict_proba(sc.transform(X[val_idx]))[:, 1]
                    pr_aucs.append(average_precision_score(y[val_idx], proba))
                    models.append(m)
                best_idx = int(np.argmax(pr_aucs))
                st.success(f"5-fold CV complete | Mean PR-AUC: {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")

    st.markdown("**Best practices from Kaggle:** Chronological split (time-based), calibrated probabilities, and XGBoost with `scale_pos_weight`.")
    metrics = load_metrics()
    if metrics:
        df_met = pd.DataFrame({
            name: {k: v[k] for k in ["pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy"] if k in v}
            for name, v in metrics.items()
        }).T
        st.markdown("### Model Performance Benchmark")
        st.dataframe(df_met.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        df_met.plot(kind="bar", ax=ax, colormap="viridis")
        ax.set_ylim(0, 1.0)
        ax.set_title("Model Performance Comparison")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        # Confusion matrices
        st.markdown("### Confusion Matrices")
        cols = st.columns(len(metrics))
        for idx, (name, m) in enumerate(metrics.items()):
            with cols[idx]:
                st.markdown(f"**{name}**")
                st.markdown(f"TP: {m['tp']} | FP: {m['fp']}  \nFN: {m['fn']} | TN: {m['tn']}")
    else:
        st.warning("No trained models found. Click the button above to run the benchmark.")

# =========================================================================
# PAGE 6 — Cost-Sensitive Threshold Optimization
# =========================================================================
elif page == "6. Cost-Sensitive Threshold":
    st.markdown("<div class='big'>6. Cost-Sensitive Threshold Optimization</div>", unsafe_allow_html=True)
    st.markdown(
        "Optimizing for accuracy in fraud detection is dangerous. A model that predicts 'Normal' "
        "for everything gets 99.8% accuracy but catches 0% fraud. We must optimize for **business cost**."
    )
    st.markdown("**Assumed Costs (simulated):**")
    st.markdown("- **FN:** Missing fraud = **$100** | **FP:** Blocking legitimate = **$10**")
    fn_cost = st.number_input("False Negative Cost ($)", min_value=1, value=100, step=10)
    fp_cost = st.number_input("False Positive Cost ($)", min_value=1, value=10, step=5)
    metrics = load_metrics()
    if metrics:
        best_name = max(metrics, key=lambda n: metrics[n].get("pr_auc", 0))
        best = metrics[best_name]
        st.info(f"Best model by PR-AUC: **{best_name}**")
        y_true = np.array(best.get("y_true", []))
        y_proba = np.array(best.get("y_proba", []))
        if len(y_true) > 0 and len(y_proba) > 0:
            thresholds = np.linspace(0.01, 0.99, 100)
            costs = []
            from sklearn.metrics import confusion_matrix
            for t in thresholds:
                pred = (y_proba >= t).astype(int)
                cm = confusion_matrix(y_true, pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn, fp, fn, tp = 0, 0, 0, 0
                costs.append(fn * fn_cost + fp * fp_cost)
            opt_idx = int(np.argmin(costs))
            opt_thresh = thresholds[opt_idx]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(thresholds, costs, color="#e74c3c", linewidth=2)
            ax.axvline(opt_thresh, color="#2ecc71", linestyle="--",
                       label=f"Optimal Threshold = {opt_thresh:.3f}")
            ax.set_xlabel("Decision Threshold")
            ax.set_ylabel("Total Business Cost (USD)")
            ax.set_title("Cost-Sensitive Threshold Optimization")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            c1, c2 = st.columns(2)
            c1.metric("Optimal Threshold", f"{opt_thresh:.4f}")
            c2.metric("Minimum Cost", f"${costs[opt_idx]:,.2f}")
        else:
            st.warning("y_true / y_proba not found in metrics. Re-run training.")
    else:
        st.warning("No metrics found. Run Model Training first.")

# =========================================================================
# PAGE 7 — SHAP Explainability
# =========================================================================
elif page == "7. SHAP Explainability":
    st.markdown("<div class='big'>7. SHAP Explainability (XAI)</div>", unsafe_allow_html=True)
    st.markdown(
        "Banking regulators demand **'White-Box AI'**. SHAP (SHapley Additive exPlanations) uses "
        "game theory to explain the marginal contribution of each feature to a prediction. "
        "This is critical for justifying why a transaction was blocked."
    )
    st.latex(
        r"\phi_i = \sum_{S \subseteq N \setminus \{i\}} "
        r"\frac{|S|!\,(n - |S| - 1)!}{n!} \big( v(S \cup \{i\}) - v(S) \big)"
    )
    metrics = load_metrics()
    if metrics:
        best_name = max(metrics, key=lambda n: metrics[n].get("pr_auc", 0))
        st.info(f"Using best model: **{best_name}**")
        try:
            import shap
            import pickle
            if "XGBoost" in best_name:
                model_path = Path("models/model_xgboost.pkl")
            elif "LightGBM" in best_name:
                model_path = Path("models/model_lgb.pkl")
            else:
                model_path = Path("models/model_lr.pkl")
            if model_path.exists():
                artifact = pickle.load(open(model_path, "rb"))
                model = artifact.get("model") or artifact.get("estimator")
                scaler = artifact.get("scaler")
                features = artifact.get("features", [f"V{i}" for i in range(1, 30)] + ["Amount"])
                d = load_data()
                X = d["X"][:500]  # sample for speed
                X_s = scaler.transform(X) if scaler else X
                model_type = str(type(model)).lower()
                if "lgbm" in model_type or "lightgbm" in model_type or "xgb" in model_type:
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(X_s)
                    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
                    st.markdown("#### SHAP Summary Plot (Top 15 Features)")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(sv, X, feature_names=features, show=False, max_display=15, ax=ax)
                    st.pyplot(fig)
                    st.caption("Features are ranked by absolute SHAP value. Red = high feature value, Blue = low.")
                else:
                    st.info("SHAP summary plot is most informative for tree-based models (XGBoost/LightGBM).")
                # Save best model
                st.markdown("---")
                if st.button("Save Best Model as master_model.pkl"):
                    threshold = None
                    # Recompute optimal threshold
                    y_true = np.array(metrics[best_name].get("y_true", []))
                    y_proba = np.array(metrics[best_name].get("y_proba", []))
                    if len(y_true) > 0 and len(y_proba) > 0:
                        from sklearn.metrics import confusion_matrix
                        thresh = np.linspace(0.01, 0.99, 100)
                        costs = [(confusion_matrix(y_true, (y_proba >= t).astype(int)).ravel())
                                 for t in thresh]
                        valid = [c for c in costs if len(c) == 4]
                        if valid:
                            costs_total = [fn * 100 + fp * 10 for tn, fp, fn, tp in valid]
                            threshold = float(thresh[np.argmin(costs_total)])
                    artifact = {
                        "model": model,
                        "scaler": scaler,
                        "features": features,
                        "threshold": threshold or 0.5,
                        "metrics": metrics[best_name],
                        "name": best_name,
                    }
                    Path("models/master_model.pkl").parent.mkdir(parents=True, exist_ok=True)
                    pickle.dump(artifact, open("models/master_model.pkl", "wb"))
                    st.success(f"Best model ({best_name}) saved to models/master_model.pkl")
            else:
                st.warning(f"Model file not found: {model_path}. Run training first.")
        except ImportError:
            st.warning("SHAP library not installed. Run: pip install shap")
    else:
        st.warning("No metrics found. Run Model Training first (Page 5).")

st.markdown("---")
st.caption("FraudGuard v1.0 | Mirrors notebooks/FraudGuard_Master.ipynb | Built for JPMorgan Chase AI/ML Recruitment")
