"""Fraud classifiers: logistic regression baseline plus gradient-boosted trees.

The dataset is heavily imbalanced (~0.17% fraud), so every backend weights the
positive class and evaluation centres on PR-AUC rather than accuracy. The
chronological split option exists because random splits leak future
transactions into training, which inflates metrics on time-ordered data.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

from src.core import Standardizer, train_test_split, roc_auc_score, accuracy_score, f1_score


def _chronological_split(X, y, test_size=0.25):
    # ULB creditcard.csv has Time as the last engineered column; fall back to
    # row order for synthetic data, which is generated sequentially anyway.
    time_col = X[:, -1] if X.shape[1] > 29 else np.arange(len(X))
    order = np.argsort(time_col)
    X_sorted, y_sorted = X[order], y[order]
    n_test = int(len(X_sorted) * test_size)
    return (X_sorted[:-n_test], X_sorted[-n_test:],
            y_sorted[:-n_test], y_sorted[-n_test:])


def _estimator(algorithm, ytr, params=None):
    """Build the requested classifier, weighting positives from the actual
    training class ratio rather than a hardcoded constant."""
    pos_weight = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    params = params or {}

    if algorithm == "xgboost":
        import xgboost as xgb
        defaults = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                        eval_metric="logloss", random_state=42)
        defaults.update(params)
        defaults.setdefault("scale_pos_weight", pos_weight)
        return "xgboost", xgb.XGBClassifier(**defaults)

    if algorithm == "lightgbm":
        import lightgbm as lgb
        defaults = dict(n_estimators=300, learning_rate=0.05,
                        class_weight="balanced", random_state=42, verbose=-1)
        defaults.update(params)
        return "lightgbm", lgb.LGBMClassifier(**defaults)

    from sklearn.linear_model import LogisticRegression
    return "logistic_regression", LogisticRegression(
        class_weight="balanced", max_iter=400, random_state=7)


def fit_and_evaluate(data, algorithm="logistic_regression", calibrate=False,
                     chronological=False, params=None):
    """Train one backend and evaluate on a held-out slice.

    data: dict with 'X', 'y', 'features'.
    algorithm: 'logistic_regression', 'xgboost' or 'lightgbm'.
    calibrate: wrap the estimator in isotonic calibration (3-fold). Worth it
        when the scores feed a cost-based threshold rather than a ranking.
    chronological: split by transaction time instead of randomly.
    params: optional hyperparameter overrides, e.g. from Optuna tuning.

    Returns (model dict, metrics dict). The model dict keeps the fitted scaler
    alongside the estimator so predict_proba() applies the same preprocessing.
    """
    X = np.asarray(data["X"], float)
    y = np.asarray(data["y"], int)

    if chronological:
        Xtr, Xte, ytr, yte = _chronological_split(X, y, test_size=0.25)
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, 0.25, 7)

    sc = Standardizer().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    backend, est = _estimator(algorithm, ytr, params)

    if calibrate:
        from sklearn.calibration import CalibratedClassifierCV
        est = CalibratedClassifierCV(est, method="isotonic", cv=3)
        backend += "_calibrated"

    est.fit(Xtr_s, ytr)
    proba = est.predict_proba(Xte_s)[:, 1]
    pred = (proba >= 0.5).astype(int)

    cm = confusion_matrix(yte, pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    metrics = {
        "backend": backend,
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "roc_auc": roc_auc_score(yte, proba),
        "pr_auc": float(average_precision_score(yte, proba)),
        "accuracy": accuracy_score(yte, pred),
        "f1": f1_score(yte, pred),
        "positive_rate": float(yte.mean()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "y_true": yte.tolist(),
        "y_proba": proba.tolist(),
    }

    return {"scaler": sc, "estimator": est, "backend": backend,
            "features": data.get("features")}, metrics


def predict_proba(model, X):
    Xs = model["scaler"].transform(np.asarray(X, float))
    return model["estimator"].predict_proba(Xs)[:, 1]
