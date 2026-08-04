"""train.py - build data, train, evaluate, persist. Supports XGBoost, LightGBM, LogisticRegression."""
import argparse
from src.data import load_real_creditcard, engineer_features
from src.model import fit_and_evaluate
from src.evaluate import save_metrics, print_report
from src.persist import save_model


def tune_optuna(data, n_trials=20):
    """Optional Optuna hyperparameter tuning for XGBoost."""
    try:
        import optuna
        import xgboost as xgb
        import numpy as np
        from sklearn.metrics import average_precision_score
        from sklearn.model_selection import StratifiedKFold
        from src.core import Standardizer
    except ImportError:
        print("optuna not installed. Skipping hyperparameter tuning.")
        return None

    X = np.asarray(data["X"], float)
    y = np.asarray(data["y"], int)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pos_weight = float((y == 0).sum() / max(1, (y == 1).sum()))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": pos_weight,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        pr_aucs = []
        for train_idx, val_idx in skf.split(X, y):
            Xtr_f, Xval_f = X[train_idx], X[val_idx]
            ytr_f, yval_f = y[train_idx], y[val_idx]
            sc = Standardizer().fit(Xtr_f)
            model = xgb.XGBClassifier(**params)
            model.fit(sc.transform(Xtr_f), ytr_f)
            proba = model.predict_proba(sc.transform(Xval_f))[:, 1]
            pr_aucs.append(average_precision_score(yval_f, proba))
        return float(np.mean(pr_aucs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best PR-AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        default="logistic_regression",
        choices=["logistic_regression", "xgboost", "lightgbm"],
        help="Classifier backend",
    )
    parser.add_argument("--calibrate", action="store_true", help="Apply probability calibration")
    parser.add_argument("--chronological", action="store_true", help="Use time-aware split")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=20, help="Optuna trials")
    args = parser.parse_args()

    print("Loading dataset...")
    data = load_real_creditcard()

    best_params = None
    if args.tune:
        if args.algorithm != "xgboost":
            print("--tune only supports --algorithm xgboost; skipping.")
        else:
            # Tune on the TRAINING portion only, using the same split the final
            # evaluation uses. Tuning on the full dataset (train+test) would let
            # Optuna peek at test rows and inflate the reported test metrics.
            import numpy as np
            if args.chronological:
                from src.model import _chronological_split
                Xtr, Xte, ytr, yte = _chronological_split(
                    np.asarray(data["X"], float), np.asarray(data["y"], int), test_size=0.25)
            else:
                from src.core import train_test_split
                Xtr, Xte, ytr, yte = train_test_split(
                    np.asarray(data["X"], float), np.asarray(data["y"], int), 0.25, 7)
            tune_data = {"X": Xtr, "y": ytr, "features": data["features"]}
            print("Running Optuna hyperparameter tuning (train split only)...")
            best_params = tune_optuna(tune_data, n_trials=args.n_trials)

    print(f"Training {args.algorithm}...")
    print(f"  Calibrate={args.calibrate}, Chronological={args.chronological}")
    model, metrics = fit_and_evaluate(
        data, algorithm=args.algorithm,
        calibrate=args.calibrate, chronological=args.chronological,
        params=best_params,
    )

    save_model(model)
    suffix = f"_{args.algorithm}"
    save_metrics(metrics, f"models/metrics{suffix}.json")
    print_report(metrics)
    print(f"\nSaved model -> models/model{suffix}.pkl and metrics -> models/metrics{suffix}.json")


if __name__ == "__main__":
    main()
