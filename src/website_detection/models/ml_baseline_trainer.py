from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

SEED = 42
ALLOWED_PREFIXES = ("url_", "dom_", "cred_", "net_", "beh_", "graph_")
EXCLUDED_COLUMNS = {
    "net_external_domains",
    "beh_server_redirect_chain",
    "beh_initial_response_url",
    "graph_artifact_saved",
    "graph_artifact_path",
    "graph_artifact_file_size",
    "graph_artifact_reused",
    "graph_artifact_schema_version",
    "graph_artifact_error_type",
    "graph_artifact_error",
}
EXCLUDED_PREFIXES = ("visual_", "graph_artifact_")


def target(df: pd.DataFrame) -> pd.Series:
    labels = pd.to_numeric(df["source_label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("source_label must contain only 0/1")
    # Modelling target: 1 = phishing, 0 = legitimate.
    return labels.eq(0).astype(int)


def to_numeric_series(series: pd.Series) -> pd.Series:
    values = series.replace({
        True: 1, False: 0,
        "True": 1, "False": 0,
        "TRUE": 1, "FALSE": 0,
        "true": 1, "false": 0,
        "yes": 1, "no": 0,
        "YES": 1, "NO": 0,
    })
    return pd.to_numeric(values, errors="coerce")


def candidate_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in EXCLUDED_COLUMNS:
            continue
        if any(c.startswith(p) for p in EXCLUDED_PREFIXES):
            continue
        if c.startswith(ALLOWED_PREFIXES):
            cols.append(c)
    return cols


def build_features(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    candidates = candidate_columns(train)
    if not candidates:
        raise ValueError("No candidate feature columns found")

    def convert(df: pd.DataFrame) -> pd.DataFrame:
        data = {}
        for c in candidates:
            if c in df.columns:
                data[c] = to_numeric_series(df[c])
            else:
                data[c] = pd.Series(np.nan, index=df.index)
        return pd.DataFrame(data, index=df.index)

    x_train = convert(train)
    x_val = convert(val)
    x_test = convert(test)

    kept, dropped = [], []
    for c in candidates:
        s = x_train[c]
        if s.notna().sum() == 0 or s.dropna().nunique() <= 1:
            dropped.append(c)
        else:
            kept.append(c)

    if not kept:
        raise ValueError("No usable numeric features remain")

    return x_train[kept], x_val[kept], x_test[kept], kept, dropped


def build_models(y_train: pd.Series) -> dict[str, Any]:
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos else 1.0

    scaled = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    tree = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    models: dict[str, Any] = {
        "Logistic Regression": Pipeline([
            ("prep", scaled),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED)),
        ]),
        "SVM": Pipeline([
            ("prep", scaled),
            ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)),
        ]),
        "Random Forest": Pipeline([
            ("prep", tree),
            ("model", RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1)),
        ]),
        "Extra Trees": Pipeline([
            ("prep", tree),
            ("model", ExtraTreesClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1)),
        ]),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline([
            ("prep", tree),
            ("model", XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=SEED,
                n_jobs=-1,
            )),
        ])

    if LGBMClassifier is not None:
        models["LightGBM"] = Pipeline([
            ("prep", tree),
            ("model", LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
                verbosity=-1,
            )),
        ])

    return models


def probabilities(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(x).astype(float)


def metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_phishing": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_phishing": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_phishing": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train website phishing ML baselines")
    parser.add_argument("--split-dir", default="data/processed/final_splits")
    parser.add_argument("--output-dir", default="results/ml_baselines")
    parser.add_argument("--model-dir", default="models/website_detection/ml_baselines")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    out_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(split_dir / "train.csv")
    val = pd.read_csv(split_dir / "validation.csv")
    test = pd.read_csv(split_dir / "test.csv")

    y_train, y_val, y_test = target(train), target(val), target(test)
    x_train, x_val, x_test, features, dropped = build_features(train, val, test)

    models = build_models(y_train)
    wanted = {"Logistic Regression", "SVM", "Random Forest", "Extra Trees", "XGBoost", "LightGBM"}
    missing = wanted - set(models)
    if missing:
        print("Missing optional libraries for:", ", ".join(sorted(missing)))
        print("Install with: pip install xgboost lightgbm")

    print("=" * 72)
    print("WEBSITE PHISHING ML BASELINES")
    print("=" * 72)
    print(f"Train: {len(train)} | Validation: {len(val)} | Test: {len(test)}")
    print(f"Usable tabular features: {len(features)}")
    print(f"Dropped unusable/constant features: {len(dropped)}")
    print("Positive class for metrics: PHISHING")
    print("=" * 72)

    rows = []
    val_pred_df = pd.DataFrame({"source_url": val["source_url"], "actual_phishing": y_val})
    test_pred_df = pd.DataFrame({"source_url": test["source_url"], "actual_phishing": y_test})

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        joblib.dump(model, model_dir / f"{slug(name)}.joblib")

        for split_name, x, y, pred_df in [
            ("validation", x_val, y_val, val_pred_df),
            ("test", x_test, y_test, test_pred_df),
        ]:
            pred = model.predict(x)
            prob = probabilities(model, x)
            m = metrics(y, pred, prob)
            rows.append({"model": name, "split": split_name, **m})
            pred_df[f"{slug(name)}_pred"] = pred
            pred_df[f"{slug(name)}_prob"] = prob

        print("  done")

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "model_metrics.csv", index=False)
    val_pred_df.to_csv(out_dir / "validation_predictions.csv", index=False)
    test_pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    with (out_dir / "feature_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({
            "positive_class": "phishing",
            "feature_count": len(features),
            "features": features,
            "dropped_columns": dropped,
        }, f, indent=2)

    ranking = (
        result[result["split"] == "validation"]
        .sort_values(["f1_phishing", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )
    ranking.to_csv(out_dir / "validation_ranking.csv", index=False)

    print()
    print("VALIDATION RANKING")
    print(ranking[[
        "model", "accuracy", "precision_phishing", "recall_phishing",
        "f1_phishing", "roc_auc", "pr_auc"
    ]].to_string(index=False))
    print()
    print("Best validation model:", ranking.iloc[0]["model"])
    print("Results saved to:", out_dir)
    print("Models saved to:", model_dir)
    print()
    print("Do not tune using TEST results. Use validation for model decisions.")


if __name__ == "__main__":
    main()
