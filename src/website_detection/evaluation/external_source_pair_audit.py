from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import tldextract

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
DOMAIN_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

ALLOWED_PREFIXES = (
    "url_",
    "dom_",
    "cred_",
    "net_",
    "beh_",
    "graph_",
)

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

EXCLUDED_PREFIXES = (
    "visual_",
    "graph_artifact_",
)


# ============================================================
# DATA HELPERS
# ============================================================

def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def filter_eligible(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "crawl_status",
        "training_candidate",
        "visual_screenshot_saved",
        "graph_artifact_saved",
        "source_label",
        "source_url",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    mask = (
        df["crawl_status"].astype(str).str.upper().eq("SUCCESS")
        & df["training_candidate"].map(as_bool)
        & df["visual_screenshot_saved"].map(as_bool)
        & df["graph_artifact_saved"].map(as_bool)
    )

    return (
        df.loc[mask]
        .drop_duplicates(subset=["source_url"], keep="first")
        .reset_index(drop=True)
    )


def hostname_from_row(row: pd.Series) -> str:
    for col in ["final_domain", "requested_domain", "source_domain"]:
        if col in row.index:
            value = str(row.get(col, "") or "").strip().lower().rstrip(".")
            if value and value != "nan":
                return value

    for col in ["final_url", "requested_url", "source_url", "URL"]:
        if col not in row.index:
            continue

        value = str(row.get(col, "") or "").strip()
        if not value or value == "nan":
            continue

        try:
            hostname = (urlparse(value).hostname or "").lower().rstrip(".")
            if hostname:
                return hostname
        except Exception:
            pass

    return ""


def registrable_domain(hostname: str) -> str:
    hostname = str(hostname or "").strip().lower().rstrip(".")
    if not hostname:
        return ""

    extracted = DOMAIN_EXTRACTOR(hostname)

    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"

    return hostname


def add_domain_column(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    output["audit_hostname"] = output.apply(
        hostname_from_row,
        axis=1,
    )

    output["audit_registrable_domain"] = output["audit_hostname"].map(
        registrable_domain
    )

    return output


def build_target(df: pd.DataFrame) -> pd.Series:
    """
    Collector labels:
        0 = phishing
        1 = legitimate

    Modelling target:
        1 = phishing
        0 = legitimate
    """
    labels = pd.to_numeric(
        df["source_label"],
        errors="coerce",
    )

    if labels.isna().any():
        raise ValueError("source_label contains invalid values.")

    if (~labels.isin([0, 1])).any():
        raise ValueError("source_label must contain only 0 and 1.")

    return labels.eq(0).astype(int).rename("is_phishing")


# ============================================================
# FEATURE PREPARATION
# ============================================================

def coerce_numeric(series: pd.Series) -> pd.Series:
    replacements = {
        True: 1,
        False: 0,
        "True": 1,
        "False": 0,
        "TRUE": 1,
        "FALSE": 0,
        "true": 1,
        "false": 0,
        "Yes": 1,
        "No": 0,
        "yes": 1,
        "no": 0,
        "YES": 1,
        "NO": 0,
    }

    return pd.to_numeric(
        series.replace(replacements),
        errors="coerce",
    )


def candidate_features(train: pd.DataFrame) -> list[str]:
    columns = []

    for column in train.columns:
        if column in EXCLUDED_COLUMNS:
            continue

        if any(
            column.startswith(prefix)
            for prefix in EXCLUDED_PREFIXES
        ):
            continue

        if column.startswith(ALLOWED_PREFIXES):
            columns.append(column)

    return columns


def prepare_feature_frames(
    train_df: pd.DataFrame,
    full_external_df: pd.DataFrame,
    balanced_external_df: pd.DataFrame,
):
    candidates = candidate_features(train_df)

    if not candidates:
        raise ValueError("No candidate tabular feature columns found.")

    train_numeric = pd.DataFrame(
        {
            c: coerce_numeric(train_df[c])
            for c in candidates
        },
        index=train_df.index,
    )

    full_numeric = pd.DataFrame(
        {
            c: (
                coerce_numeric(full_external_df[c])
                if c in full_external_df.columns
                else np.nan
            )
            for c in candidates
        },
        index=full_external_df.index,
    )

    balanced_numeric = pd.DataFrame(
        {
            c: (
                coerce_numeric(balanced_external_df[c])
                if c in balanced_external_df.columns
                else np.nan
            )
            for c in candidates
        },
        index=balanced_external_df.index,
    )

    kept = []
    dropped = []

    for column in candidates:
        values = train_numeric[column].dropna()

        if len(values) == 0:
            dropped.append(column)
            continue

        if values.nunique() <= 1:
            dropped.append(column)
            continue

        kept.append(column)

    if not kept:
        raise ValueError("No usable non-constant numerical features remain.")

    return (
        train_numeric[kept].copy(),
        full_numeric[kept].copy(),
        balanced_numeric[kept].copy(),
        kept,
        dropped,
    )


# ============================================================
# MODELS
# ============================================================

def build_models(y_train: pd.Series, seed: int):
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = (
        negative_count / positive_count
        if positive_count > 0
        else 1.0
    )

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),

        "SVM": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        probability=False,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "Extra Trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.90,
                        colsample_bytree=0.90,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if LGBMClassifier is not None:
        models["LightGBM"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=400,
                        learning_rate=0.05,
                        num_leaves=31,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )

    return models


# ============================================================
# METRICS
# ============================================================

def score_vector(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))

    return model.predict(X).astype(float)


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    prediction = model.predict(X)
    score = score_vector(model, X)

    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, prediction)
        ),
        "precision_phishing": float(
            precision_score(y, prediction, zero_division=0)
        ),
        "recall_phishing": float(
            recall_score(y, prediction, zero_division=0)
        ),
        "f1_phishing": float(
            f1_score(y, prediction, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
    }


# ============================================================
# MAIN
# ============================================================

def run_audit(
    v1_file: Path,
    tranco_file: Path,
    openphish_file: Path,
    output_dir: Path,
    seed: int,
):
    for file_path in [v1_file, tranco_file, openphish_file]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {file_path}"
            )

    v1 = add_domain_column(
        filter_eligible(pd.read_csv(v1_file))
    )

    tranco = add_domain_column(
        filter_eligible(pd.read_csv(tranco_file))
    )

    openphish = add_domain_column(
        filter_eligible(pd.read_csv(openphish_file))
    )

    if not pd.to_numeric(
        tranco["source_label"],
        errors="coerce",
    ).eq(1).all():
        raise ValueError(
            "Tranco eligible set contains non-legitimate labels."
        )

    if not pd.to_numeric(
        openphish["source_label"],
        errors="coerce",
    ).eq(0).all():
        raise ValueError(
            "OpenPhish eligible set contains non-phishing labels."
        )

    v1_domains = set(
        v1["audit_registrable_domain"]
        .dropna()
        .astype(str)
    )

    tranco_before = len(tranco)
    openphish_before = len(openphish)

    # Remove any external-domain overlap with Dataset V1.
    tranco = tranco[
        ~tranco["audit_registrable_domain"].isin(v1_domains)
    ].copy()

    openphish = openphish[
        ~openphish["audit_registrable_domain"].isin(v1_domains)
    ].copy()

    tranco_overlap_removed = tranco_before - len(tranco)
    openphish_overlap_removed = openphish_before - len(openphish)

    # Remove cross-class domain conflicts between the NEW sources.
    conflict_domains = (
        set(tranco["audit_registrable_domain"].dropna().astype(str))
        & set(openphish["audit_registrable_domain"].dropna().astype(str))
    )

    if conflict_domains:
        tranco = tranco[
            ~tranco["audit_registrable_domain"].isin(conflict_domains)
        ].copy()

        openphish = openphish[
            ~openphish["audit_registrable_domain"].isin(conflict_domains)
        ].copy()

    if tranco.empty or openphish.empty:
        raise ValueError(
            "An external class became empty after domain filtering."
        )

    external_full = (
        pd.concat([tranco, openphish], ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    # Balanced external test for easiest model-to-model comparison.
    n = min(len(tranco), len(openphish))

    external_balanced = (
        pd.concat(
            [
                tranco.sample(
                    n=n,
                    random_state=seed,
                    replace=False,
                ),
                openphish.sample(
                    n=n,
                    random_state=seed,
                    replace=False,
                ),
            ],
            ignore_index=True,
        )
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    external_domains = set(
        external_full["audit_registrable_domain"]
        .dropna()
        .astype(str)
    )

    remaining_overlap = v1_domains & external_domains

    if remaining_overlap:
        raise RuntimeError(
            "Domain leakage remains between Dataset V1 and "
            f"the new external test: {len(remaining_overlap)} domains."
        )

    y_train = build_target(v1)
    y_full = build_target(external_full)
    y_balanced = build_target(external_balanced)

    (
        X_train,
        X_full,
        X_balanced,
        kept_features,
        dropped_features,
    ) = prepare_feature_frames(
        train_df=v1,
        full_external_df=external_full,
        balanced_external_df=external_balanced,
    )

    models = build_models(
        y_train=y_train,
        seed=seed,
    )

    rows = []

    print()
    print("=" * 92)
    print("NEW-SOURCE EXTERNAL GENERALIZATION AUDIT")
    print("=" * 92)
    print(f"Dataset V1 training rows: {len(v1)}")
    print(f"Tranco eligible before overlap removal: {tranco_before}")
    print(f"OpenPhish eligible before overlap removal: {openphish_before}")
    print(f"Tranco rows removed for V1 domain overlap: {tranco_overlap_removed}")
    print(f"OpenPhish rows removed for V1 domain overlap: {openphish_overlap_removed}")
    print(f"New cross-class domains removed: {len(conflict_domains)}")
    print(f"V1 / external domain overlap: {len(remaining_overlap)}")
    print(f"Tranco external rows: {len(tranco)}")
    print(f"OpenPhish external rows: {len(openphish)}")
    print(f"Full external rows: {len(external_full)}")
    print(f"Balanced external rows: {len(external_balanced)}")
    print(f"Usable tabular features: {len(kept_features)}")
    print(f"Dropped unusable/constant features: {len(dropped_features)}")
    print("=" * 92)

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        full_metrics = evaluate(
            model,
            X_full,
            y_full,
        )

        balanced_metrics = evaluate(
            model,
            X_balanced,
            y_balanced,
        )

        rows.append(
            {
                "model": name,
                "split": "external_full",
                "rows": len(external_full),
                **full_metrics,
            }
        )

        rows.append(
            {
                "model": name,
                "split": "external_balanced",
                "rows": len(external_balanced),
                **balanced_metrics,
            }
        )

    results = pd.DataFrame(rows)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_dir / "external_source_pair_metrics.csv",
        index=False,
    )

    summary = {
        "v1_eligible_rows": len(v1),
        "tranco_eligible_before_overlap_removal": tranco_before,
        "openphish_eligible_before_overlap_removal": openphish_before,
        "tranco_rows_removed_for_v1_domain_overlap": tranco_overlap_removed,
        "openphish_rows_removed_for_v1_domain_overlap": openphish_overlap_removed,
        "new_cross_class_domains_removed": len(conflict_domains),
        "tranco_external_rows": len(tranco),
        "openphish_external_rows": len(openphish),
        "external_full_rows": len(external_full),
        "external_balanced_rows": len(external_balanced),
        "v1_external_domain_overlap": len(remaining_overlap),
        "feature_count": len(kept_features),
        "dropped_feature_count": len(dropped_features),
    }

    with (
        output_dir / "external_source_pair_summary.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("RESULTS")
    print(results.to_string(index=False))
    print()
    print(
        "Use external_balanced for the clearest comparison; "
        "external_full is also retained to report all new usable sites."
    )
    print(f"Outputs: {output_dir}")
    print("=" * 92)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Dataset V1 models on completely new "
            "Tranco legitimate and OpenPhish phishing sources."
        )
    )

    parser.add_argument(
        "--v1",
        default="data/processed/final_multimodal_collection_all.csv",
    )

    parser.add_argument(
        "--tranco",
        default="data/processed/final_tranco_collection.csv",
    )

    parser.add_argument(
        "--openphish",
        default="data/processed/final_openphish_collection.csv",
    )

    parser.add_argument(
        "--output-dir",
        default="results/external_source_pair_audit",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_audit(
        v1_file=Path(args.v1),
        tranco_file=Path(args.tranco),
        openphish_file=Path(args.openphish),
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )
