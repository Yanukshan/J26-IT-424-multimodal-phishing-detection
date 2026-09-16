from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tldextract

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_SEED = 42

DOMAIN_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=()
)

ALLOWED_PREFIXES_ALL = (
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
# HELPERS
# ============================================================

def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def build_target(df: pd.DataFrame) -> pd.Series:
    labels = pd.to_numeric(
        df["source_label"],
        errors="coerce",
    )

    if labels.isna().any():
        raise ValueError(
            "Invalid source_label values."
        )

    return labels.eq(0).astype(int)


def normalize_source(series: pd.Series) -> pd.Series:
    values = (
        series.fillna("")
        .astype(str)
        .str.strip()
    )

    return values.mask(
        values.eq(""),
        "LEGACY",
    )


def registrable_domain_from_row(
    row: pd.Series,
) -> str:
    for column in [
        "final_domain",
        "requested_domain",
        "source_domain",
    ]:
        if column in row.index:
            value = str(
                row.get(
                    column,
                    "",
                )
                or ""
            ).strip().lower()

            if value:
                extracted = DOMAIN_EXTRACTOR(
                    value
                )

                if (
                    extracted.domain
                    and extracted.suffix
                ):
                    return (
                        f"{extracted.domain}."
                        f"{extracted.suffix}"
                    )

                return value

    return ""


def eligible_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    success = (
        df[
            "crawl_status"
        ]
        .astype(str)
        .str.upper()
        .eq(
            "SUCCESS"
        )
    )

    training = df[
        "training_candidate"
    ].map(
        as_bool
    )

    screenshot = df[
        "visual_screenshot_saved"
    ].map(
        as_bool
    )

    graph = df[
        "graph_artifact_saved"
    ].map(
        as_bool
    )

    output = df[
        success
        & training
        & screenshot
        & graph
    ].copy()

    output = (
        output
        .drop_duplicates(
            subset=[
                "source_url",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    output[
        "source_bucket"
    ] = normalize_source(
        output[
            "source_dataset"
        ]
        if "source_dataset"
        in output.columns
        else pd.Series(
            [
                ""
            ]
            * len(
                output
            )
        )
    )

    output[
        "registrable_domain_external"
    ] = output.apply(
        registrable_domain_from_row,
        axis=1,
    )

    return output


# ============================================================
# DOMAIN-AWARE LEGITIMATE HOLDOUT
# ============================================================

def choose_legitimate_holdout_domains(
    legitimate: pd.DataFrame,
    target_rows: int,
    seed: int,
) -> set[str]:
    """
    Pick whole legitimate registrable-domain groups until the
    holdout is approximately the same size as the legacy
    phishing holdout.
    """

    grouped = (
        legitimate.groupby(
            "registrable_domain_external"
        )
        .size()
        .reset_index(
            name="rows"
        )
    )

    grouped = grouped[
        grouped[
            "registrable_domain_external"
        ]
        .astype(str)
        .str.strip()
        .ne(
            ""
        )
    ].copy()

    rng = random.Random(
        seed
    )

    records = [
        (
            str(
                row[
                    "registrable_domain_external"
                ]
            ),
            int(
                row[
                    "rows"
                ]
            ),
            rng.random(),
        )
        for _, row
        in grouped.iterrows()
    ]

    records.sort(
        key=lambda item: item[
            2
        ]
    )

    selected: set[str] = set()
    count = 0

    for domain, rows, _ in records:
        if count >= target_rows:
            break

        selected.add(
            domain
        )
        count += rows

    return selected


# ============================================================
# DEVELOPMENT TRAIN / VALIDATION SPLIT
# ============================================================

def split_dev_by_domain(
    dev: pd.DataFrame,
    validation_ratio: float,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Split phishing and legitimate independently by whole
    registrable domains.
    """

    train_parts = []
    validation_parts = []

    for label in [
        0,
        1,
    ]:
        subset = dev[
            dev[
                "source_label"
            ]
            == label
        ].copy()

        domains = (
            subset[
                "registrable_domain_external"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        rng = random.Random(
            seed
            + label
        )

        rng.shuffle(
            domains
        )

        target_validation_rows = max(
            1,
            round(
                len(
                    subset
                )
                * validation_ratio
            ),
        )

        selected_validation_domains = set()
        selected_rows = 0

        domain_sizes = (
            subset.groupby(
                "registrable_domain_external"
            )
            .size()
            .to_dict()
        )

        for domain in domains:
            if selected_rows >= target_validation_rows:
                break

            selected_validation_domains.add(
                domain
            )

            selected_rows += int(
                domain_sizes.get(
                    domain,
                    0,
                )
            )

        validation_part = subset[
            subset[
                "registrable_domain_external"
            ].isin(
                selected_validation_domains
            )
        ].copy()

        train_part = subset[
            ~subset[
                "registrable_domain_external"
            ].isin(
                selected_validation_domains
            )
        ].copy()

        train_parts.append(
            train_part
        )

        validation_parts.append(
            validation_part
        )

    train = (
        pd.concat(
            train_parts,
            ignore_index=True,
        )
        .sample(
            frac=1,
            random_state=seed,
        )
        .reset_index(
            drop=True
        )
    )

    validation = (
        pd.concat(
            validation_parts,
            ignore_index=True,
        )
        .sample(
            frac=1,
            random_state=seed,
        )
        .reset_index(
            drop=True
        )
    )

    return (
        train,
        validation,
    )


# ============================================================
# FEATURE PREPARATION
# ============================================================

def coerce_numeric(
    series: pd.Series,
) -> pd.Series:
    replacements = {
        True: 1,
        False: 0,
        "True": 1,
        "False": 0,
        "TRUE": 1,
        "FALSE": 0,
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "YES": 1,
        "NO": 0,
    }

    return pd.to_numeric(
        series.replace(
            replacements
        ),
        errors="coerce",
    )


def feature_columns(
    train: pd.DataFrame,
    prefixes: tuple[str, ...],
) -> list[str]:
    columns = []

    for column in train.columns:
        if column in EXCLUDED_COLUMNS:
            continue

        if any(
            column.startswith(prefix)
            for prefix in EXCLUDED_PREFIXES
        ):
            continue

        if column.startswith(
            prefixes
        ):
            columns.append(
                column
            )

    return columns


def prepare_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    external_test: pd.DataFrame,
    prefixes: tuple[str, ...],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    candidates = feature_columns(
        train,
        prefixes,
    )

    train_numeric = pd.DataFrame(
        {
            column:
                coerce_numeric(
                    train[
                        column
                    ]
                )
            for column in candidates
        },
        index=train.index,
    )

    validation_numeric = pd.DataFrame(
        {
            column:
                coerce_numeric(
                    validation[
                        column
                    ]
                )
                if column
                in validation.columns
                else np.nan
            for column in candidates
        },
        index=validation.index,
    )

    external_numeric = pd.DataFrame(
        {
            column:
                coerce_numeric(
                    external_test[
                        column
                    ]
                )
                if column
                in external_test.columns
                else np.nan
            for column in candidates
        },
        index=external_test.index,
    )

    kept = []

    for column in candidates:
        values = (
            train_numeric[
                column
            ]
            .dropna()
        )

        if len(
            values
        ) == 0:
            continue

        if values.nunique() <= 1:
            continue

        kept.append(
            column
        )

    if not kept:
        raise ValueError(
            f"No usable features for prefixes {prefixes}"
        )

    return (
        train_numeric[
            kept
        ].copy(),

        validation_numeric[
            kept
        ].copy(),

        external_numeric[
            kept
        ].copy(),

        kept,
    )


# ============================================================
# MODELS
# ============================================================

def build_models(
    seed: int,
) -> dict[str, Pipeline]:
    return {
        "Logistic Regression":
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
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

        "Extra Trees":
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                        ),
                    ),
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


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    prediction = model.predict(
        X
    )

    probability = (
        model.predict_proba(
            X
        )[
            :,
            1
        ]
    )

    return {
        "accuracy":
            float(
                accuracy_score(
                    y,
                    prediction,
                )
            ),

        "precision_phishing":
            float(
                precision_score(
                    y,
                    prediction,
                    zero_division=0,
                )
            ),

        "recall_phishing":
            float(
                recall_score(
                    y,
                    prediction,
                    zero_division=0,
                )
            ),

        "f1_phishing":
            float(
                f1_score(
                    y,
                    prediction,
                    zero_division=0,
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    y,
                    probability,
                )
            ),

        "pr_auc":
            float(
                average_precision_score(
                    y,
                    probability,
                )
            ),
    }


# ============================================================
# AUDIT
# ============================================================

def run_audit(
    input_file: Path,
    output_directory: Path,
    seed: int,
) -> None:
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    raw = pd.read_csv(
        input_file
    )

    data = eligible_rows(
        raw
    )

    # --------------------------------------------------------
    # Identify source groups
    # --------------------------------------------------------

    phishtank = data[
        data[
            "source_bucket"
        ]
        .astype(str)
        .str.lower()
        .eq(
            "phishtank"
        )
    ].copy()

    legacy = data[
        ~data.index.isin(
            phishtank.index
        )
    ].copy()

    legacy_phishing = legacy[
        legacy[
            "source_label"
        ]
        == 0
    ].copy()

    legacy_legitimate = legacy[
        legacy[
            "source_label"
        ]
        == 1
    ].copy()

    phishtank_phishing = phishtank[
        phishtank[
            "source_label"
        ]
        == 0
    ].copy()

    if legacy_phishing.empty:
        raise ValueError(
            "No legacy/non-PhishTank phishing rows found."
        )

    if phishtank_phishing.empty:
        raise ValueError(
            "No PhishTank phishing rows found."
        )

    if legacy_legitimate.empty:
        raise ValueError(
            "No legacy legitimate rows found."
        )

    # --------------------------------------------------------
    # External cross-source test
    # --------------------------------------------------------
    #
    # Reserve ALL legacy/non-PhishTank phishing rows as the
    # external phishing test.
    #
    # IMPORTANT:
    # The same phishing registrable domain may appear in both
    # the legacy source and PhishTank. If those overlapping
    # PhishTank rows were kept in development, the external
    # test would no longer be domain-independent.
    #
    # Therefore:
    #   1. reserve legacy phishing domains for external test
    #   2. remove those domains from PhishTank development
    #   3. select legitimate external domains that also do not
    #      appear in the remaining development phishing pool
    # --------------------------------------------------------

    external_phishing_domains = set(
        legacy_phishing[
            "registrable_domain_external"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    phishtank_phishing_before_overlap_removal = len(
        phishtank_phishing
    )

    phishtank_phishing = phishtank_phishing[
        ~phishtank_phishing[
            "registrable_domain_external"
        ].isin(
            external_phishing_domains
        )
    ].copy()

    phishtank_overlap_rows_removed = (
        phishtank_phishing_before_overlap_removal
        - len(
            phishtank_phishing
        )
    )

    phishtank_dev_domains = set(
        phishtank_phishing[
            "registrable_domain_external"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    # Legitimate candidates for the external test must not use
    # domains already present in the phishing development pool.
    external_legitimate_candidates = legacy_legitimate[
        ~legacy_legitimate[
            "registrable_domain_external"
        ].isin(
            phishtank_dev_domains
        )
    ].copy()

    selected_legitimate_domains = (
        choose_legitimate_holdout_domains(
            legitimate=
                external_legitimate_candidates,

            target_rows=
                len(
                    legacy_phishing
                ),

            seed=
                seed,
        )
    )

    external_legitimate = legacy_legitimate[
        legacy_legitimate[
            "registrable_domain_external"
        ].isin(
            selected_legitimate_domains
        )
    ].copy()

    remaining_legitimate = legacy_legitimate[
        ~legacy_legitimate[
            "registrable_domain_external"
        ].isin(
            selected_legitimate_domains
        )
    ].copy()

    external_test = (
        pd.concat(
            [
                legacy_phishing,
                external_legitimate,
            ],
            ignore_index=True,
        )
        .sample(
            frac=1,
            random_state=seed,
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Final domain guard
    # --------------------------------------------------------
    #
    # Remove ANY development row whose registrable domain is
    # reserved by the external test. This is intentionally
    # conservative and guarantees the external test is truly
    # unseen at the domain level.
    # --------------------------------------------------------

    external_domains_reserved = set(
        external_test[
            "registrable_domain_external"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    phishtank_phishing = phishtank_phishing[
        ~phishtank_phishing[
            "registrable_domain_external"
        ].isin(
            external_domains_reserved
        )
    ].copy()

    remaining_legitimate = remaining_legitimate[
        ~remaining_legitimate[
            "registrable_domain_external"
        ].isin(
            external_domains_reserved
        )
    ].copy()

    # Development pool deliberately has:
    #   phishing -> PhishTank
    #   legitimate -> legacy legitimate
    #
    # External phishing is from a different source AND from
    # registrable domains never seen during development.
    dev = (
        pd.concat(
            [
                phishtank_phishing,
                remaining_legitimate,
            ],
            ignore_index=True,
        )
        .sample(
            frac=1,
            random_state=seed,
        )
        .reset_index(
            drop=True
        )
    )

    train, validation = split_dev_by_domain(
        dev=
            dev,

        validation_ratio=
            0.15,

        seed=
            seed,
    )

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    train_domains = set(
        train[
            "registrable_domain_external"
        ]
    )

    validation_domains = set(
        validation[
            "registrable_domain_external"
        ]
    )

    external_domains = set(
        external_test[
            "registrable_domain_external"
        ]
    )

    overlap_counts = {
        "train_validation":
            len(
                train_domains
                & validation_domains
            ),

        "train_external_test":
            len(
                train_domains
                & external_domains
            ),

        "validation_external_test":
            len(
                validation_domains
                & external_domains
            ),
    }

    if any(
        overlap_counts.values()
    ):
        raise RuntimeError(
            "Domain leakage detected in cross-source audit: "
            + json.dumps(
                overlap_counts
            )
        )

    # --------------------------------------------------------
    # Run URL-only and ALL-tabular experiments
    # --------------------------------------------------------

    feature_sets = {
        "URL only":
            (
                "url_",
            ),

        "All tabular evidence":
            ALLOWED_PREFIXES_ALL,
    }

    results = []

    for feature_set_name, prefixes in feature_sets.items():
        (
            X_train,
            X_validation,
            X_external,
            kept_features,
        ) = prepare_features(
            train=
                train,

            validation=
                validation,

            external_test=
                external_test,

            prefixes=
                prefixes,
        )

        y_train = build_target(
            train
        )

        y_validation = build_target(
            validation
        )

        y_external = build_target(
            external_test
        )

        for model_name, model in build_models(
            seed
        ).items():
            model.fit(
                X_train,
                y_train,
            )

            validation_metrics = (
                calculate_metrics(
                    model=
                        model,

                    X=
                        X_validation,

                    y=
                        y_validation,
                )
            )

            external_metrics = (
                calculate_metrics(
                    model=
                        model,

                    X=
                        X_external,

                    y=
                        y_external,
                )
            )

            results.append(
                {
                    "feature_set":
                        feature_set_name,

                    "model":
                        model_name,

                    "split":
                        "validation",

                    "feature_count":
                        len(
                            kept_features
                        ),

                    **validation_metrics,
                }
            )

            results.append(
                {
                    "feature_set":
                        feature_set_name,

                    "model":
                        model_name,

                    "split":
                        "external_cross_source_test",

                    "feature_count":
                        len(
                            kept_features
                        ),

                    **external_metrics,
                }
            )

    results_df = pd.DataFrame(
        results
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_directory
        / "cross_source_metrics.csv",
        index=False,
    )

    external_test[
        [
            "source_url",
            "class_name",
            "source_bucket",
            "registrable_domain_external",
        ]
    ].to_csv(
        output_directory
        / "external_cross_source_test_manifest.csv",
        index=False,
    )

    summary = {
        "eligible_total":
            len(
                data
            ),

        "phishtank_phishing":
            len(
                phishtank_phishing
            ),

        "phishtank_rows_removed_for_external_domain_overlap":
            phishtank_overlap_rows_removed,

        "legacy_phishing_external_test":
            len(
                legacy_phishing
            ),

        "legacy_legitimate_total":
            len(
                legacy_legitimate
            ),

        "external_legitimate":
            len(
                external_legitimate
            ),

        "development_rows":
            len(
                dev
            ),

        "train_rows":
            len(
                train
            ),

        "validation_rows":
            len(
                validation
            ),

        "external_test_rows":
            len(
                external_test
            ),

        "domain_overlap_counts":
            overlap_counts,
    }

    with (
        output_directory
        / "cross_source_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "=" * 88
    )
    print(
        "CROSS-SOURCE GENERALIZATION AUDIT"
    )
    print(
        "=" * 88
    )

    print(
        f"Eligible total: {len(data)}"
    )

    print(
        f"PhishTank phishing used for development: "
        f"{len(phishtank_phishing)}"
    )

    print(
        "PhishTank rows removed because their domains also "
        f"occur in the external legacy phishing set: "
        f"{phishtank_overlap_rows_removed}"
    )

    print(
        f"Legacy phishing reserved for EXTERNAL TEST: "
        f"{len(legacy_phishing)}"
    )

    print(
        f"Legacy legitimate available: "
        f"{len(legacy_legitimate)}"
    )

    print(
        f"External legitimate selected: "
        f"{len(external_legitimate)}"
    )

    print()
    print(
        f"Train rows: {len(train)}"
    )

    print(
        f"Validation rows: {len(validation)}"
    )

    print(
        f"External cross-source test rows: "
        f"{len(external_test)}"
    )

    print()
    print(
        "DOMAIN OVERLAP"
    )

    for key, value in overlap_counts.items():
        print(
            f"  {key}: {value}"
        )

    print()
    print(
        "RESULTS"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "- Validation measures in-source development behavior."
    )

    print(
        "- External test measures whether a detector trained "
        "mainly on PhishTank phishing generalizes to phishing "
        "from the legacy/non-PhishTank source."
    )

    print(
        "- A large validation -> external-test drop indicates "
        "source/distribution dependence."
    )

    print(
        f"Outputs: {output_directory}"
    )

    print(
        "=" * 88
    )


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-source generalization audit for the finalized "
            "website phishing dataset."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "data/processed/"
            "final_multimodal_collection_all.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/cross_source_audit"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=
            DEFAULT_SEED,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    run_audit(
        input_file=
            Path(
                args.input
            ),

        output_directory=
            Path(
                args.output_dir
            ),

        seed=
            args.seed,
    )


if __name__ == "__main__":
    main()
