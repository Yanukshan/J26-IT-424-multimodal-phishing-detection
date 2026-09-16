from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_SEED = 42


# ============================================================
# TARGET
# ============================================================

def build_target(df: pd.DataFrame) -> pd.Series:
    labels = pd.to_numeric(
        df["source_label"],
        errors="coerce",
    )

    if labels.isna().any():
        raise ValueError(
            "source_label contains invalid values."
        )

    return labels.eq(0).astype(int)


# ============================================================
# NUMERIC CONVERSION
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


def prepare_url_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    candidates = [
        column
        for column
        in train.columns
        if column.startswith(
            "url_"
        )
    ]

    train_numeric = pd.DataFrame(
        {
            column:
                coerce_numeric(
                    train[column]
                )
            for column in candidates
        },
        index=train.index,
    )

    validation_numeric = pd.DataFrame(
        {
            column:
                coerce_numeric(
                    validation[column]
                )
                if column
                in validation.columns
                else np.nan
            for column in candidates
        },
        index=validation.index,
    )

    test_numeric = pd.DataFrame(
        {
            column:
                coerce_numeric(
                    test[column]
                )
                if column
                in test.columns
                else np.nan
            for column in candidates
        },
        index=test.index,
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
            "No usable URL features found."
        )

    return (
        train_numeric[
            kept
        ].copy(),

        validation_numeric[
            kept
        ].copy(),

        test_numeric[
            kept
        ].copy(),

        kept,
    )


# ============================================================
# MODELS
# ============================================================

def logistic_model(
    seed: int,
) -> Pipeline:
    return Pipeline(
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
    )


def extra_trees_model(
    seed: int,
) -> Pipeline:
    return Pipeline(
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
    )


# ============================================================
# METRICS
# ============================================================

def metrics(
    y_true: pd.Series,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    prediction,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    prediction,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    prediction,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    prediction,
                    zero_division=0,
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    y_true,
                    probability,
                )
            ),
    }


def evaluate_model(
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

    return metrics(
        y_true=
            y,

        prediction=
            prediction,

        probability=
            probability,
    )


# ============================================================
# SOURCE / CLASS ASSOCIATION
# ============================================================

def normalize_source(
    series: pd.Series,
) -> pd.Series:
    values = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return values.mask(
        values.eq(""),
        "UNKNOWN_OR_LEGACY",
    )


def source_class_audit(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    float | None,
]:
    if "source_dataset" not in dataframe.columns:
        return (
            pd.DataFrame(),
            None,
        )

    audit = dataframe[
        [
            "source_dataset",
            "source_label",
        ]
    ].copy()

    audit[
        "source_dataset"
    ] = normalize_source(
        audit[
            "source_dataset"
        ]
    )

    audit[
        "class_name_audit"
    ] = np.where(
        pd.to_numeric(
            audit[
                "source_label"
            ],
            errors="coerce",
        ).eq(
            0
        ),
        "phishing",
        "legitimate",
    )

    table = (
        audit.groupby(
            [
                "source_dataset",
                "class_name_audit",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    class_columns = [
        column
        for column
        in [
            "phishing",
            "legitimate",
        ]
        if column
        in table.columns
    ]

    for column in [
        "phishing",
        "legitimate",
    ]:
        if column not in table.columns:
            table[
                column
            ] = 0

    table[
        "total"
    ] = (
        table[
            "phishing"
        ]
        + table[
            "legitimate"
        ]
    )

    table[
        "majority_class_share"
    ] = (
        table[
            [
                "phishing",
                "legitimate",
            ]
        ]
        .max(
            axis=1
        )
        / table[
            "total"
        ]
    )

    weighted_purity = float(
        (
            table[
                "majority_class_share"
            ]
            * table[
                "total"
            ]
        ).sum()
        / table[
            "total"
        ].sum()
    )

    return (
        table.sort_values(
            "total",
            ascending=False,
        ),
        weighted_purity,
    )


# ============================================================
# SINGLE-FEATURE AUDIT
# ============================================================

def single_feature_audit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    seed: int,
) -> pd.DataFrame:
    rows = []

    for column in X_train.columns:
        model = logistic_model(
            seed
        )

        model.fit(
            X_train[
                [
                    column
                ]
            ],
            y_train,
        )

        result = evaluate_model(
            model,
            X_validation[
                [
                    column
                ]
            ],
            y_validation,
        )

        rows.append(
            {
                "feature":
                    column,

                **result,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            by=[
                "f1",
                "roc_auc",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# DISTRIBUTION EFFECT SIZE
# ============================================================

def standardized_mean_difference(
    legitimate: pd.Series,
    phishing: pd.Series,
) -> float:
    legitimate = legitimate.dropna()
    phishing = phishing.dropna()

    if (
        len(
            legitimate
        )
        < 2
        or len(
            phishing
        )
        < 2
    ):
        return float(
            "nan"
        )

    var_legit = float(
        legitimate.var(
            ddof=1
        )
    )

    var_phish = float(
        phishing.var(
            ddof=1
        )
    )

    pooled = (
        (
            (
                len(
                    legitimate
                )
                - 1
            )
            * var_legit
        )
        +
        (
            (
                len(
                    phishing
                )
                - 1
            )
            * var_phish
        )
    ) / (
        len(
            legitimate
        )
        + len(
            phishing
        )
        - 2
    )

    if pooled <= 0:
        return 0.0

    return float(
        (
            phishing.mean()
            - legitimate.mean()
        )
        / np.sqrt(
            pooled
        )
    )


def distribution_audit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    rows = []

    for column in X_train.columns:
        legitimate = X_train.loc[
            y_train.eq(0),
            column,
        ]

        phishing = X_train.loc[
            y_train.eq(1),
            column,
        ]

        rows.append(
            {
                "feature":
                    column,

                "legitimate_mean":
                    float(
                        legitimate.mean()
                    ),

                "phishing_mean":
                    float(
                        phishing.mean()
                    ),

                "legitimate_median":
                    float(
                        legitimate.median()
                    ),

                "phishing_median":
                    float(
                        phishing.median()
                    ),

                "standardized_mean_difference":
                    standardized_mean_difference(
                        legitimate,
                        phishing,
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "absolute_smd"
    ] = (
        result[
            "standardized_mean_difference"
        ].abs()
    )

    return result.sort_values(
        "absolute_smd",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

def permutation_audit(
    model,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    seed: int,
) -> pd.DataFrame:
    result = permutation_importance(
        estimator=
            model,

        X=
            X_validation,

        y=
            y_validation,

        scoring=
            "f1",

        n_repeats=
            30,

        random_state=
            seed,

        n_jobs=
            -1,
    )

    return (
        pd.DataFrame(
            {
                "feature":
                    X_validation.columns,

                "importance_mean":
                    result[
                        "importances_mean"
                    ],

                "importance_std":
                    result[
                        "importances_std"
                    ],
            }
        )
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# TOP-K URL FEATURE EXPERIMENT
# ============================================================

def top_k_audit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    ranked_features: list[str],
    seed: int,
) -> pd.DataFrame:
    rows = []

    requested_k = [
        1,
        2,
        3,
        5,
        10,
        20,
    ]

    for k in requested_k:
        available = min(
            k,
            len(
                ranked_features
            ),
        )

        selected = ranked_features[
            :available
        ]

        model = logistic_model(
            seed
        )

        model.fit(
            X_train[
                selected
            ],
            y_train,
        )

        result = evaluate_model(
            model,
            X_validation[
                selected
            ],
            y_validation,
        )

        rows.append(
            {
                "top_k":
                    available,

                "features":
                    "|".join(
                        selected
                    ),

                **result,
            }
        )

    return pd.DataFrame(
        rows
    ).drop_duplicates(
        subset=[
            "top_k",
        ]
    )


# ============================================================
# SOURCE-STRATIFIED PREDICTION AUDIT
# ============================================================

def source_stratified_predictions(
    dataframe: pd.DataFrame,
    y_true: pd.Series,
    prediction: np.ndarray,
) -> pd.DataFrame:
    if "source_dataset" not in dataframe.columns:
        return pd.DataFrame()

    sources = normalize_source(
        dataframe[
            "source_dataset"
        ]
    )

    frame = pd.DataFrame(
        {
            "source_dataset":
                sources,

            "actual":
                y_true.to_numpy(),

            "predicted":
                prediction,
        }
    )

    rows = []

    for source_name, group in frame.groupby(
        "source_dataset"
    ):
        phishing_rows = group[
            group[
                "actual"
            ].eq(
                1
            )
        ]

        legitimate_rows = group[
            group[
                "actual"
            ].eq(
                0
            )
        ]

        rows.append(
            {
                "source_dataset":
                    source_name,

                "rows":
                    len(
                        group
                    ),

                "phishing_rows":
                    len(
                        phishing_rows
                    ),

                "legitimate_rows":
                    len(
                        legitimate_rows
                    ),

                "overall_accuracy":
                    float(
                        (
                            group[
                                "actual"
                            ]
                            == group[
                                "predicted"
                            ]
                        ).mean()
                    ),

                "phishing_recall":
                    (
                        float(
                            (
                                phishing_rows[
                                    "predicted"
                                ]
                                == 1
                            ).mean()
                        )
                        if len(
                            phishing_rows
                        )
                        > 0
                        else np.nan
                    ),

                "legitimate_specificity":
                    (
                        float(
                            (
                                legitimate_rows[
                                    "predicted"
                                ]
                                == 0
                            ).mean()
                        )
                        if len(
                            legitimate_rows
                        )
                        > 0
                        else np.nan
                    ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "rows",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# MAIN AUDIT
# ============================================================

def run_audit(
    split_directory: Path,
    output_directory: Path,
    seed: int,
) -> None:
    train = pd.read_csv(
        split_directory
        / "train.csv"
    )

    validation = pd.read_csv(
        split_directory
        / "validation.csv"
    )

    test = pd.read_csv(
        split_directory
        / "test.csv"
    )

    y_train = build_target(
        train
    )

    y_validation = build_target(
        validation
    )

    y_test = build_target(
        test
    )

    (
        X_train,
        X_validation,
        X_test,
        features,
    ) = prepare_url_features(
        train=
            train,

        validation=
            validation,

        test=
            test,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Re-check URL-only performance
    # --------------------------------------------------------

    logistic = logistic_model(
        seed
    )

    logistic.fit(
        X_train,
        y_train,
    )

    extra_trees = extra_trees_model(
        seed
    )

    extra_trees.fit(
        X_train,
        y_train,
    )

    performance_rows = []

    for name, model in [
        (
            "Logistic Regression",
            logistic,
        ),
        (
            "Extra Trees",
            extra_trees,
        ),
    ]:
        for split_name, X, y in [
            (
                "validation",
                X_validation,
                y_validation,
            ),
            (
                "test",
                X_test,
                y_test,
            ),
        ]:
            result = evaluate_model(
                model,
                X,
                y,
            )

            performance_rows.append(
                {
                    "model":
                        name,

                    "split":
                        split_name,

                    **result,
                }
            )

    performance = pd.DataFrame(
        performance_rows
    )

    performance.to_csv(
        output_directory
        / "url_only_recheck.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Provenance / source association
    # --------------------------------------------------------

    combined = pd.concat(
        [
            train,
            validation,
            test,
        ],
        ignore_index=True,
    )

    source_table, source_purity = (
        source_class_audit(
            combined
        )
    )

    source_table.to_csv(
        output_directory
        / "source_class_association.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Single feature predictive power
    # --------------------------------------------------------

    single_feature = (
        single_feature_audit(
            X_train=
                X_train,

            y_train=
                y_train,

            X_validation=
                X_validation,

            y_validation=
                y_validation,

            seed=
                seed,
        )
    )

    single_feature.to_csv(
        output_directory
        / "single_url_feature_validation.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Class-wise distributions
    # --------------------------------------------------------

    distributions = (
        distribution_audit(
            X_train=
                X_train,

            y_train=
                y_train,
        )
    )

    distributions.to_csv(
        output_directory
        / "url_feature_class_distributions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Permutation importance
    # --------------------------------------------------------

    permutation = (
        permutation_audit(
            model=
                logistic,

            X_validation=
                X_validation,

            y_validation=
                y_validation,

            seed=
                seed,
        )
    )

    permutation.to_csv(
        output_directory
        / "logistic_url_permutation_importance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Top-k based on TRAIN Extra Trees importance
    # --------------------------------------------------------

    transformed_train = (
        extra_trees.named_steps[
            "imputer"
        ].transform(
            X_train
        )
    )

    trained_tree = (
        extra_trees.named_steps[
            "model"
        ]
    )

    importance_order = np.argsort(
        trained_tree.feature_importances_
    )[
        ::-1
    ]

    ranked_features = [
        features[
            index
        ]
        for index
        in importance_order
    ]

    top_k = top_k_audit(
        X_train=
            X_train,

        y_train=
            y_train,

        X_validation=
            X_validation,

        y_validation=
            y_validation,

        ranked_features=
            ranked_features,

        seed=
            seed,
    )

    top_k.to_csv(
        output_directory
        / "top_k_url_features_validation.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Source-stratified URL-only test predictions
    # --------------------------------------------------------

    test_prediction = logistic.predict(
        X_test
    )

    source_test = (
        source_stratified_predictions(
            dataframe=
                test,

            y_true=
                y_test,

            prediction=
                test_prediction,
        )
    )

    source_test.to_csv(
        output_directory
        / "test_performance_by_source.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Audit flags
    # --------------------------------------------------------

    best_single_f1 = float(
        single_feature.iloc[
            0
        ][
            "f1"
        ]
    )

    flags = {
        "url_only_validation_f1_logistic":
            float(
                performance.loc[
                    (
                        performance[
                            "model"
                        ]
                        == "Logistic Regression"
                    )
                    &
                    (
                        performance[
                            "split"
                        ]
                        == "validation"
                    ),
                    "f1",
                ].iloc[
                    0
                ]
            ),

        "best_single_url_feature":
            str(
                single_feature.iloc[
                    0
                ][
                    "feature"
                ]
            ),

        "best_single_url_feature_validation_f1":
            best_single_f1,

        "source_weighted_class_purity":
            (
                source_purity
                if source_purity
                is not None
                else None
            ),

        "single_feature_shortcut_warning":
            bool(
                best_single_f1
                >= 0.95
            ),

        "source_confounding_warning":
            bool(
                source_purity
                is not None
                and source_purity
                >= 0.90
            ),
    }

    with (
        output_directory
        / "audit_flags.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            flags,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print(
        "=" * 82
    )
    print(
        "URL / DATA-SOURCE BIAS AUDIT"
    )
    print(
        "=" * 82
    )

    print()
    print(
        "URL-ONLY RECHECK"
    )
    print(
        performance.to_string(
            index=False
        )
    )

    print()
    print(
        "TOP 10 SINGLE URL FEATURES"
    )
    print(
        single_feature.head(
            10
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "TOP 10 TRAIN CLASS-SEPARATION FEATURES"
    )
    print(
        distributions[
            [
                "feature",
                "legitimate_mean",
                "phishing_mean",
                "standardized_mean_difference",
                "absolute_smd",
            ]
        ]
        .head(
            10
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "TOP 10 LOGISTIC PERMUTATION IMPORTANCE"
    )
    print(
        permutation.head(
            10
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "TOP-K URL FEATURE EXPERIMENT"
    )
    print(
        top_k.to_string(
            index=False
        )
    )

    if not source_table.empty:
        print()
        print(
            "SOURCE / CLASS ASSOCIATION"
        )
        print(
            source_table.to_string(
                index=False
            )
        )

        print()
        print(
            "Weighted source/class purity: "
            f"{source_purity:.4f}"
        )

    if not source_test.empty:
        print()
        print(
            "TEST PERFORMANCE BY SOURCE"
        )
        print(
            source_test.to_string(
                index=False
            )
        )

    print()
    print(
        "AUDIT FLAGS"
    )
    print(
        json.dumps(
            flags,
            indent=2,
        )
    )

    print()
    print(
        f"Saved to: {output_directory}"
    )
    print(
        "=" * 82
    )


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit URL-only performance and possible "
            "dataset/source shortcuts before deep learning."
        )
    )

    parser.add_argument(
        "--split-dir",
        default=(
            "data/processed/final_splits"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/source_bias_audit"
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
        split_directory=
            Path(
                args.split_dir
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
