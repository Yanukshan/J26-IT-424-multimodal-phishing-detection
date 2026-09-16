from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SEED = 42

# Artifact / string metadata must never become model inputs.
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
    "visual_",          # screenshot metadata, not image pixels
    "graph_artifact_",  # serialized graph metadata
)


# ============================================================
# EVIDENCE GROUPS
# ============================================================

FEATURE_GROUPS = {
    "URL only": (
        "url_",
    ),

    "DOM only": (
        "dom_",
    ),

    "Credential only": (
        "cred_",
    ),

    "Network only": (
        "net_",
    ),

    "Behavior only": (
        "beh_",
    ),

    "Graph summary only": (
        "graph_",
    ),

    "Static content": (
        "url_",
        "dom_",
        "cred_",
    ),

    "Runtime evidence": (
        "net_",
        "beh_",
    ),

    "Structural evidence": (
        "dom_",
        "graph_",
    ),

    "All tabular evidence": (
        "url_",
        "dom_",
        "cred_",
        "net_",
        "beh_",
        "graph_",
    ),
}


# ============================================================
# LABEL HANDLING
# ============================================================

def build_target(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Collector labels:
        0 = phishing
        1 = legitimate

    Modelling target:
        1 = phishing
        0 = legitimate
    """

    if "source_label" not in dataframe.columns:
        raise ValueError(
            "Missing required column: source_label"
        )

    labels = pd.to_numeric(
        dataframe["source_label"],
        errors="coerce",
    )

    if labels.isna().any():
        raise ValueError(
            "source_label contains invalid/missing values."
        )

    if (
        ~labels.isin(
            [0, 1]
        )
    ).any():
        raise ValueError(
            "source_label must contain only 0 and 1."
        )

    return (
        labels.eq(0)
        .astype(int)
        .rename("is_phishing")
    )


# ============================================================
# FEATURE PREPARATION
# ============================================================

def coerce_feature_series(
    series: pd.Series,
) -> pd.Series:

    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.astype(
            float
        )

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

    values = series.replace(
        replacements
    )

    return pd.to_numeric(
        values,
        errors="coerce",
    )


def select_group_columns(
    dataframe: pd.DataFrame,
    prefixes: tuple[str, ...],
) -> list[str]:
    """
    Select only tabular numerical evidence columns belonging to
    one evidence group.
    """

    columns: list[str] = []

    for column in dataframe.columns:

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


def prepare_group_features(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    prefixes: tuple[str, ...],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[str],
]:
    """
    Feature keep/drop decisions are made from TRAIN only.
    """

    candidates = select_group_columns(
        train_df,
        prefixes,
    )

    if not candidates:
        raise ValueError(
            f"No feature columns found for prefixes: {prefixes}"
        )

    train_numeric = pd.DataFrame(
        {
            column:
                coerce_feature_series(
                    train_df[column]
                )
            for column in candidates
        },
        index=train_df.index,
    )

    validation_numeric = pd.DataFrame(
        {
            column:
                coerce_feature_series(
                    validation_df[column]
                )
                if column in validation_df.columns
                else np.nan
            for column in candidates
        },
        index=validation_df.index,
    )

    test_numeric = pd.DataFrame(
        {
            column:
                coerce_feature_series(
                    test_df[column]
                )
                if column in test_df.columns
                else np.nan
            for column in candidates
        },
        index=test_df.index,
    )

    kept: list[str] = []
    dropped: list[str] = []

    for column in candidates:

        train_column = train_numeric[
            column
        ]

        if (
            train_column.notna().sum()
            == 0
        ):
            dropped.append(
                column
            )
            continue

        if (
            train_column.dropna().nunique()
            <= 1
        ):
            dropped.append(
                column
            )
            continue

        kept.append(
            column
        )

    if not kept:
        raise ValueError(
            f"No usable features remain for prefixes: {prefixes}"
        )

    return (
        train_numeric[kept].copy(),
        validation_numeric[kept].copy(),
        test_numeric[kept].copy(),
        kept,
        dropped,
    )


# ============================================================
# MODELS
# ============================================================

def build_models(
    seed: int,
) -> dict[str, Any]:
    """
    Two fixed models are used for ablation:

    Logistic Regression:
        interpretable linear baseline and the current
        validation-selected baseline.

    Extra Trees:
        nonlinear sanity check.

    Keeping the same model configurations across every evidence
    group isolates the effect of the evidence rather than tuning
    each group independently.
    """

    logistic = Pipeline(
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

    extra_trees = Pipeline(
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

    return {
        "Logistic Regression":
            logistic,

        "Extra Trees":
            extra_trees,
    }


# ============================================================
# METRICS
# ============================================================

def positive_probability(
    model,
    features: pd.DataFrame,
) -> np.ndarray:

    probabilities = model.predict_proba(
        features
    )

    return probabilities[
        :,
        1
    ]


def safe_roc_auc(
    y_true: pd.Series,
    probability: np.ndarray,
) -> float:

    if (
        len(
            np.unique(
                y_true
            )
        )
        < 2
    ):
        return float(
            "nan"
        )

    return float(
        roc_auc_score(
            y_true,
            probability,
        )
    )


def safe_pr_auc(
    y_true: pd.Series,
    probability: np.ndarray,
) -> float:

    if (
        len(
            np.unique(
                y_true
            )
        )
        < 2
    ):
        return float(
            "nan"
        )

    return float(
        average_precision_score(
            y_true,
            probability,
        )
    )


def calculate_metrics(
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

        "precision_phishing":
            float(
                precision_score(
                    y_true,
                    prediction,
                    zero_division=0,
                )
            ),

        "recall_phishing":
            float(
                recall_score(
                    y_true,
                    prediction,
                    zero_division=0,
                )
            ),

        "f1_phishing":
            float(
                f1_score(
                    y_true,
                    prediction,
                    zero_division=0,
                )
            ),

        "roc_auc":
            safe_roc_auc(
                y_true,
                probability,
            ),

        "pr_auc":
            safe_pr_auc(
                y_true,
                probability,
            ),
    }


# ============================================================
# CHARTS
# ============================================================

def save_validation_f1_chart(
    results: pd.DataFrame,
    model_name: str,
    output_file: Path,
) -> None:

    subset = (
        results[
            (
                results[
                    "split"
                ]
                == "validation"
            )
            &
            (
                results[
                    "model"
                ]
                == model_name
            )
        ]
        .sort_values(
            "f1_phishing",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    axis.barh(
        subset[
            "feature_group"
        ],
        subset[
            "f1_phishing"
        ],
    )

    axis.set_xlabel(
        "Validation F1 (Phishing)"
    )

    axis.set_ylabel(
        "Evidence Group"
    )

    axis.set_title(
        f"Ablation Study - {model_name}"
    )

    axis.set_xlim(
        0.0,
        1.0,
    )

    figure.tight_layout()

    figure.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# EXPERIMENT
# ============================================================

def run_ablation(
    split_directory: Path,
    output_directory: Path,
    seed: int,
) -> None:

    train_file = (
        split_directory
        / "train.csv"
    )

    validation_file = (
        split_directory
        / "validation.csv"
    )

    test_file = (
        split_directory
        / "test.csv"
    )

    for path in [
        train_file,
        validation_file,
        test_file,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Split file not found: {path}"
            )

    train_df = pd.read_csv(
        train_file
    )

    validation_df = pd.read_csv(
        validation_file
    )

    test_df = pd.read_csv(
        test_file
    )

    y_train = build_target(
        train_df
    )

    y_validation = build_target(
        validation_df
    )

    y_test = build_target(
        test_df
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[
        dict[str, Any]
    ] = []

    feature_manifest: dict[
        str,
        Any,
    ] = {}

    print()
    print(
        "=" * 82
    )

    print(
        "WEBSITE PHISHING FEATURE-GROUP ABLATION"
    )

    print(
        "=" * 82
    )

    print(
        f"Train: {len(train_df)} | "
        f"Validation: {len(validation_df)} | "
        f"Test: {len(test_df)}"
    )

    print(
        "Positive class: PHISHING"
    )

    print(
        "=" * 82
    )

    for group_name, prefixes in FEATURE_GROUPS.items():

        (
            X_train,
            X_validation,
            X_test,
            kept_features,
            dropped_features,
        ) = prepare_group_features(
            train_df=
                train_df,

            validation_df=
                validation_df,

            test_df=
                test_df,

            prefixes=
                prefixes,
        )

        feature_manifest[
            group_name
        ] = {
            "prefixes":
                list(
                    prefixes
                ),

            "feature_count":
                len(
                    kept_features
                ),

            "features":
                kept_features,

            "dropped_train_features":
                dropped_features,
        }

        print()
        print(
            f"[{group_name}] "
            f"{len(kept_features)} usable features"
        )

        models = build_models(
            seed=
                seed
        )

        for model_name, model in models.items():

            model.fit(
                X_train,
                y_train,
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            validation_prediction = (
                model.predict(
                    X_validation
                )
            )

            validation_probability = (
                positive_probability(
                    model,
                    X_validation,
                )
            )

            validation_metrics = (
                calculate_metrics(
                    y_true=
                        y_validation,

                    prediction=
                        validation_prediction,

                    probability=
                        validation_probability,
                )
            )

            rows.append(
                {
                    "feature_group":
                        group_name,

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

            # ------------------------------------------------
            # Test
            # ------------------------------------------------

            test_prediction = (
                model.predict(
                    X_test
                )
            )

            test_probability = (
                positive_probability(
                    model,
                    X_test,
                )
            )

            test_metrics = (
                calculate_metrics(
                    y_true=
                        y_test,

                    prediction=
                        test_prediction,

                    probability=
                        test_probability,
                )
            )

            rows.append(
                {
                    "feature_group":
                        group_name,

                    "model":
                        model_name,

                    "split":
                        "test",

                    "feature_count":
                        len(
                            kept_features
                        ),

                    **test_metrics,
                }
            )

            print(
                f"  {model_name:20s} "
                f"VAL F1={validation_metrics['f1_phishing']:.4f} "
                f"| TEST F1={test_metrics['f1_phishing']:.4f}"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        output_directory
        / "ablation_metrics.csv",
        index=False,
    )

    with (
        output_directory
        / "ablation_feature_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            feature_manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    validation_results = (
        results[
            results[
                "split"
            ]
            == "validation"
        ]
        .sort_values(
            by=[
                "model",
                "f1_phishing",
                "roc_auc",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True,
        )
    )

    validation_results.to_csv(
        output_directory
        / "ablation_validation_ranking.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Compute improvement over best single evidence group
    # for each model.
    # --------------------------------------------------------

    single_groups = {
        "URL only",
        "DOM only",
        "Credential only",
        "Network only",
        "Behavior only",
        "Graph summary only",
    }

    comparison_rows = []

    for model_name in sorted(
        results[
            "model"
        ].unique()
    ):

        model_validation = results[
            (
                results[
                    "model"
                ]
                == model_name
            )
            &
            (
                results[
                    "split"
                ]
                == "validation"
            )
        ].copy()

        singles = model_validation[
            model_validation[
                "feature_group"
            ].isin(
                single_groups
            )
        ]

        all_evidence = model_validation[
            model_validation[
                "feature_group"
            ]
            == "All tabular evidence"
        ]

        if singles.empty or all_evidence.empty:
            continue

        best_single_row = singles.loc[
            singles[
                "f1_phishing"
            ].idxmax()
        ]

        all_row = all_evidence.iloc[
            0
        ]

        comparison_rows.append(
            {
                "model":
                    model_name,

                "best_single_group":
                    best_single_row[
                        "feature_group"
                    ],

                "best_single_validation_f1":
                    float(
                        best_single_row[
                            "f1_phishing"
                        ]
                    ),

                "all_evidence_validation_f1":
                    float(
                        all_row[
                            "f1_phishing"
                        ]
                    ),

                "absolute_f1_gain":
                    float(
                        all_row[
                            "f1_phishing"
                        ]
                        -
                        best_single_row[
                            "f1_phishing"
                        ]
                    ),
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison.to_csv(
        output_directory
        / "ablation_combination_gain.csv",
        index=False,
    )

    # Charts
    for model_name in [
        "Logistic Regression",
        "Extra Trees",
    ]:

        save_validation_f1_chart(
            results=
                results,

            model_name=
                model_name,

            output_file=
                output_directory
                / (
                    model_name.lower()
                    .replace(
                        " ",
                        "_",
                    )
                    + "_validation_f1.png"
                ),
        )

    print()
    print(
        "=" * 82
    )

    print(
        "VALIDATION ABLATION RANKING"
    )

    print(
        "=" * 82
    )

    display_columns = [
        "feature_group",
        "model",
        "feature_count",
        "accuracy",
        "precision_phishing",
        "recall_phishing",
        "f1_phishing",
        "roc_auc",
        "pr_auc",
    ]

    print(
        validation_results[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "COMBINATION GAIN"
    )

    print(
        "=" * 82
    )

    if not comparison.empty:

        print(
            comparison.to_string(
                index=False
            )
        )

    print()
    print(
        "Interpret validation results first. "
        "Do not choose feature groups using test performance."
    )

    print(
        f"Results saved to: {output_directory}"
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
            "Run evidence-group ablation experiments on the "
            "finalized website phishing train/validation/test "
            "splits."
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
            "results/ablation"
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

    run_ablation(
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
