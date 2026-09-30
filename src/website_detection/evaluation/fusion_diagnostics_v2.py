from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def predict(probability: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(
            probability,
            errors="raise",
        )
        >= 0.5
    ).astype(int)


def fusion_probability(
    dataframe: pd.DataFrame,
    weights: dict,
) -> pd.Series:
    return (
        float(weights["tabular"])
        * dataframe["p_tabular"]
        + float(weights["visual"])
        * dataframe["p_visual"]
        + float(weights["graph"])
        * dataframe["p_graph"]
    )


def load_split_with_predictions(
    split_file: Path,
    probability_file: Path,
) -> pd.DataFrame:
    split = pd.read_csv(split_file)
    probabilities = pd.read_csv(probability_file)

    required_split = {
        "source_url",
        "source_dataset",
        "class_name",
        "source_label",
    }

    required_probabilities = {
        "source_url",
        "actual_phishing",
        "p_tabular",
        "p_visual",
        "p_graph",
    }

    missing_split = required_split - set(split.columns)
    missing_probabilities = (
        required_probabilities
        - set(probabilities.columns)
    )

    if missing_split:
        raise ValueError(
            f"{split_file} missing: "
            + ", ".join(sorted(missing_split))
        )

    if missing_probabilities:
        raise ValueError(
            f"{probability_file} missing: "
            + ", ".join(
                sorted(missing_probabilities)
            )
        )

    if split["source_url"].duplicated().any():
        raise ValueError(
            f"{split_file} has duplicate source_url values."
        )

    if probabilities["source_url"].duplicated().any():
        raise ValueError(
            f"{probability_file} has duplicate source_url values."
        )

    merged = split[
        [
            "source_url",
            "source_dataset",
            "class_name",
            "source_label",
        ]
    ].merge(
        probabilities,
        on="source_url",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(split):
        raise RuntimeError(
            f"Only {len(merged)} of {len(split)} rows "
            "aligned with branch probabilities."
        )

    expected_target = (
        pd.to_numeric(
            merged["source_label"],
            errors="raise",
        )
        .eq(0)
        .astype(int)
    )

    if not expected_target.equals(
        merged["actual_phishing"].astype(int)
    ):
        raise RuntimeError(
            "Target mismatch after joining split and "
            "branch probability data."
        )

    return merged


def add_predictions(
    dataframe: pd.DataFrame,
    manifest: dict,
) -> pd.DataFrame:
    output = dataframe.copy()

    output["pred_tabular"] = predict(
        output["p_tabular"]
    )

    output["pred_visual"] = predict(
        output["p_visual"]
    )

    output["pred_graph"] = predict(
        output["p_graph"]
    )

    equal_weights = manifest["equal_weights"]

    unconstrained = manifest[
        "validation_selected_unconstrained_weights"
    ]

    multimodal = manifest[
        "validation_selected_multimodal_weights"
    ]

    output["p_equal_fusion"] = fusion_probability(
        output,
        equal_weights,
    )

    output["p_unconstrained_fusion"] = fusion_probability(
        output,
        unconstrained,
    )

    output["p_multimodal_fusion"] = fusion_probability(
        output,
        multimodal,
    )

    output["pred_equal_fusion"] = predict(
        output["p_equal_fusion"]
    )

    output["pred_unconstrained_fusion"] = predict(
        output["p_unconstrained_fusion"]
    )

    output["pred_multimodal_fusion"] = predict(
        output["p_multimodal_fusion"]
    )

    actual = output["actual_phishing"].astype(int)

    for name in [
        "tabular",
        "visual",
        "graph",
        "equal_fusion",
        "unconstrained_fusion",
        "multimodal_fusion",
    ]:
        output[f"error_{name}"] = (
            output[f"pred_{name}"]
            != actual
        )

    return output


def error_summary(
    dataframe: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    actual = dataframe["actual_phishing"].astype(int)

    rows = []

    for name in [
        "tabular",
        "visual",
        "graph",
        "equal_fusion",
        "unconstrained_fusion",
        "multimodal_fusion",
    ]:
        prediction = dataframe[
            f"pred_{name}"
        ].astype(int)

        rows.append(
            {
                "split": split_name,
                "strategy": name,
                "rows": len(dataframe),
                "errors": int(
                    (prediction != actual).sum()
                ),
                "false_positives": int(
                    (
                        (prediction == 1)
                        & (actual == 0)
                    ).sum()
                ),
                "false_negatives": int(
                    (
                        (prediction == 0)
                        & (actual == 1)
                    ).sum()
                ),
                "correct": int(
                    (prediction == actual).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def complementarity_summary(
    dataframe: pd.DataFrame,
    split_name: str,
) -> dict:
    actual = dataframe["actual_phishing"].astype(int)

    tabular_wrong = (
        dataframe["pred_tabular"].astype(int)
        != actual
    )

    visual_correct = (
        dataframe["pred_visual"].astype(int)
        == actual
    )

    graph_correct = (
        dataframe["pred_graph"].astype(int)
        == actual
    )

    equal_correct = (
        dataframe["pred_equal_fusion"].astype(int)
        == actual
    )

    unconstrained_correct = (
        dataframe["pred_unconstrained_fusion"].astype(int)
        == actual
    )

    multimodal_correct = (
        dataframe["pred_multimodal_fusion"].astype(int)
        == actual
    )

    return {
        "split": split_name,
        "tabular_error_count": int(tabular_wrong.sum()),
        "tabular_errors_visual_correct": int(
            (tabular_wrong & visual_correct).sum()
        ),
        "tabular_errors_graph_correct": int(
            (tabular_wrong & graph_correct).sum()
        ),
        "tabular_errors_either_visual_or_graph_correct": int(
            (
                tabular_wrong
                & (
                    visual_correct
                    | graph_correct
                )
            ).sum()
        ),
        "tabular_errors_equal_fusion_correct": int(
            (tabular_wrong & equal_correct).sum()
        ),
        "tabular_errors_unconstrained_fusion_correct": int(
            (
                tabular_wrong
                & unconstrained_correct
            ).sum()
        ),
        "tabular_errors_multimodal_fusion_correct": int(
            (
                tabular_wrong
                & multimodal_correct
            ).sum()
        ),
    }


def source_summary(
    dataframe: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    rows = []

    for source, group in dataframe.groupby(
        "source_dataset",
        dropna=False,
    ):
        actual = group["actual_phishing"].astype(int)

        row = {
            "split": split_name,
            "source_dataset": source,
            "rows": len(group),
            "legitimate_rows": int(
                (actual == 0).sum()
            ),
            "phishing_rows": int(
                (actual == 1).sum()
            ),
        }

        for name in [
            "tabular",
            "visual",
            "graph",
            "equal_fusion",
            "unconstrained_fusion",
            "multimodal_fusion",
        ]:
            prediction = group[
                f"pred_{name}"
            ].astype(int)

            row[f"{name}_errors"] = int(
                (prediction != actual).sum()
            )

            row[f"{name}_false_positives"] = int(
                (
                    (prediction == 1)
                    & (actual == 0)
                ).sum()
            )

            row[f"{name}_false_negatives"] = int(
                (
                    (prediction == 0)
                    & (actual == 1)
                ).sum()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def disagreement_rows(
    dataframe: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    actual = dataframe["actual_phishing"].astype(int)

    tabular_wrong = (
        dataframe["pred_tabular"].astype(int)
        != actual
    )

    other_branch_correct = (
        (
            dataframe["pred_visual"].astype(int)
            == actual
        )
        | (
            dataframe["pred_graph"].astype(int)
            == actual
        )
    )

    selected = dataframe.loc[
        tabular_wrong
        | other_branch_correct & tabular_wrong
    ].copy()

    selected.insert(
        0,
        "split",
        split_name,
    )

    columns = [
        "split",
        "source_url",
        "source_dataset",
        "class_name",
        "actual_phishing",
        "p_tabular",
        "p_visual",
        "p_graph",
        "p_equal_fusion",
        "p_unconstrained_fusion",
        "p_multimodal_fusion",
        "pred_tabular",
        "pred_visual",
        "pred_graph",
        "pred_equal_fusion",
        "pred_unconstrained_fusion",
        "pred_multimodal_fusion",
    ]

    return selected[columns]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose modality complementarity and source-level "
            "errors for Dataset V2 multimodal fusion."
        )
    )

    parser.add_argument(
        "--split-dir",
        default="data/processed/final_splits_v2",
    )

    parser.add_argument(
        "--fusion-dir",
        default="results/multimodal_fusion_v2",
    )

    parser.add_argument(
        "--output-dir",
        default="results/fusion_diagnostics_v2",
    )

    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    fusion_dir = Path(args.fusion_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = fusion_dir / "fusion_manifest.json"

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    validation = load_split_with_predictions(
        split_dir / "validation.csv",
        fusion_dir / "validation_branch_probabilities.csv",
    )

    test = load_split_with_predictions(
        split_dir / "test.csv",
        fusion_dir / "test_branch_probabilities.csv",
    )

    validation = add_predictions(
        validation,
        manifest,
    )

    test = add_predictions(
        test,
        manifest,
    )

    errors = pd.concat(
        [
            error_summary(
                validation,
                "validation",
            ),
            error_summary(
                test,
                "test",
            ),
        ],
        ignore_index=True,
    )

    complementarity = pd.DataFrame(
        [
            complementarity_summary(
                validation,
                "validation",
            ),
            complementarity_summary(
                test,
                "test",
            ),
        ]
    )

    by_source = pd.concat(
        [
            source_summary(
                validation,
                "validation",
            ),
            source_summary(
                test,
                "test",
            ),
        ],
        ignore_index=True,
    )

    disagreements = pd.concat(
        [
            disagreement_rows(
                validation,
                "validation",
            ),
            disagreement_rows(
                test,
                "test",
            ),
        ],
        ignore_index=True,
    )

    errors.to_csv(
        output_dir / "strategy_error_summary.csv",
        index=False,
    )

    complementarity.to_csv(
        output_dir / "branch_complementarity.csv",
        index=False,
    )

    by_source.to_csv(
        output_dir / "source_error_summary.csv",
        index=False,
    )

    disagreements.to_csv(
        output_dir / "tabular_error_cases.csv",
        index=False,
    )

    print()
    print("=" * 92)
    print("DATASET V2 FUSION DIAGNOSTICS")
    print("=" * 92)

    print()
    print("OVERALL ERROR COUNTS")
    print(errors.to_string(index=False))

    print()
    print("BRANCH COMPLEMENTARITY")
    print(complementarity.to_string(index=False))

    print()
    print("SOURCE-LEVEL ERROR COUNTS")
    print(by_source.to_string(index=False))

    print()
    print("TABULAR ERROR CASES")
    if disagreements.empty:
        print("No tabular errors found.")
    else:
        display_columns = [
            "split",
            "source_url",
            "source_dataset",
            "actual_phishing",
            "p_tabular",
            "p_visual",
            "p_graph",
            "pred_tabular",
            "pred_visual",
            "pred_graph",
            "pred_unconstrained_fusion",
            "pred_multimodal_fusion",
        ]

        print(
            disagreements[
                display_columns
            ].to_string(index=False)
        )

    print()
    print(f"Outputs: {output_dir}")
    print("=" * 92)


if __name__ == "__main__":
    main()
