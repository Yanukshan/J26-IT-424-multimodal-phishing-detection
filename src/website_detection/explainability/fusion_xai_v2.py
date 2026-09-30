from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_name(value: str, max_length: int = 70) -> str:
    text = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(value),
    ).strip("_")
    return (text or "sample")[:max_length]


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_branch_probabilities(
    fusion_dir: Path,
    split_name: str,
) -> pd.DataFrame:
    path = fusion_dir / f"{split_name}_branch_probabilities.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Branch probability file not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "source_url",
        "actual_phishing",
        "p_tabular",
        "p_visual",
        "p_graph",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df.copy()
    df["split"] = split_name

    return df


def apply_weights(
    df: pd.DataFrame,
    weights: dict,
    prefix: str,
) -> pd.DataFrame:
    out = df.copy()

    w_tabular = float(weights["tabular"])
    w_visual = float(weights["visual"])
    w_graph = float(weights["graph"])

    out[f"{prefix}_tabular_contribution"] = (
        w_tabular * out["p_tabular"]
    )

    out[f"{prefix}_visual_contribution"] = (
        w_visual * out["p_visual"]
    )

    out[f"{prefix}_graph_contribution"] = (
        w_graph * out["p_graph"]
    )

    out[f"{prefix}_probability"] = (
        out[f"{prefix}_tabular_contribution"]
        + out[f"{prefix}_visual_contribution"]
        + out[f"{prefix}_graph_contribution"]
    )

    out[f"{prefix}_prediction"] = (
        out[f"{prefix}_probability"] >= 0.5
    ).astype(int)

    out[f"{prefix}_correct"] = (
        out[f"{prefix}_prediction"]
        == out["actual_phishing"].astype(int)
    )

    # Share of the final fused probability attributable to each
    # weighted branch. These are descriptive weighted contributions.
    denominator = out[f"{prefix}_probability"].replace(
        0.0,
        np.nan,
    )

    out[f"{prefix}_tabular_share"] = (
        out[f"{prefix}_tabular_contribution"]
        / denominator
    )

    out[f"{prefix}_visual_share"] = (
        out[f"{prefix}_visual_contribution"]
        / denominator
    )

    out[f"{prefix}_graph_share"] = (
        out[f"{prefix}_graph_contribution"]
        / denominator
    )

    return out


def save_case_plot(
    row: pd.Series,
    weights: dict,
    prefix: str,
    output_path: Path,
    title_suffix: str,
) -> None:
    branch_names = [
        "Tabular (LightGBM)",
        "Visual (ResNet18)",
        "Graph (GraphSAGE)",
    ]

    probabilities = [
        float(row["p_tabular"]),
        float(row["p_visual"]),
        float(row["p_graph"]),
    ]

    weight_values = [
        float(weights["tabular"]),
        float(weights["visual"]),
        float(weights["graph"]),
    ]

    contributions = [
        probabilities[i] * weight_values[i]
        for i in range(3)
    ]

    fig = plt.figure(
        figsize=(11, 6)
    )

    ax = fig.add_subplot(1, 1, 1)

    x = np.arange(
        len(branch_names)
    )

    bars = ax.bar(
        x,
        contributions,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        branch_names,
        rotation=15,
        ha="right",
    )

    ax.set_ylabel(
        "Weighted contribution to fused phishing probability"
    )

    fused_probability = sum(
        contributions
    )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1.0,
        label="Decision threshold = 0.5",
    )

    ax.set_title(
        (
            f"{row['source_url']}\n"
            f"{title_suffix} | "
            f"fused p(phishing)={fused_probability:.4f}"
        )
    )

    for bar, probability, weight, contribution in zip(
        bars,
        probabilities,
        weight_values,
        contributions,
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            (
                f"p={probability:.3f}\n"
                f"w={weight:.2f}\n"
                f"c={contribution:.3f}"
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def target_urls_from_lightgbm_misses(
    missed_path: Path,
) -> list[str]:
    if not missed_path.exists():
        return []

    df = pd.read_csv(
        missed_path
    )

    if "source_url" not in df.columns:
        return []

    return list(
        dict.fromkeys(
            df["source_url"]
            .astype(str)
            .tolist()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create fusion-level modality contribution explanations "
            "for Dataset V2."
        )
    )

    parser.add_argument(
        "--fusion-dir",
        default="results/multimodal_fusion_v2",
    )

    parser.add_argument(
        "--lightgbm-misses",
        default=(
            "results/xai_v2/lightgbm/"
            "missed_phishing_explanations.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="results/xai_v2/fusion",
    )

    args = parser.parse_args()

    fusion_dir = Path(
        args.fusion_dir
    )

    missed_path = Path(
        args.lightgbm_misses
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest(
        fusion_dir / "fusion_manifest.json"
    )

    unconstrained_weights = manifest[
        "validation_selected_unconstrained_weights"
    ]

    multimodal_weights = manifest[
        "validation_selected_multimodal_weights"
    ]

    validation = load_branch_probabilities(
        fusion_dir,
        "validation",
    )

    test = load_branch_probabilities(
        fusion_dir,
        "test",
    )

    combined = pd.concat(
        [
            validation,
            test,
        ],
        ignore_index=True,
    )

    explained = apply_weights(
        combined,
        unconstrained_weights,
        "unconstrained",
    )

    explained = apply_weights(
        explained,
        multimodal_weights,
        "multimodal",
    )

    explained.to_csv(
        output_dir
        / "fusion_modality_contributions_all.csv",
        index=False,
    )

    missed_urls = target_urls_from_lightgbm_misses(
        missed_path
    )

    if missed_urls:
        missed_cases = explained[
            explained["source_url"]
            .astype(str)
            .isin(
                missed_urls
            )
        ].copy()
    else:
        missed_cases = explained.iloc[0:0].copy()

    missed_cases.to_csv(
        output_dir
        / "fusion_missed_phishing_explanations.csv",
        index=False,
    )

    plot_dir = (
        output_dir
        / "case_plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for _, row in missed_cases.iterrows():
        slug = (
            f"{row['split']}_"
            f"{safe_name(row['source_url'])}"
        )

        save_case_plot(
            row=row,
            weights=unconstrained_weights,
            prefix="unconstrained",
            output_path=(
                plot_dir
                / f"{slug}_unconstrained.png"
            ),
            title_suffix=(
                "Validation-selected unconstrained fusion"
            ),
        )

        save_case_plot(
            row=row,
            weights=multimodal_weights,
            prefix="multimodal",
            output_path=(
                plot_dir
                / f"{slug}_multimodal.png"
            ),
            title_suffix=(
                "Validation-selected 3-modality fusion"
            ),
        )

    summary_rows = []

    for split_name in [
        "validation",
        "test",
    ]:
        split_df = explained[
            explained["split"]
            == split_name
        ]

        for strategy in [
            "unconstrained",
            "multimodal",
        ]:
            summary_rows.append(
                {
                    "split":
                        split_name,

                    "strategy":
                        strategy,

                    "rows":
                        len(split_df),

                    "mean_tabular_contribution":
                        float(
                            split_df[
                                f"{strategy}_tabular_contribution"
                            ].mean()
                        ),

                    "mean_visual_contribution":
                        float(
                            split_df[
                                f"{strategy}_visual_contribution"
                            ].mean()
                        ),

                    "mean_graph_contribution":
                        float(
                            split_df[
                                f"{strategy}_graph_contribution"
                            ].mean()
                        ),

                    "mean_fused_probability":
                        float(
                            split_df[
                                f"{strategy}_probability"
                            ].mean()
                        ),

                    "correct_predictions":
                        int(
                            split_df[
                                f"{strategy}_correct"
                            ].sum()
                        ),

                    "errors":
                        int(
                            (
                                ~split_df[
                                    f"{strategy}_correct"
                                ]
                            ).sum()
                        ),
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        output_dir
        / "fusion_contribution_summary.csv",
        index=False,
    )

    xai_manifest = {
        "fusion_type":
            "late_probability_fusion",

        "unconstrained_weights":
            unconstrained_weights,

        "multimodal_weights":
            multimodal_weights,

        "explanation":
            (
                "Per-modality contribution is computed as "
                "validation-selected weight multiplied by that branch's "
                "phishing probability. These are descriptive contribution "
                "values for the late-fusion equation, not causal effects."
            ),

        "test_tuning":
            False,

        "note":
            (
                "Weights are read from the existing fusion manifest. "
                "No weights or thresholds are changed by this script."
            ),
    }

    with (
        output_dir
        / "fusion_xai_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            xai_manifest,
            f,
            indent=2,
        )

    print()
    print("=" * 96)
    print("DATASET V2 FUSION-LEVEL EXPLAINABILITY")
    print("=" * 96)

    print(
        "Unconstrained weights: "
        f"tabular={unconstrained_weights['tabular']:.2f}, "
        f"visual={unconstrained_weights['visual']:.2f}, "
        f"graph={unconstrained_weights['graph']:.2f}"
    )

    print(
        "3-modality weights: "
        f"tabular={multimodal_weights['tabular']:.2f}, "
        f"visual={multimodal_weights['visual']:.2f}, "
        f"graph={multimodal_weights['graph']:.2f}"
    )

    print()
    print("FUSION CONTRIBUTION SUMMARY")
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("LIGHTGBM MISSED-PHISHING CASES")
    print()

    if missed_cases.empty:
        print(
            "No missed-phishing cases were found."
        )
    else:
        display_columns = [
            "split",
            "source_url",
            "actual_phishing",
            "p_tabular",
            "p_visual",
            "p_graph",
            "unconstrained_tabular_contribution",
            "unconstrained_visual_contribution",
            "unconstrained_graph_contribution",
            "unconstrained_probability",
            "unconstrained_prediction",
            "multimodal_tabular_contribution",
            "multimodal_visual_contribution",
            "multimodal_graph_contribution",
            "multimodal_probability",
            "multimodal_prediction",
        ]

        print(
            missed_cases[
                display_columns
            ].to_string(
                index=False
            )
        )

    print()
    print(
        f"Outputs: {output_dir}"
    )

    print("=" * 96)


if __name__ == "__main__":
    main()
