from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = Path(
    "results/final_external_evaluation_2026"
)

DEFAULT_OUTPUT_DIR = Path(
    "results/final_external_evaluation_2026/thesis_artifacts"
)


MODEL_ORDER = [
    "LightGBM",
    "ResNet18",
    "GraphSAGE",
    "Fusion_80_10_10",
]


def percent(value: float) -> float:
    return float(value) * 100.0


def load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def model_label(value: str) -> str:
    mapping = {
        "LightGBM": "LightGBM",
        "ResNet18": "ResNet18",
        "GraphSAGE": "GraphSAGE",
        "Fusion_80_10_10": "Fusion 80/10/10",
    }
    return mapping.get(str(value), str(value))


def save_metric_bar_chart(
    frame: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    as_percent: bool = True,
) -> None:
    ordered = (
        frame.set_index("model")
        .reindex(MODEL_ORDER)
        .reset_index()
    )

    values = ordered[metric].astype(float).to_numpy()
    plot_values = values * 100.0 if as_percent else values
    labels = [model_label(model) for model in ordered["model"]]

    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(1, 1, 1)
    bars = axis.bar(labels, plot_values)

    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=15)

    if as_percent:
        axis.set_ylim(0, 105)

    for bar, value in zip(bars, plot_values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrix(
    row: pd.Series,
    output_path: Path,
) -> None:
    matrix = np.array(
        [
            [int(row["tn"]), int(row["fp"])],
            [int(row["fn"]), int(row["tp"])],
        ]
    )

    figure = plt.figure(figsize=(6, 5))
    axis = figure.add_subplot(1, 1, 1)
    image = axis.imshow(matrix)

    axis.set_xticks(
        [0, 1],
        ["Predicted Legitimate", "Predicted Phishing"],
    )
    axis.set_yticks(
        [0, 1],
        ["Actual Legitimate", "Actual Phishing"],
    )
    axis.set_title(
        f"{model_label(row['model'])} - Full External Confusion Matrix"
    )

    for i in range(2):
        for j in range(2):
            axis.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
            )

    figure.colorbar(image, ax=axis)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_source_error_chart(
    errors: pd.DataFrame,
    output_path: Path,
) -> None:
    grouped = (
        errors.groupby(
            ["source_dataset", "actual_phishing"],
            dropna=False,
        )
        .size()
        .reset_index(name="error_count")
    )

    grouped["label"] = grouped.apply(
        lambda row: (
            f"{row['source_dataset']}\n"
            + (
                "Phishing FN"
                if int(row["actual_phishing"]) == 1
                else "Legitimate FP"
            )
        ),
        axis=1,
    )

    figure = plt.figure(figsize=(10, 6))
    axis = figure.add_subplot(1, 1, 1)
    bars = axis.bar(
        grouped["label"],
        grouped["error_count"],
    )

    axis.set_title("Fusion Errors by External Source")
    axis.set_ylabel("Number of Errors")
    axis.tick_params(axis="x", rotation=20)

    for bar, value in zip(bars, grouped["error_count"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(value)),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_thesis_model_table(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    full = metrics[
        metrics["evaluation_set"].astype(str).eq("full_external")
    ].copy()

    full["model"] = full["model"].map(model_label)

    table = full[
        [
            "model",
            "rows",
            "accuracy",
            "balanced_accuracy",
            "precision_phishing",
            "recall_phishing",
            "f1_phishing",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "ece_10",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
    ].copy()

    for column in [
        "accuracy",
        "balanced_accuracy",
        "precision_phishing",
        "recall_phishing",
        "f1_phishing",
        "roc_auc",
        "pr_auc",
    ]:
        table[column] = table[column].astype(float) * 100.0

    return table


def build_source_summary(
    source_metrics: pd.DataFrame,
) -> pd.DataFrame:
    full = source_metrics[
        source_metrics["evaluation_set"].astype(str).eq("full_external")
    ].copy()

    full["model"] = full["model"].map(model_label)

    table = full[
        [
            "source_dataset",
            "model",
            "rows",
            "accuracy",
            "precision_phishing",
            "recall_phishing",
            "f1_phishing",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
    ].copy()

    for column in [
        "accuracy",
        "precision_phishing",
        "recall_phishing",
        "f1_phishing",
    ]:
        table[column] = table[column].astype(float) * 100.0

    return table


def write_results_summary(
    metrics: pd.DataFrame,
    review: pd.DataFrame,
    errors: pd.DataFrame,
    output_path: Path,
) -> None:
    full_metrics = metrics[
        metrics["evaluation_set"].astype(str).eq("full_external")
    ].set_index("model")

    lightgbm = full_metrics.loc["LightGBM"]
    fusion = full_metrics.loc["Fusion_80_10_10"]
    graph = full_metrics.loc["GraphSAGE"]
    visual = full_metrics.loc["ResNet18"]

    review_row = review[
        review["evaluation_set"].astype(str).eq("full_external")
    ].iloc[0]

    error_breakdown = (
        errors.groupby(
            ["source_dataset", "actual_phishing"],
            dropna=False,
        )
        .size()
    )

    lines = [
        "FINAL EXTERNAL EVALUATION - THESIS SUMMARY",
        "=" * 60,
        "",
        "Dataset",
        "-------",
        "Fresh external full dataset: 11,140 pages",
        "Legitimate: 8,816",
        "Phishing: 2,324",
        "Balanced subset: 4,648 pages (2,324 per class)",
        "Dataset V2 registrable-domain overlap: 0",
        "Cross-class domain conflicts: 0",
        "",
        "Frozen evaluation protocol",
        "--------------------------",
        "No retraining",
        "No threshold tuning",
        "No fusion-weight search",
        "No external-set calibration fitting",
        "Decision threshold: 0.50",
        "Fusion: LightGBM 0.80 + ResNet18 0.10 + GraphSAGE 0.10",
        "",
        "Full external results",
        "---------------------",
        (
            "LightGBM: "
            f"accuracy={percent(lightgbm['accuracy']):.2f}%, "
            f"balanced_accuracy={percent(lightgbm['balanced_accuracy']):.2f}%, "
            f"precision={percent(lightgbm['precision_phishing']):.2f}%, "
            f"recall={percent(lightgbm['recall_phishing']):.2f}%, "
            f"F1={percent(lightgbm['f1_phishing']):.2f}%"
        ),
        (
            "Fusion 80/10/10: "
            f"accuracy={percent(fusion['accuracy']):.2f}%, "
            f"balanced_accuracy={percent(fusion['balanced_accuracy']):.2f}%, "
            f"precision={percent(fusion['precision_phishing']):.2f}%, "
            f"recall={percent(fusion['recall_phishing']):.2f}%, "
            f"F1={percent(fusion['f1_phishing']):.2f}%"
        ),
        (
            "GraphSAGE: "
            f"recall={percent(graph['recall_phishing']):.2f}%, "
            f"F1={percent(graph['f1_phishing']):.2f}%"
        ),
        (
            "ResNet18: "
            f"recall={percent(visual['recall_phishing']):.2f}%, "
            f"F1={percent(visual['f1_phishing']):.2f}%"
        ),
        "",
        "Key interpretation",
        "------------------",
        (
            "The frozen LightGBM tabular branch remained the strongest "
            "classifier on the fresh external dataset."
        ),
        (
            "The fixed 80/10/10 multimodal fusion preserved similar "
            "performance but did not improve F1 or recall over LightGBM."
        ),
        (
            "GraphSAGE and ResNet18 are therefore best interpreted as "
            "complementary evidence branches and disagreement signals "
            "rather than direct accuracy-improving branches in this experiment."
        ),
        "",
        "Review mechanism",
        "----------------",
        (
            f"Review-required pages: "
            f"{int(review_row['review_required_count']):,} "
            f"({percent(review_row['review_required_rate']):.2f}%)"
        ),
        f"Fusion errors: {int(review_row['fusion_errors']):,}",
        (
            f"Fusion errors flagged for review: "
            f"{int(review_row['fusion_errors_flagged_for_review']):,}"
        ),
        (
            f"Fusion error review-capture rate: "
            f"{percent(review_row['fusion_error_review_capture_rate']):.2f}%"
        ),
        "",
        "Important limitation",
        "--------------------",
        (
            "The external dataset is domain-disjoint from Dataset V2, "
            "but source and class remain associated: Tranco and Cloudflare "
            "Radar are legitimate sources, while PhishTank and OpenPhish "
            "are phishing sources. Strong tabular performance may therefore "
            "partly reflect source-associated characteristics."
        ),
        "",
        "Fusion error source breakdown",
        "-----------------------------",
    ]

    for (
        source_dataset,
        actual_phishing,
    ), count in error_breakdown.items():
        error_type = (
            "False Negative"
            if int(actual_phishing) == 1
            else "False Positive"
        )

        lines.append(
            f"{source_dataset}: {int(count)} {error_type}(s)"
        )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis-ready tables, figures and a concise "
            "results summary from the frozen final external evaluation."
        )
    )

    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_required_csv(
        results_dir / "final_external_metrics.csv"
    )
    source_metrics = load_required_csv(
        results_dir / "final_external_source_metrics.csv"
    )
    review = load_required_csv(
        results_dir / "final_external_review_summary.csv"
    )
    errors = load_required_csv(
        results_dir / "final_external_full_fusion_errors.csv"
    )

    full_metrics = metrics[
        metrics["evaluation_set"].astype(str).eq("full_external")
    ].copy()

    thesis_model_table = build_thesis_model_table(metrics)
    thesis_model_table.to_csv(
        output_dir / "thesis_model_comparison_full_external.csv",
        index=False,
    )

    thesis_source_table = build_source_summary(source_metrics)
    thesis_source_table.to_csv(
        output_dir / "thesis_source_performance_full_external.csv",
        index=False,
    )

    review.to_csv(
        output_dir / "thesis_review_summary.csv",
        index=False,
    )

    save_metric_bar_chart(
        full_metrics,
        metric="f1_phishing",
        title="Final External Phishing F1 Score",
        ylabel="F1 Score (%)",
        output_path=output_dir / "figure_external_f1_comparison.png",
    )

    save_metric_bar_chart(
        full_metrics,
        metric="balanced_accuracy",
        title="Final External Balanced Accuracy",
        ylabel="Balanced Accuracy (%)",
        output_path=output_dir / "figure_external_balanced_accuracy.png",
    )

    save_metric_bar_chart(
        full_metrics,
        metric="roc_auc",
        title="Final External ROC-AUC",
        ylabel="ROC-AUC (%)",
        output_path=output_dir / "figure_external_roc_auc.png",
    )

    save_metric_bar_chart(
        full_metrics,
        metric="brier_score",
        title="Final External Brier Score",
        ylabel="Brier Score (Lower is Better)",
        output_path=output_dir / "figure_external_brier_score.png",
        as_percent=False,
    )

    for _, row in full_metrics.iterrows():
        slug = (
            str(row["model"])
            .lower()
            .replace("/", "_")
            .replace(" ", "_")
        )
        save_confusion_matrix(
            row,
            output_dir / f"confusion_matrix_{slug}.png",
        )

    save_source_error_chart(
        errors,
        output_dir / "figure_fusion_errors_by_source.png",
    )

    write_results_summary(
        metrics,
        review,
        errors,
        output_dir / "thesis_results_summary.txt",
    )

    print()
    print("=" * 88)
    print("THESIS RESULT ARTIFACTS GENERATED")
    print("=" * 88)
    print(f"Output directory: {output_dir}")
    print()
    print("Tables:")
    print("  thesis_model_comparison_full_external.csv")
    print("  thesis_source_performance_full_external.csv")
    print("  thesis_review_summary.csv")
    print()
    print("Figures:")
    print("  figure_external_f1_comparison.png")
    print("  figure_external_balanced_accuracy.png")
    print("  figure_external_roc_auc.png")
    print("  figure_external_brier_score.png")
    print("  confusion_matrix_lightgbm.png")
    print("  confusion_matrix_resnet18.png")
    print("  confusion_matrix_graphsage.png")
    print("  confusion_matrix_fusion_80_10_10.png")
    print("  figure_fusion_errors_by_source.png")
    print()
    print("Text:")
    print("  thesis_results_summary.txt")
    print("=" * 88)


if __name__ == "__main__":
    main()
