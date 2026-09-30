from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.explain import Explainer, GNNExplainer

from src.website_detection.models.gnn_v2_trainer import (
    EDGE_TYPES,
    METHODS,
    NODE_TYPES,
    RESOURCE_TYPES,
    GraphClassifier,
    get_attr,
    graph_json_to_data,
    source_label_to_target,
)


def safe_name(value: str, max_length: int = 70) -> str:
    text = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(value),
    ).strip("_")

    return (text or "sample")[:max_length]


def node_feature_names() -> list[str]:
    names = []

    names.extend(
        [
            f"node_type:{value}"
            for value in NODE_TYPES
        ]
    )

    names.extend(
        [
            f"method:{value}"
            for value in METHODS
        ]
    )

    names.extend(
        [
            f"resource_type:{value}"
            for value in RESOURCE_TYPES
        ]
    )

    names.extend(
        [
            "external",
            "runtime_requested",
            "has_url",
            "has_hostname",
            "log_in_degree",
            "log_out_degree",
            "log_total_degree",
        ]
    )

    names.extend(
        [
            f"incoming_edge_count:{value}"
            for value in EDGE_TYPES
        ]
    )

    names.extend(
        [
            f"outgoing_edge_count:{value}"
            for value in EDGE_TYPES
        ]
    )

    return names


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
):
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location=device,
        )


def build_model(
    checkpoint: dict,
    device: torch.device,
) -> GraphClassifier:
    architecture = str(
        checkpoint.get(
            "architecture",
            "graphsage",
        )
    ).lower()

    input_dim = int(
        checkpoint[
            "input_dim"
        ]
    )

    hidden_dim = int(
        checkpoint.get(
            "hidden_dim",
            64,
        )
    )

    dropout = float(
        checkpoint.get(
            "dropout",
            0.30,
        )
    )

    model = GraphClassifier(
        architecture=architecture,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    return model


class GraphModelWrapper(nn.Module):
    """
    PyG Explainer-friendly wrapper around the project's
    graph-level classifier.
    """

    def __init__(
        self,
        model: GraphClassifier,
    ) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x,
        edge_index,
        batch,
    ):
        return self.model(
            x,
            edge_index,
            batch,
        )


def load_raw_nodes(
    graph_path: Path,
) -> list[dict]:
    with graph_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(
            file
        )

    nodes = payload.get(
        "nodes",
        []
    )

    if not isinstance(nodes, list):
        return []

    return nodes


def describe_node(
    node: dict,
    index: int,
) -> dict:
    node_type = str(
        get_attr(
            node,
            "node_type",
            "other",
        )
        or "other"
    )

    url = str(
        get_attr(
            node,
            "url",
            "",
        )
        or ""
    )

    hostname = str(
        get_attr(
            node,
            "hostname",
            "",
        )
        or ""
    )

    method = str(
        get_attr(
            node,
            "method",
            "",
        )
        or ""
    )

    resource_type = str(
        get_attr(
            node,
            "resource_type",
            "",
        )
        or ""
    )

    external = bool(
        get_attr(
            node,
            "external",
            False,
        )
    )

    runtime_requested = bool(
        get_attr(
            node,
            "runtime_requested",
            False,
        )
    )

    label_parts = [
        f"#{index}",
        node_type,
    ]

    if hostname:
        label_parts.append(
            hostname[:30]
        )
    elif url:
        label_parts.append(
            url[:30]
        )

    return {
        "node_index":
            index,

        "node_type":
            node_type,

        "hostname":
            hostname,

        "url":
            url,

        "method":
            method,

        "resource_type":
            resource_type,

        "external":
            external,

        "runtime_requested":
            runtime_requested,

        "node_label":
            " | ".join(
                label_parts
            ),
    }


def prediction_for_graph(
    model: GraphClassifier,
    data,
    device: torch.device,
):
    data = data.to(
        device
    )

    batch = torch.zeros(
        data.num_nodes,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        logits = model(
            data.x,
            data.edge_index,
            batch,
        )

        probability = torch.softmax(
            logits,
            dim=1,
        )[
            0,
            1
        ].item()

        prediction = int(
            torch.argmax(
                logits,
                dim=1,
            ).item()
        )

    return (
        prediction,
        float(
            probability
        ),
    )


def explain_graph(
    model: GraphClassifier,
    data,
    device: torch.device,
    epochs: int,
):
    wrapped = GraphModelWrapper(
        model
    ).to(
        device
    )

    wrapped.eval()

    explainer = Explainer(
        model=wrapped,
        algorithm=GNNExplainer(
            epochs=epochs,
        ),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            # PyG 2.8 expects the full ModelMode enum value.
            # "multiclass" is invalid; our model outputs two logits
            # (legitimate / phishing), so this is multiclass classification.
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
    )

    data = data.to(
        device
    )

    batch = torch.zeros(
        data.num_nodes,
        dtype=torch.long,
        device=device,
    )

    explanation = explainer(
        data.x,
        data.edge_index,
        batch=batch,
        index=0,
    )

    return explanation


def node_importance_from_mask(
    node_mask: torch.Tensor | None,
    number_of_nodes: int,
) -> np.ndarray:
    if node_mask is None:
        return np.zeros(
            number_of_nodes,
            dtype=np.float64,
        )

    mask = (
        node_mask
        .detach()
        .cpu()
        .numpy()
    )

    if mask.ndim == 1:
        return np.abs(
            mask
        )

    return np.mean(
        np.abs(
            mask
        ),
        axis=1,
    )


def feature_importance_from_mask(
    node_mask: torch.Tensor | None,
    feature_count: int,
) -> np.ndarray:
    if node_mask is None:
        return np.zeros(
            feature_count,
            dtype=np.float64,
        )

    mask = (
        node_mask
        .detach()
        .cpu()
        .numpy()
    )

    if mask.ndim == 1:
        return np.zeros(
            feature_count,
            dtype=np.float64,
        )

    return np.mean(
        np.abs(
            mask
        ),
        axis=0,
    )


def edge_importance_from_mask(
    edge_mask: torch.Tensor | None,
    edge_count: int,
) -> np.ndarray:
    if edge_mask is None:
        return np.zeros(
            edge_count,
            dtype=np.float64,
        )

    return (
        edge_mask
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )


def plot_top_nodes(
    dataframe: pd.DataFrame,
    output_path: Path,
    title: str,
    top_n: int = 15,
) -> None:
    top = (
        dataframe
        .sort_values(
            "node_importance",
            ascending=False,
        )
        .head(
            top_n
        )
        .sort_values(
            "node_importance",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(
            10,
            7,
        )
    )

    plt.barh(
        top[
            "node_label"
        ],
        top[
            "node_importance"
        ],
    )

    plt.xlabel(
        "GNNExplainer node importance"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def plot_top_features(
    dataframe: pd.DataFrame,
    output_path: Path,
    title: str,
    top_n: int = 15,
) -> None:
    top = (
        dataframe
        .sort_values(
            "feature_importance",
            ascending=False,
        )
        .head(
            top_n
        )
        .sort_values(
            "feature_importance",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(
            10,
            7,
        )
    )

    plt.barh(
        top[
            "feature"
        ],
        top[
            "feature_importance"
        ],
    )

    plt.xlabel(
        "GNNExplainer feature importance"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def find_sample_row(
    split_dir: Path,
    source_url: str,
):
    for split_name in [
        "validation",
        "test",
        "train",
    ]:
        split_path = (
            split_dir
            / f"{split_name}.csv"
        )

        dataframe = pd.read_csv(
            split_path
        )

        matches = dataframe[
            dataframe[
                "source_url"
            ].astype(str)
            == str(
                source_url
            )
        ]

        if not matches.empty:
            return (
                split_name,
                matches.iloc[
                    0
                ],
            )

    return (
        None,
        None,
    )


def unique_urls_from_lightgbm_misses(
    path: Path,
) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"LightGBM missed-phishing file not found: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    if (
        "source_url"
        not in dataframe.columns
    ):
        raise ValueError(
            "missed_phishing_explanations.csv does not "
            "contain source_url."
        )

    return list(
        dict.fromkeys(
            dataframe[
                "source_url"
            ]
            .astype(str)
            .tolist()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Explain GraphSAGE predictions for the phishing "
            "pages missed by the Dataset V2 LightGBM model."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "models/website_detection/"
            "gnn_v2/graphsage_v2.pt"
        ),
    )

    parser.add_argument(
        "--split-dir",
        default=(
            "data/processed/"
            "final_splits_v2"
        ),
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
        default=(
            "results/xai_v2/graphsage"
        ),
    )

    parser.add_argument(
        "--explainer-epochs",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--top-nodes",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--top-edges",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    checkpoint_path = Path(
        args.checkpoint
    )

    split_dir = Path(
        args.split_dir
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

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"GraphSAGE checkpoint not found: {checkpoint_path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = load_checkpoint(
        checkpoint_path,
        device,
    )

    model = build_model(
        checkpoint,
        device,
    )

    feature_names = node_feature_names()

    expected_input_dim = int(
        checkpoint[
            "input_dim"
        ]
    )

    if (
        len(
            feature_names
        )
        != expected_input_dim
    ):
        raise RuntimeError(
            "Feature-name layout does not match the saved "
            f"GraphSAGE input dimension: names={len(feature_names)}, "
            f"checkpoint={expected_input_dim}"
        )

    target_urls = unique_urls_from_lightgbm_misses(
        missed_path
    )

    summary_rows = []

    print()
    print(
        "=" * 92
    )

    print(
        "GRAPHSAGE GNN EXPLAINABILITY - DATASET V2"
    )

    print(
        "=" * 92
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    print(
        f"PyG GNNExplainer epochs per graph: "
        f"{args.explainer_epochs}"
    )

    print(
        f"LightGBM missed-phishing URLs to explain: "
        f"{len(target_urls)}"
    )

    print(
        "=" * 92
    )

    for source_url in target_urls:
        split_name, row = find_sample_row(
            split_dir,
            source_url,
        )

        if row is None:
            print(
                f"SKIP: URL not found in splits: {source_url}"
            )
            continue

        graph_path = Path(
            str(
                row[
                    "graph_artifact_path"
                ]
            )
        )

        if not graph_path.exists():
            print(
                f"SKIP: graph file missing: {graph_path}"
            )
            continue

        actual_target = source_label_to_target(
            row[
                "source_label"
            ]
        )

        data = graph_json_to_data(
            graph_path=
                graph_path,

            target=
                actual_target,

            sample_index=
                0,
        )

        prediction, probability = prediction_for_graph(
            model,
            data,
            device,
        )

        print()
        print(
            f"Explaining: {source_url}"
        )

        print(
            f"  split={split_name}"
        )

        print(
            f"  actual={'phishing' if actual_target == 1 else 'legitimate'}"
        )

        print(
            f"  GraphSAGE prediction="
            f"{'phishing' if prediction == 1 else 'legitimate'}"
        )

        print(
            f"  p(phishing)={probability:.6f}"
        )

        print(
            f"  nodes={data.num_nodes}, "
            f"message_edges={data.edge_index.shape[1]}"
        )

        explanation = explain_graph(
            model=
                model,

            data=
                data,

            device=
                device,

            epochs=
                args.explainer_epochs,
        )

        node_importance = node_importance_from_mask(
            explanation.node_mask,
            data.num_nodes,
        )

        feature_importance = feature_importance_from_mask(
            explanation.node_mask,
            len(
                feature_names
            ),
        )

        edge_importance = edge_importance_from_mask(
            explanation.edge_mask,
            data.edge_index.shape[
                1
            ],
        )

        raw_nodes = load_raw_nodes(
            graph_path
        )

        node_rows = []

        for node_index in range(
            data.num_nodes
        ):
            raw_node = (
                raw_nodes[
                    node_index
                ]
                if node_index
                < len(
                    raw_nodes
                )
                else {}
            )

            description = describe_node(
                raw_node,
                node_index,
            )

            node_rows.append(
                {
                    **description,
                    "node_importance":
                        float(
                            node_importance[
                                node_index
                            ]
                        ),
                }
            )

        node_frame = pd.DataFrame(
            node_rows
        ).sort_values(
            "node_importance",
            ascending=False,
        )

        feature_frame = pd.DataFrame(
            {
                "feature":
                    feature_names,

                "feature_importance":
                    feature_importance,
            }
        ).sort_values(
            "feature_importance",
            ascending=False,
        )

        edge_rows = []

        edge_index_cpu = (
            data.edge_index
            .detach()
            .cpu()
            .numpy()
        )

        for edge_position in range(
            edge_index_cpu.shape[
                1
            ]
        ):
            source_index = int(
                edge_index_cpu[
                    0,
                    edge_position
                ]
            )

            target_index = int(
                edge_index_cpu[
                    1,
                    edge_position
                ]
            )

            source_node = (
                node_rows[
                    source_index
                ]
            )

            target_node = (
                node_rows[
                    target_index
                ]
            )

            edge_rows.append(
                {
                    "edge_position":
                        edge_position,

                    "source_index":
                        source_index,

                    "source_type":
                        source_node[
                            "node_type"
                        ],

                    "source_hostname":
                        source_node[
                            "hostname"
                        ],

                    "target_index":
                        target_index,

                    "target_type":
                        target_node[
                            "node_type"
                        ],

                    "target_hostname":
                        target_node[
                            "hostname"
                        ],

                    "edge_importance":
                        float(
                            edge_importance[
                                edge_position
                            ]
                        ),
                }
            )

        edge_frame = pd.DataFrame(
            edge_rows
        ).sort_values(
            "edge_importance",
            ascending=False,
        )

        slug = (
            f"{split_name}_"
            f"{safe_name(source_url)}"
        )

        node_frame.to_csv(
            output_dir
            / f"{slug}_top_nodes.csv",
            index=False,
        )

        feature_frame.to_csv(
            output_dir
            / f"{slug}_node_feature_importance.csv",
            index=False,
        )

        edge_frame.to_csv(
            output_dir
            / f"{slug}_top_edges.csv",
            index=False,
        )

        plot_top_nodes(
            dataframe=
                node_frame,

            output_path=
                output_dir
                / f"{slug}_top_nodes.png",

            title=(
                "GraphSAGE explanation - top resource nodes\n"
                f"{source_url}"
            ),

            top_n=
                args.top_nodes,
        )

        plot_top_features(
            dataframe=
                feature_frame,

            output_path=
                output_dir
                / f"{slug}_top_node_features.png",

            title=(
                "GraphSAGE explanation - node feature importance\n"
                f"{source_url}"
            ),

            top_n=
                15,
        )

        top_nodes = (
            node_frame.head(
                5
            )[
                [
                    "node_index",
                    "node_type",
                    "hostname",
                    "external",
                    "runtime_requested",
                    "node_importance",
                ]
            ]
        )

        print(
            "  Top 5 important nodes:"
        )

        print(
            top_nodes.to_string(
                index=False
            )
        )

        print(
            "  Top 10 important node features:"
        )

        print(
            feature_frame.head(
                10
            ).to_string(
                index=False
            )
        )

        summary_rows.append(
            {
                "split":
                    split_name,

                "source_url":
                    source_url,

                "source_dataset":
                    str(
                        row.get(
                            "source_dataset",
                            "",
                        )
                    ),

                "actual_phishing":
                    int(
                        actual_target
                    ),

                "graphsage_predicted_phishing":
                    int(
                        prediction
                    ),

                "graphsage_phishing_probability":
                    float(
                        probability
                    ),

                "graph_correct":
                    bool(
                        prediction
                        == actual_target
                    ),

                "node_count":
                    int(
                        data.num_nodes
                    ),

                "message_edge_count":
                    int(
                        data.edge_index.shape[
                            1
                        ]
                    ),

                "top_node_type":
                    (
                        str(
                            node_frame.iloc[
                                0
                            ][
                                "node_type"
                            ]
                        )
                        if not node_frame.empty
                        else ""
                    ),

                "top_node_hostname":
                    (
                        str(
                            node_frame.iloc[
                                0
                            ][
                                "hostname"
                            ]
                        )
                        if not node_frame.empty
                        else ""
                    ),

                "top_node_importance":
                    (
                        float(
                            node_frame.iloc[
                                0
                            ][
                                "node_importance"
                            ]
                        )
                        if not node_frame.empty
                        else 0.0
                    ),

                "top_node_feature":
                    (
                        str(
                            feature_frame.iloc[
                                0
                            ][
                                "feature"
                            ]
                        )
                        if not feature_frame.empty
                        else ""
                    ),

                "top_node_feature_importance":
                    (
                        float(
                            feature_frame.iloc[
                                0
                            ][
                                "feature_importance"
                            ]
                        )
                        if not feature_frame.empty
                        else 0.0
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        output_dir
        / "graphsage_explanation_summary.csv",
        index=False,
    )

    manifest = {
        "checkpoint":
            str(
                checkpoint_path
            ),

        "architecture":
            "GraphSAGE",

        "explainer":
            "PyG GNNExplainer",

        "explanation_type":
            "model",

        "task_level":
            "graph",

        "explainer_epochs_per_graph":
            args.explainer_epochs,

        "graphs_requested":
            len(
                target_urls
            ),

        "graphs_explained":
            len(
                summary
            ),

        "node_feature_dimension":
            len(
                feature_names
            ),

        "note":
            (
                "This XAI stage is descriptive only. "
                "It does not retrain GraphSAGE, change its "
                "threshold, or use explanations to tune the "
                "current test-set model."
            ),
    }

    with (
        output_dir
        / "graphsage_xai_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print()
    print(
        "=" * 92
    )

    print(
        "GRAPHSAGE EXPLANATION SUMMARY"
    )

    print(
        "=" * 92
    )

    if summary.empty:
        print(
            "No graphs were explained."
        )
    else:
        print(
            summary[
                [
                    "split",
                    "source_url",
                    "actual_phishing",
                    "graphsage_predicted_phishing",
                    "graphsage_phishing_probability",
                    "graph_correct",
                    "node_count",
                    "message_edge_count",
                    "top_node_type",
                    "top_node_hostname",
                    "top_node_feature",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        f"Outputs: {output_dir}"
    )

    print(
        "=" * 92
    )


if __name__ == "__main__":
    main()
