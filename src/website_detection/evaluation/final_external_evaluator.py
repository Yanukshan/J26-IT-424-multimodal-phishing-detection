from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torchvision.models import ResNet18_Weights, resnet18

from torch_geometric.loader import DataLoader as GraphDataLoader

from src.website_detection.models.gnn_v2_trainer import (
    GraphClassifier,
    WebsiteGraphDataset,
    get_device,
)


DEFAULT_EXTERNAL_FULL = (
    "data/processed/final_external_2026/final/"
    "final_external_full.csv"
)

DEFAULT_EXTERNAL_BALANCED = (
    "data/processed/final_external_2026/final/"
    "final_external_balanced.csv"
)

DEFAULT_LIGHTGBM_MODEL = (
    "models/website_detection/ml_baselines_v2/lightgbm.joblib"
)

DEFAULT_LIGHTGBM_FEATURE_MANIFEST = (
    "results/ml_baselines_v2/feature_manifest.json"
)

DEFAULT_VISUAL_CHECKPOINT = (
    "models/website_detection/visual_cnn_v2/resnet18_v2.pt"
)

DEFAULT_GRAPH_CHECKPOINT = (
    "models/website_detection/gnn_v2/graphsage_v2.pt"
)

DEFAULT_FUSION_MANIFEST = (
    "results/multimodal_fusion_v2/fusion_manifest.json"
)

DEFAULT_OUTPUT_DIR = (
    "results/final_external_evaluation_2026"
)

DECISION_THRESHOLD = 0.50


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def torch_load_checkpoint(
    path: Path,
    map_location: torch.device,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def target_from_source_label(
    series: pd.Series,
) -> np.ndarray:
    labels = pd.to_numeric(
        series,
        errors="raise",
    ).astype(int)

    if not labels.isin(
        [0, 1]
    ).all():
        raise ValueError(
            "source_label must contain only 0/1."
        )

    # Collector:
    #   0 = phishing
    #   1 = legitimate
    #
    # Frozen model target:
    #   1 = phishing
    #   0 = legitimate
    return labels.eq(0).astype(int).to_numpy()


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
        "Yes": 1,
        "No": 0,
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


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    if len(y_true) == 0:
        return float("nan")

    ece = 0.0

    for index in range(
        bins
    ):
        lower = edges[index]
        upper = edges[index + 1]

        if index == bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        mean_probability = float(
            probabilities[
                mask
            ].mean()
        )

        observed_rate = float(
            y_true[
                mask
            ].mean()
        )

        ece += (
            count
            / len(y_true)
        ) * abs(
            mean_probability
            - observed_rate
        )

    return float(
        ece
    )


def safe_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, Any]:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[
            0,
            1,
        ],
    )

    two_classes = (
        len(
            np.unique(
                y_true
            )
        )
        == 2
    )

    return {
        "rows":
            int(
                len(
                    y_true
                )
            ),

        "phishing_rows":
            int(
                (
                    y_true == 1
                ).sum()
            ),

        "legitimate_rows":
            int(
                (
                    y_true == 0
                ).sum()
            ),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),

        "balanced_accuracy":
            (
                float(
                    balanced_accuracy_score(
                        y_true,
                        predictions,
                    )
                )
                if two_classes
                else float("nan")
            ),

        "precision_phishing":
            float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),

        "recall_phishing":
            float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),

        "f1_phishing":
            float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),

        "roc_auc":
            (
                float(
                    roc_auc_score(
                        y_true,
                        probabilities,
                    )
                )
                if two_classes
                else float("nan")
            ),

        "pr_auc":
            (
                float(
                    average_precision_score(
                        y_true,
                        probabilities,
                    )
                )
                if two_classes
                else float("nan")
            ),

        "brier_score":
            float(
                brier_score_loss(
                    y_true,
                    probabilities,
                )
            ),

        "log_loss":
            float(
                log_loss(
                    y_true,
                    probabilities,
                    labels=[
                        0,
                        1,
                    ],
                )
            ),

        "ece_10":
            expected_calibration_error(
                y_true,
                probabilities,
                bins=10,
            ),

        "tn":
            int(
                matrix[
                    0,
                    0
                ]
            ),

        "fp":
            int(
                matrix[
                    0,
                    1
                ]
            ),

        "fn":
            int(
                matrix[
                    1,
                    0
                ]
            ),

        "tp":
            int(
                matrix[
                    1,
                    1
                ]
            ),
    }


# ============================================================
# DATASET VALIDATION
# ============================================================

def load_external_dataset(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"External dataset not found: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "source_url",
        "source_label",
        "source_dataset",
        "visual_screenshot_path",
        "graph_artifact_path",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            f"{path} missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if (
        df[
            "source_url"
        ].astype(str).duplicated().any()
    ):
        raise ValueError(
            f"{path} contains duplicate source_url values."
        )

    df = df.copy()

    df[
        "source_url"
    ] = (
        df[
            "source_url"
        ]
        .astype(str)
    )

    derived_target = target_from_source_label(
        df[
            "source_label"
        ]
    )

    if "actual_phishing" in df.columns:
        existing_target = pd.to_numeric(
            df[
                "actual_phishing"
            ],
            errors="raise",
        ).astype(int).to_numpy()

        if not np.array_equal(
            existing_target,
            derived_target,
        ):
            raise RuntimeError(
                "actual_phishing does not match source_label mapping."
            )

    df[
        "actual_phishing"
    ] = derived_target

    return df


def assert_artifacts_exist(
    df: pd.DataFrame,
) -> None:
    missing_screenshots = df[
        "visual_screenshot_path"
    ].astype(str).map(
        lambda value:
            not Path(
                value
            ).exists()
    )

    missing_graphs = df[
        "graph_artifact_path"
    ].astype(str).map(
        lambda value:
            not Path(
                value
            ).exists()
    )

    if missing_screenshots.any():
        examples = df.loc[
            missing_screenshots,
            [
                "source_url",
                "visual_screenshot_path",
            ],
        ].head(
            10
        )

        raise FileNotFoundError(
            "Final external dataset contains missing screenshots.\n"
            + examples.to_string(
                index=False
            )
        )

    if missing_graphs.any():
        examples = df.loc[
            missing_graphs,
            [
                "source_url",
                "graph_artifact_path",
            ],
        ].head(
            10
        )

        raise FileNotFoundError(
            "Final external dataset contains missing graph JSON files.\n"
            + examples.to_string(
                index=False
            )
        )


# ============================================================
# FROZEN LIGHTGBM
# ============================================================

def frozen_tabular_feature_names(
    model,
    feature_manifest_path: Path,
) -> list[str]:
    candidates = []

    if hasattr(
        model,
        "feature_names_in_",
    ):
        candidates = list(
            model.feature_names_in_
        )

    if (
        not candidates
        and hasattr(
            model,
            "named_steps",
        )
    ):
        for step_name in [
            "imputer",
            "prep",
            "model",
        ]:
            step = model.named_steps.get(
                step_name
            )

            if (
                step is not None
                and hasattr(
                    step,
                    "feature_names_in_",
                )
            ):
                candidates = list(
                    step.feature_names_in_
                )
                break

    manifest_features = []

    if feature_manifest_path.exists():
        manifest = load_json(
            feature_manifest_path
        )

        manifest_features = list(
            manifest.get(
                "features",
                [],
            )
        )

    if (
        candidates
        and manifest_features
        and candidates
        != manifest_features
    ):
        raise RuntimeError(
            "Saved LightGBM feature names do not match "
            "the saved feature manifest."
        )

    features = (
        candidates
        or manifest_features
    )

    if not features:
        raise RuntimeError(
            "Unable to determine frozen LightGBM feature list. "
            "Expected model.feature_names_in_ or "
            f"{feature_manifest_path}."
        )

    return [
        str(
            name
        )
        for name
        in features
    ]


def build_frozen_tabular_matrix(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    values = {}

    for column in features:
        if column in df.columns:
            values[
                column
            ] = coerce_numeric(
                df[
                    column
                ]
            )
        else:
            values[
                column
            ] = pd.Series(
                np.nan,
                index=df.index,
            )

    return pd.DataFrame(
        values,
        index=df.index,
    )


def predict_lightgbm(
    df: pd.DataFrame,
    model_path: Path,
    feature_manifest_path: Path,
) -> pd.DataFrame:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Frozen LightGBM model not found: {model_path}"
        )

    model = joblib.load(
        model_path
    )

    features = frozen_tabular_feature_names(
        model,
        feature_manifest_path,
    )

    matrix = build_frozen_tabular_matrix(
        df,
        features,
    )

    probability = model.predict_proba(
        matrix
    )[
        :,
        1
    ]

    return pd.DataFrame(
        {
            "source_url":
                df[
                    "source_url"
                ].astype(str).values,

            "actual_phishing":
                df[
                    "actual_phishing"
                ].astype(int).values,

            "p_tabular":
                probability.astype(
                    float
                ),
        }
    )


# ============================================================
# FROZEN RESNET18
# ============================================================

class ExternalScreenshotDataset(
    TorchDataset
):
    def __init__(
        self,
        df: pd.DataFrame,
        transform,
    ) -> None:
        self.df = (
            df.reset_index(
                drop=True
            )
        )

        self.transform = transform

    def __len__(
        self,
    ) -> int:
        return len(
            self.df
        )

    def __getitem__(
        self,
        index: int,
    ):
        row = self.df.iloc[
            index
        ]

        image_path = Path(
            str(
                row[
                    "visual_screenshot_path"
                ]
            )
        )

        with Image.open(
            image_path
        ) as image:
            image = image.convert(
                "RGB"
            )

            image = self.transform(
                image
            )

        return (
            image,
            int(
                row[
                    "actual_phishing"
                ]
            ),
            str(
                row[
                    "source_url"
                ]
            ),
        )


def build_frozen_resnet18(
    checkpoint_path: Path,
    device: torch.device,
):
    checkpoint = torch_load_checkpoint(
        checkpoint_path,
        device,
    )

    # IMPORTANT:
    # Build architecture without downloading ImageNet weights.
    # The checkpoint contains the trained weights.
    model = resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        2,
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

    # Exact preprocessing used during original training/evaluation.
    transform = (
        ResNet18_Weights
        .DEFAULT
        .transforms()
    )

    return (
        model,
        transform,
        checkpoint,
    )


def predict_resnet18(
    df: pd.DataFrame,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    (
        model,
        transform,
        _,
    ) = build_frozen_resnet18(
        checkpoint_path,
        device,
    )

    dataset = ExternalScreenshotDataset(
        df,
        transform,
    )

    loader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    urls = []
    actual = []
    probabilities = []

    with torch.no_grad():
        for (
            images,
            labels,
            batch_urls,
        ) in loader:
            images = images.to(
                device
            )

            logits = model(
                images
            )

            probability = torch.softmax(
                logits,
                dim=1,
            )[
                :,
                1
            ]

            urls.extend(
                list(
                    batch_urls
                )
            )

            actual.extend(
                labels.numpy().tolist()
            )

            probabilities.extend(
                probability
                .cpu()
                .numpy()
                .tolist()
            )

    return pd.DataFrame(
        {
            "source_url":
                urls,

            "actual_phishing":
                np.asarray(
                    actual,
                    dtype=int,
                ),

            "p_visual":
                np.asarray(
                    probabilities,
                    dtype=float,
                ),
        }
    )


# ============================================================
# FROZEN GRAPHSAGE
# ============================================================

def predict_graphsage(
    full_csv_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    dataset = WebsiteGraphDataset(
        full_csv_path
    )

    if dataset.skipped:
        skipped_preview = pd.DataFrame(
            dataset.skipped
        ).head(
            10
        )

        raise RuntimeError(
            "GraphSAGE loader skipped external graphs. "
            "Final external evaluation requires complete alignment.\n"
            + skipped_preview.to_string(
                index=False
            )
        )

    checkpoint = torch_load_checkpoint(
        checkpoint_path,
        device,
    )

    architecture = str(
        checkpoint.get(
            "architecture",
            "graphsage",
        )
    )

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

    if (
        dataset.feature_dim
        != input_dim
    ):
        raise RuntimeError(
            "External graph node-feature dimension does not match "
            f"the frozen checkpoint: external={dataset.feature_dim}, "
            f"checkpoint={input_dim}."
        )

    model = GraphClassifier(
        architecture=architecture,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    loader = GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    rows = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(
                device
            )

            logits = model(
                batch.x,
                batch.edge_index,
                batch.batch,
            )

            probability = torch.softmax(
                logits,
                dim=1,
            )[
                :,
                1
            ].cpu().numpy()

            metadata_ids = (
                batch.metadata_id
                .cpu()
                .numpy()
                .tolist()
            )

            actual = (
                batch.y
                .cpu()
                .numpy()
                .tolist()
            )

            for (
                metadata_id,
                target_value,
                probability_value,
            ) in zip(
                metadata_ids,
                actual,
                probability,
            ):
                metadata = dataset.metadata[
                    int(
                        metadata_id
                    )
                ]

                rows.append(
                    {
                        "source_url":
                            metadata[
                                "source_url"
                            ],

                        "actual_phishing":
                            int(
                                target_value
                            ),

                        "p_graph":
                            float(
                                probability_value
                            ),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# ALIGNMENT / FUSION
# ============================================================

def merge_branch_predictions(
    base_df: pd.DataFrame,
    tabular: pd.DataFrame,
    visual: pd.DataFrame,
    graph: pd.DataFrame,
) -> pd.DataFrame:
    for name, frame in [
        (
            "tabular",
            tabular,
        ),
        (
            "visual",
            visual,
        ),
        (
            "graph",
            graph,
        ),
    ]:
        if (
            frame[
                "source_url"
            ].duplicated().any()
        ):
            raise RuntimeError(
                f"{name} predictions contain duplicate source_url."
            )

    merged = (
        base_df[
            [
                "source_url",
                "source_dataset",
                "class_name",
                "actual_phishing",
            ]
        ]
        .merge(
            tabular[
                [
                    "source_url",
                    "actual_phishing",
                    "p_tabular",
                ]
            ],
            on="source_url",
            how="inner",
            suffixes=(
                "_base",
                "_tabular",
            ),
            validate="one_to_one",
        )
        .merge(
            visual[
                [
                    "source_url",
                    "actual_phishing",
                    "p_visual",
                ]
            ],
            on="source_url",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            graph[
                [
                    "source_url",
                    "actual_phishing",
                    "p_graph",
                ]
            ],
            on="source_url",
            how="inner",
            validate="one_to_one",
        )
    )

    target_columns = [
        column
        for column
        in merged.columns
        if column.startswith(
            "actual_phishing"
        )
    ]

    reference = (
        merged[
            target_columns[
                0
            ]
        ]
        .astype(int)
        .to_numpy()
    )

    for column in target_columns[
        1:
    ]:
        candidate = (
            merged[
                column
            ]
            .astype(int)
            .to_numpy()
        )

        if not np.array_equal(
            reference,
            candidate,
        ):
            raise RuntimeError(
                "Target mismatch across external branch predictions."
            )

    if (
        len(
            merged
        )
        != len(
            base_df
        )
    ):
        raise RuntimeError(
            "Not all final external rows aligned across modalities: "
            f"{len(merged):,} of {len(base_df):,}."
        )

    merged[
        "actual_phishing"
    ] = reference

    return merged[
        [
            "source_url",
            "source_dataset",
            "class_name",
            "actual_phishing",
            "p_tabular",
            "p_visual",
            "p_graph",
        ]
    ].copy()


def load_frozen_fusion_weights(
    path: Path,
) -> dict[str, float]:
    manifest = load_json(
        path
    )

    weights = manifest.get(
        "validation_selected_multimodal_weights"
    )

    if not isinstance(
        weights,
        dict,
    ):
        raise RuntimeError(
            "Fusion manifest has no "
            "validation_selected_multimodal_weights."
        )

    required = {
        "tabular",
        "visual",
        "graph",
    }

    if (
        required
        - set(
            weights
        )
    ):
        raise RuntimeError(
            "Fusion manifest has incomplete multimodal weights."
        )

    normalized = {
        key:
            float(
                weights[
                    key
                ]
            )
        for key
        in required
    }

    if abs(
        sum(
            normalized.values()
        )
        - 1.0
    ) > 1e-9:
        raise RuntimeError(
            "Frozen fusion weights do not sum to 1.0."
        )

    return normalized


def add_fusion_and_review_fields(
    df: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    out = df.copy()

    out[
        "p_fusion"
    ] = (
        weights[
            "tabular"
        ]
        * out[
            "p_tabular"
        ]
        +
        weights[
            "visual"
        ]
        * out[
            "p_visual"
        ]
        +
        weights[
            "graph"
        ]
        * out[
            "p_graph"
        ]
    )

    out[
        "tabular_prediction"
    ] = (
        out[
            "p_tabular"
        ]
        >= DECISION_THRESHOLD
    ).astype(int)

    out[
        "visual_prediction"
    ] = (
        out[
            "p_visual"
        ]
        >= DECISION_THRESHOLD
    ).astype(int)

    out[
        "graph_prediction"
    ] = (
        out[
            "p_graph"
        ]
        >= DECISION_THRESHOLD
    ).astype(int)

    out[
        "fusion_prediction"
    ] = (
        out[
            "p_fusion"
        ]
        >= DECISION_THRESHOLD
    ).astype(int)

    out[
        "branch_phishing_votes"
    ] = (
        out[
            "tabular_prediction"
        ]
        + out[
            "visual_prediction"
        ]
        + out[
            "graph_prediction"
        ]
    )

    out[
        "mixed_evidence"
    ] = (
        out[
            "branch_phishing_votes"
        ].isin(
            [
                1,
                2,
            ]
        )
    )

    out[
        "review_required"
    ] = out[
        "mixed_evidence"
    ]

    out[
        "majority_branch_prediction"
    ] = (
        out[
            "branch_phishing_votes"
        ]
        >= 2
    ).astype(int)

    out[
        "majority_disagrees_with_fusion"
    ] = (
        out[
            "majority_branch_prediction"
        ]
        != out[
            "fusion_prediction"
        ]
    )

    out[
        "probability_spread"
    ] = (
        out[
            [
                "p_tabular",
                "p_visual",
                "p_graph",
            ]
        ].max(
            axis=1
        )
        -
        out[
            [
                "p_tabular",
                "p_visual",
                "p_graph",
            ]
        ].min(
            axis=1
        )
    )

    out[
        "fusion_correct"
    ] = (
        out[
            "fusion_prediction"
        ]
        == out[
            "actual_phishing"
        ]
    )

    return out


# ============================================================
# REPORT BUILDERS
# ============================================================

def evaluate_models_for_frame(
    frame: pd.DataFrame,
    evaluation_set: str,
) -> list[dict]:
    rows = []

    for (
        model_name,
        probability_column,
    ) in [
        (
            "LightGBM",
            "p_tabular",
        ),
        (
            "ResNet18",
            "p_visual",
        ),
        (
            "GraphSAGE",
            "p_graph",
        ),
        (
            "Fusion_80_10_10",
            "p_fusion",
        ),
    ]:
        rows.append(
            {
                "evaluation_set":
                    evaluation_set,

                "model":
                    model_name,

                **safe_metrics(
                    frame[
                        "actual_phishing"
                    ].to_numpy(),

                    frame[
                        probability_column
                    ].to_numpy(),
                ),
            }
        )

    return rows


def evaluate_by_source(
    frame: pd.DataFrame,
    evaluation_set: str,
) -> list[dict]:
    rows = []

    for (
        source_name,
        source_frame,
    ) in frame.groupby(
        "source_dataset",
        dropna=False,
    ):
        for (
            model_name,
            probability_column,
        ) in [
            (
                "LightGBM",
                "p_tabular",
            ),
            (
                "ResNet18",
                "p_visual",
            ),
            (
                "GraphSAGE",
                "p_graph",
            ),
            (
                "Fusion_80_10_10",
                "p_fusion",
            ),
        ]:
            rows.append(
                {
                    "evaluation_set":
                        evaluation_set,

                    "source_dataset":
                        source_name,

                    "model":
                        model_name,

                    **safe_metrics(
                        source_frame[
                            "actual_phishing"
                        ].to_numpy(),

                        source_frame[
                            probability_column
                        ].to_numpy(),
                    ),
                }
            )

    return rows


def review_summary(
    frame: pd.DataFrame,
    evaluation_set: str,
) -> dict:
    errors = ~frame[
        "fusion_correct"
    ]

    flagged_errors = (
        errors
        & frame[
            "review_required"
        ]
    )

    total_errors = int(
        errors.sum()
    )

    return {
        "evaluation_set":
            evaluation_set,

        "rows":
            int(
                len(
                    frame
                )
            ),

        "review_required_count":
            int(
                frame[
                    "review_required"
                ].sum()
            ),

        "review_required_rate":
            float(
                frame[
                    "review_required"
                ].mean()
            ),

        "majority_disagrees_with_fusion_count":
            int(
                frame[
                    "majority_disagrees_with_fusion"
                ].sum()
            ),

        "fusion_errors":
            total_errors,

        "fusion_errors_flagged_for_review":
            int(
                flagged_errors.sum()
            ),

        "fusion_error_review_capture_rate":
            (
                float(
                    flagged_errors.sum()
                    / total_errors
                )
                if total_errors > 0
                else float("nan")
            ),

        "fusion_false_positives":
            int(
                (
                    (
                        frame[
                            "actual_phishing"
                        ]
                        == 0
                    )
                    &
                    (
                        frame[
                            "fusion_prediction"
                        ]
                        == 1
                    )
                ).sum()
            ),

        "fusion_false_negatives":
            int(
                (
                    (
                        frame[
                            "actual_phishing"
                        ]
                        == 1
                    )
                    &
                    (
                        frame[
                            "fusion_prediction"
                        ]
                        == 0
                    )
                ).sum()
            ),
    }


def subset_by_urls(
    full_predictions: pd.DataFrame,
    subset_df: pd.DataFrame,
) -> pd.DataFrame:
    selected = (
        subset_df[
            [
                "source_url",
            ]
        ]
        .merge(
            full_predictions,
            on="source_url",
            how="left",
            validate="one_to_one",
        )
    )

    probability_columns = [
        "p_tabular",
        "p_visual",
        "p_graph",
        "p_fusion",
    ]

    if selected[
        probability_columns
    ].isna().any().any():
        raise RuntimeError(
            "Balanced dataset contains rows without full-set predictions."
        )

    return selected


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen Dataset V2 LightGBM, ResNet18, "
            "GraphSAGE and validation-selected 80/10/10 late fusion "
            "on the fresh final external dataset. "
            "This script performs NO fitting or tuning."
        )
    )

    parser.add_argument(
        "--full",
        default=DEFAULT_EXTERNAL_FULL,
    )

    parser.add_argument(
        "--balanced",
        default=DEFAULT_EXTERNAL_BALANCED,
    )

    parser.add_argument(
        "--lightgbm-model",
        default=DEFAULT_LIGHTGBM_MODEL,
    )

    parser.add_argument(
        "--lightgbm-feature-manifest",
        default=DEFAULT_LIGHTGBM_FEATURE_MANIFEST,
    )

    parser.add_argument(
        "--visual-checkpoint",
        default=DEFAULT_VISUAL_CHECKPOINT,
    )

    parser.add_argument(
        "--graph-checkpoint",
        default=DEFAULT_GRAPH_CHECKPOINT,
    )

    parser.add_argument(
        "--fusion-manifest",
        default=DEFAULT_FUSION_MANIFEST,
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--visual-batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--graph-batch-size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    full_path = Path(
        args.full
    )

    balanced_path = Path(
        args.balanced
    )

    lightgbm_model_path = Path(
        args.lightgbm_model
    )

    feature_manifest_path = Path(
        args.lightgbm_feature_manifest
    )

    visual_checkpoint_path = Path(
        args.visual_checkpoint
    )

    graph_checkpoint_path = Path(
        args.graph_checkpoint
    )

    fusion_manifest_path = Path(
        args.fusion_manifest
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 100)
    print("FINAL FRESH EXTERNAL EVALUATION - FROZEN MODELS ONLY")
    print("=" * 100)
    print("Training / fitting operations: NONE")
    print("Threshold tuning: NONE")
    print("Fusion-weight search: NONE")
    print("Calibration fitting: NONE")
    print("=" * 100)

    full_df = load_external_dataset(
        full_path
    )

    balanced_df = load_external_dataset(
        balanced_path
    )

    print(
        f"Full external rows: "
        f"{len(full_df):,}"
    )

    print(
        f"Balanced external rows: "
        f"{len(balanced_df):,}"
    )

    print()
    print(
        "Verifying screenshot and graph artifacts..."
    )

    assert_artifacts_exist(
        full_df
    )

    print(
        "Artifact verification: PASSED"
    )

    device = get_device()

    print(
        f"Device: {device}"
    )

    weights = load_frozen_fusion_weights(
        fusion_manifest_path
    )

    print(
        "Frozen fusion weights: "
        f"tabular={weights['tabular']:.2f}, "
        f"visual={weights['visual']:.2f}, "
        f"graph={weights['graph']:.2f}"
    )

    print()
    print(
        "[1/3] Frozen LightGBM inference..."
    )

    tabular_predictions = predict_lightgbm(
        full_df,
        lightgbm_model_path,
        feature_manifest_path,
    )

    print(
        f"      Predictions: "
        f"{len(tabular_predictions):,}"
    )

    print()
    print(
        "[2/3] Frozen ResNet18 inference..."
    )

    visual_predictions = predict_resnet18(
        full_df,
        visual_checkpoint_path,
        device,
        args.visual_batch_size,
    )

    print(
        f"      Predictions: "
        f"{len(visual_predictions):,}"
    )

    print()
    print(
        "[3/3] Frozen GraphSAGE inference..."
    )

    graph_predictions = predict_graphsage(
        full_path,
        graph_checkpoint_path,
        device,
        args.graph_batch_size,
    )

    print(
        f"      Predictions: "
        f"{len(graph_predictions):,}"
    )

    full_predictions = merge_branch_predictions(
        full_df,
        tabular_predictions,
        visual_predictions,
        graph_predictions,
    )

    full_predictions = add_fusion_and_review_fields(
        full_predictions,
        weights,
    )

    balanced_predictions = subset_by_urls(
        full_predictions,
        balanced_df,
    )

    metric_rows = []

    metric_rows.extend(
        evaluate_models_for_frame(
            full_predictions,
            "full_external",
        )
    )

    metric_rows.extend(
        evaluate_models_for_frame(
            balanced_predictions,
            "balanced_external",
        )
    )

    metrics = pd.DataFrame(
        metric_rows
    )

    source_metric_rows = []

    source_metric_rows.extend(
        evaluate_by_source(
            full_predictions,
            "full_external",
        )
    )

    source_metric_rows.extend(
        evaluate_by_source(
            balanced_predictions,
            "balanced_external",
        )
    )

    source_metrics = pd.DataFrame(
        source_metric_rows
    )

    review = pd.DataFrame(
        [
            review_summary(
                full_predictions,
                "full_external",
            ),
            review_summary(
                balanced_predictions,
                "balanced_external",
            ),
        ]
    )

    full_errors = full_predictions[
        ~full_predictions[
            "fusion_correct"
        ]
    ].copy()

    balanced_errors = balanced_predictions[
        ~balanced_predictions[
            "fusion_correct"
        ]
    ].copy()

    # Save outputs.
    full_predictions.to_csv(
        output_dir
        / "final_external_full_predictions.csv",
        index=False,
    )

    balanced_predictions.to_csv(
        output_dir
        / "final_external_balanced_predictions.csv",
        index=False,
    )

    metrics.to_csv(
        output_dir
        / "final_external_metrics.csv",
        index=False,
    )

    source_metrics.to_csv(
        output_dir
        / "final_external_source_metrics.csv",
        index=False,
    )

    review.to_csv(
        output_dir
        / "final_external_review_summary.csv",
        index=False,
    )

    full_errors.to_csv(
        output_dir
        / "final_external_full_fusion_errors.csv",
        index=False,
    )

    balanced_errors.to_csv(
        output_dir
        / "final_external_balanced_fusion_errors.csv",
        index=False,
    )

    manifest = {
        "purpose":
            "untouched_final_external_evaluation",

        "training_performed":
            False,

        "tuning_performed":
            False,

        "threshold":
            DECISION_THRESHOLD,

        "probability_calibration_used":
            False,

        "fusion_weights":
            weights,

        "models": {
            "tabular":
                str(
                    lightgbm_model_path
                ),

            "visual":
                str(
                    visual_checkpoint_path
                ),

            "graph":
                str(
                    graph_checkpoint_path
                ),
        },

        "datasets": {
            "full":
                str(
                    full_path
                ),

            "balanced":
                str(
                    balanced_path
                ),
        },

        "row_counts": {
            "full":
                len(
                    full_predictions
                ),

            "balanced":
                len(
                    balanced_predictions
                ),
        },

        "review_rule":
            (
                "review_required=True when the three independent "
                "branches do not all agree at threshold 0.50. "
                "This flag does not change the frozen fusion prediction."
            ),

        "methodology_note":
            (
                "All model files and fusion weights are loaded from "
                "the frozen Dataset V2 development pipeline. "
                "This evaluator never calls fit(), searches weights, "
                "changes threshold, or calibrates on external labels."
            ),
    }

    with (
        output_dir
        / "final_external_evaluation_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    display_columns = [
        "evaluation_set",
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

    print()
    print("=" * 100)
    print("FINAL EXTERNAL METRICS")
    print("=" * 100)

    print(
        metrics[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("REVIEW / DISAGREEMENT SUMMARY")
    print("=" * 100)

    print(
        review.to_string(
            index=False
        )
    )

    print()
    print(
        "NOTE: Per-source ROC-AUC/PR-AUC are left NaN for "
        "single-class source subsets."
    )

    print(
        f"Outputs: {output_dir}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
