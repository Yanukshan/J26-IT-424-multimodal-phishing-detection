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
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision.models import ResNet18_Weights, resnet18

from src.website_detection.crawler.crawler import crawl_website
from src.website_detection.dataset.collector import flatten_result
from src.website_detection.models.gnn_v2_trainer import (
    GraphClassifier,
    get_device,
    graph_json_to_data,
)


DEFAULT_LIGHTGBM_MODEL = Path(
    "models/website_detection/ml_baselines_v2/lightgbm.joblib"
)

DEFAULT_LIGHTGBM_FEATURE_MANIFEST = Path(
    "results/ml_baselines_v2/feature_manifest.json"
)

DEFAULT_VISUAL_CHECKPOINT = Path(
    "models/website_detection/visual_cnn_v2/resnet18_v2.pt"
)

DEFAULT_GRAPH_CHECKPOINT = Path(
    "models/website_detection/gnn_v2/graphsage_v2.pt"
)

DEFAULT_FUSION_MANIFEST = Path(
    "results/multimodal_fusion_v2/fusion_manifest.json"
)

DECISION_THRESHOLD = 0.50


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def torch_load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def as_bool(value: Any) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return value

    if value is None:
        return False

    if isinstance(
        value,
        (int, float),
    ):
        return bool(
            value
        )

    return (
        str(
            value
        )
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
        }
    )


def as_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        if value is None:
            return default

        if (
            isinstance(
                value,
                str,
            )
            and not value.strip()
        ):
            return default

        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return default


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


def label_from_vote(
    vote: int,
) -> str:
    return (
        "PHISHING"
        if int(
            vote
        )
        == 1
        else "LEGITIMATE"
    )


def risk_level(
    risk_score: int,
) -> str:
    if risk_score < 20:
        return "LOW"

    if risk_score < 40:
        return "MODERATE_LOW"

    if risk_score < 60:
        return "UNCERTAIN"

    if risk_score < 80:
        return "HIGH"

    return "VERY_HIGH"


class FinalWebsiteDetector:
    """
    Frozen single-URL website phishing detector.

    Important methodology rule:
    - this class performs inference only;
    - it does not fit or retrain any model;
    - it does not search fusion weights;
    - it does not tune the 0.50 decision threshold;
    - it does not treat a crawl/network failure as phishing evidence.
    """

    def __init__(
        self,
        lightgbm_model_path: Path = DEFAULT_LIGHTGBM_MODEL,
        lightgbm_feature_manifest_path: Path = (
            DEFAULT_LIGHTGBM_FEATURE_MANIFEST
        ),
        visual_checkpoint_path: Path = DEFAULT_VISUAL_CHECKPOINT,
        graph_checkpoint_path: Path = DEFAULT_GRAPH_CHECKPOINT,
        fusion_manifest_path: Path = DEFAULT_FUSION_MANIFEST,
    ) -> None:
        self.device = get_device()

        self.lightgbm_model_path = Path(
            lightgbm_model_path
        )

        self.lightgbm_feature_manifest_path = Path(
            lightgbm_feature_manifest_path
        )

        self.visual_checkpoint_path = Path(
            visual_checkpoint_path
        )

        self.graph_checkpoint_path = Path(
            graph_checkpoint_path
        )

        self.fusion_manifest_path = Path(
            fusion_manifest_path
        )

        self._load_tabular_branch()
        self._load_visual_branch()
        self._load_graph_branch()
        self._load_fusion_weights()

    # ========================================================
    # MODEL LOADING
    # ========================================================

    def _load_tabular_branch(
        self,
    ) -> None:
        if not self.lightgbm_model_path.exists():
            raise FileNotFoundError(
                "Frozen LightGBM model not found: "
                f"{self.lightgbm_model_path}"
            )

        self.tabular_model = joblib.load(
            self.lightgbm_model_path
        )

        manifest_features: list[str] = []

        if (
            self.lightgbm_feature_manifest_path.exists()
        ):
            manifest = load_json(
                self.lightgbm_feature_manifest_path
            )

            manifest_features = [
                str(
                    feature
                )
                for feature
                in manifest.get(
                    "features",
                    [],
                )
            ]

        model_features: list[str] = []

        if hasattr(
            self.tabular_model,
            "feature_names_in_",
        ):
            model_features = [
                str(
                    feature
                )
                for feature
                in self.tabular_model.feature_names_in_
            ]

        if (
            not model_features
            and hasattr(
                self.tabular_model,
                "named_steps",
            )
        ):
            for step_name in [
                "imputer",
                "prep",
                "model",
            ]:
                step = (
                    self.tabular_model
                    .named_steps
                    .get(
                        step_name
                    )
                )

                if (
                    step is not None
                    and hasattr(
                        step,
                        "feature_names_in_",
                    )
                ):
                    model_features = [
                        str(
                            feature
                        )
                        for feature
                        in step.feature_names_in_
                    ]

                    break

        if (
            model_features
            and manifest_features
            and model_features
            != manifest_features
        ):
            raise RuntimeError(
                "LightGBM model features differ from the "
                "saved feature manifest."
            )

        self.tabular_features = (
            model_features
            or manifest_features
        )

        if not self.tabular_features:
            raise RuntimeError(
                "Unable to determine the frozen LightGBM "
                "feature list."
            )

    def _load_visual_branch(
        self,
    ) -> None:
        checkpoint = torch_load_checkpoint(
            self.visual_checkpoint_path,
            self.device,
        )

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
            self.device
        )

        model.eval()

        self.visual_model = model

        # Exact preprocessing used by the original ResNet18 trainer.
        self.visual_transform = (
            ResNet18_Weights
            .DEFAULT
            .transforms()
        )

    def _load_graph_branch(
        self,
    ) -> None:
        checkpoint = torch_load_checkpoint(
            self.graph_checkpoint_path,
            self.device,
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

        model = GraphClassifier(
            architecture=architecture,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(
            self.device
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        model.eval()

        self.graph_model = model
        self.graph_input_dim = input_dim
        self.graph_architecture = architecture

    def _load_fusion_weights(
        self,
    ) -> None:
        manifest = load_json(
            self.fusion_manifest_path
        )

        weights = manifest.get(
            "validation_selected_multimodal_weights"
        )

        if not isinstance(
            weights,
            dict,
        ):
            raise RuntimeError(
                "Fusion manifest does not contain "
                "validation_selected_multimodal_weights."
            )

        self.fusion_weights = {
            "tabular":
                float(
                    weights[
                        "tabular"
                    ]
                ),

            "visual":
                float(
                    weights[
                        "visual"
                    ]
                ),

            "graph":
                float(
                    weights[
                        "graph"
                    ]
                ),
        }

        if abs(
            sum(
                self.fusion_weights.values()
            )
            - 1.0
        ) > 1e-9:
            raise RuntimeError(
                "Frozen fusion weights do not sum to 1.0."
            )

    # ========================================================
    # BRANCH INFERENCE
    # ========================================================

    def _predict_tabular(
        self,
        row: dict[str, Any],
    ) -> float:
        dataframe = pd.DataFrame(
            [
                row
            ]
        )

        matrix_data = {}

        for feature in self.tabular_features:
            if feature in dataframe.columns:
                matrix_data[
                    feature
                ] = coerce_numeric(
                    dataframe[
                        feature
                    ]
                )
            else:
                matrix_data[
                    feature
                ] = pd.Series(
                    [
                        np.nan
                    ]
                )

        matrix = pd.DataFrame(
            matrix_data,
            columns=self.tabular_features,
        )

        probability = (
            self.tabular_model
            .predict_proba(
                matrix
            )[
                0,
                1
            ]
        )

        return float(
            probability
        )

    def _predict_visual(
        self,
        screenshot_path: Path,
    ) -> float:
        if not screenshot_path.exists():
            raise FileNotFoundError(
                "Screenshot artifact not found: "
                f"{screenshot_path}"
            )

        with Image.open(
            screenshot_path
        ) as image:
            image = image.convert(
                "RGB"
            )

            tensor = self.visual_transform(
                image
            ).unsqueeze(
                0
            )

        tensor = tensor.to(
            self.device
        )

        with torch.no_grad():
            logits = self.visual_model(
                tensor
            )

            probability = torch.softmax(
                logits,
                dim=1,
            )[
                0,
                1
            ]

        return float(
            probability.item()
        )

    def _predict_graph(
        self,
        graph_path: Path,
    ) -> float:
        if not graph_path.exists():
            raise FileNotFoundError(
                "Graph artifact not found: "
                f"{graph_path}"
            )

        # Target is unused during live inference.
        graph = graph_json_to_data(
            graph_path=graph_path,
            target=0,
            sample_index=0,
        )

        if (
            int(
                graph.x.shape[
                    1
                ]
            )
            != self.graph_input_dim
        ):
            raise RuntimeError(
                "Live graph feature dimension does not match "
                "the frozen GraphSAGE checkpoint: "
                f"live={int(graph.x.shape[1])}, "
                f"checkpoint={self.graph_input_dim}."
            )

        loader = GraphDataLoader(
            [
                graph
            ],
            batch_size=1,
            shuffle=False,
        )

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(
                    self.device
                )

                logits = self.graph_model(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                )

                probability = torch.softmax(
                    logits,
                    dim=1,
                )[
                    0,
                    1
                ]

                return float(
                    probability.item()
                )

        raise RuntimeError(
            "GraphSAGE inference produced no result."
        )

    # ========================================================
    # EXPLANATION / OUTPUT HELPERS
    # ========================================================

    def _observed_indicators(
        self,
        row: dict[str, Any],
    ) -> list[str]:
        indicators: list[str] = []

        if as_bool(
            row.get(
                "cross_domain_redirect"
            )
        ):
            indicators.append(
                "The page redirected to a different domain."
            )

        if as_float(
            row.get(
                "url_suspicious_token_count"
            )
        ) > 0:
            indicators.append(
                "Suspicious lexical token(s) were observed in the URL."
            )

        if as_bool(
            row.get(
                "cred_login_intent_present"
            )
        ):
            indicators.append(
                "Login or credential-entry intent was detected."
            )

        if as_bool(
            row.get(
                "cred_payment_intent_present"
            )
        ):
            indicators.append(
                "Payment-related credential intent was detected."
            )

        if as_bool(
            row.get(
                "cred_otp_intent_present"
            )
        ):
            indicators.append(
                "OTP-related credential intent was detected."
            )

        if as_float(
            row.get(
                "cred_external_credential_forms"
            )
        ) > 0:
            indicators.append(
                "A credential-oriented form targets an external domain."
            )

        if as_bool(
            row.get(
                "cred_external_sensitive_submission"
            )
        ):
            indicators.append(
                "Sensitive form data appears directed to an external domain."
            )

        if as_bool(
            row.get(
                "cred_insecure_sensitive_submission"
            )
        ):
            indicators.append(
                "A sensitive form submission uses an insecure destination."
            )

        if as_float(
            row.get(
                "dom_insecure_form_actions"
            )
        ) > 0:
            indicators.append(
                "An insecure form action was observed in the rendered DOM."
            )

        external_request_ratio = as_float(
            row.get(
                "net_external_request_ratio"
            )
        )

        if external_request_ratio > 0.50:
            indicators.append(
                "More than half of observed network requests were external."
            )

        external_node_ratio = as_float(
            row.get(
                "graph_external_node_ratio"
            )
        )

        if external_node_ratio > 0.50:
            indicators.append(
                "More than half of resource-graph nodes were external."
            )

        if not indicators:
            indicators.append(
                "No simple rule-based indicator was highlighted; "
                "the model decision is based on the learned feature combination."
            )

        return indicators[
            :8
        ]

    def _build_decision(
        self,
        p_tabular: float,
        p_visual: float,
        p_graph: float,
    ) -> dict[str, Any]:
        weights = self.fusion_weights

        fusion_probability = (
            weights[
                "tabular"
            ]
            * p_tabular
            +
            weights[
                "visual"
            ]
            * p_visual
            +
            weights[
                "graph"
            ]
            * p_graph
        )

        tabular_vote = int(
            p_tabular
            >= DECISION_THRESHOLD
        )

        visual_vote = int(
            p_visual
            >= DECISION_THRESHOLD
        )

        graph_vote = int(
            p_graph
            >= DECISION_THRESHOLD
        )

        votes = [
            tabular_vote,
            visual_vote,
            graph_vote,
        ]

        fusion_vote = int(
            fusion_probability
            >= DECISION_THRESHOLD
        )

        phishing_votes = int(
            sum(
                votes
            )
        )

        mixed_evidence = (
            phishing_votes
            in {
                1,
                2,
            }
        )

        majority_vote = int(
            phishing_votes
            >= 2
        )

        majority_disagrees = (
            majority_vote
            != fusion_vote
        )

        review_required = (
            mixed_evidence
            or majority_disagrees
        )

        spread = (
            max(
                p_tabular,
                p_visual,
                p_graph,
            )
            - min(
                p_tabular,
                p_visual,
                p_graph,
            )
        )

        score = int(
            round(
                fusion_probability
                * 100.0
            )
        )

        evidence_status = (
            "MIXED_EVIDENCE"
            if mixed_evidence
            else "FULL_AGREEMENT"
        )

        operational_status = (
            "REVIEW_REQUIRED_MIXED_EVIDENCE"
            if review_required
            else "AUTOMATED_DECISION"
        )

        return {
            "prediction":
                label_from_vote(
                    fusion_vote
                ),

            "fusion_probability_raw":
                float(
                    fusion_probability
                ),

            "decision_threshold":
                DECISION_THRESHOLD,

            "risk_score":
                score,

            "risk_level":
                risk_level(
                    score
                ),

            "branch_scores": {
                "lightgbm": {
                    "probability":
                        float(
                            p_tabular
                        ),

                    "prediction":
                        label_from_vote(
                            tabular_vote
                        ),
                },

                "resnet18": {
                    "probability":
                        float(
                            p_visual
                        ),

                    "prediction":
                        label_from_vote(
                            visual_vote
                        ),
                },

                "graphsage": {
                    "probability":
                        float(
                            p_graph
                        ),

                    "prediction":
                        label_from_vote(
                            graph_vote
                        ),
                },
            },

            "fusion_weights":
                dict(
                    weights
                ),

            "evidence_status":
                evidence_status,

            "branch_phishing_votes":
                phishing_votes,

            "majority_branch_prediction":
                label_from_vote(
                    majority_vote
                ),

            "majority_disagrees_with_fusion":
                bool(
                    majority_disagrees
                ),

            "probability_spread":
                float(
                    spread
                ),

            "review_required":
                bool(
                    review_required
                ),

            "operational_status":
                operational_status,

            "score_interpretation":
                (
                    "Raw validation-selected weighted model score. "
                    "It is used for ranking/risk communication and "
                    "should not be described as a perfectly calibrated "
                    "real-world phishing probability."
                ),
        }

    # ========================================================
    # PUBLIC API
    # ========================================================

    def analyze(
        self,
        url: str,
    ) -> dict[str, Any]:
        crawl_result = crawl_website(
            url,
            # Keep live inference aligned with the crawler behavior used
            # to build/train the research dataset whenever possible.
            # Only recover from a full-load timeout when Chromium has
            # already rendered a real document.
            navigation_wait_until="load",
            recover_navigation_timeout=True,
            post_navigation_wait_ms=0,
        )

        status = str(
            crawl_result.get(
                "crawl_status",
                "ERROR",
            )
        )

        base_output: dict[str, Any] = {
            "input_url":
                url,

            "requested_url":
                crawl_result.get(
                    "requested_url",
                    url,
                ),

            "final_url":
                crawl_result.get(
                    "final_url",
                    "",
                ),

            "crawl_status":
                status,

            "status_code":
                crawl_result.get(
                    "status_code"
                ),

            "title":
                crawl_result.get(
                    "title",
                    "",
                ),

            "redirected":
                crawl_result.get(
                    "redirected",
                    False,
                ),

            "cross_domain_redirect":
                crawl_result.get(
                    "cross_domain_redirect",
                    False,
                ),

            "navigation_wait_until":
                crawl_result.get(
                    "navigation_wait_until",
                    "",
                ),

            "navigation_timeout_recovered":
                crawl_result.get(
                    "navigation_timeout_recovered",
                    False,
                ),
        }

        # A failed/dead/blocked URL is not itself phishing evidence.
        if status != "SUCCESS":
            return {
                **base_output,

                "analysis_status":
                    "MODEL_INFERENCE_NOT_RUN",

                "prediction":
                    None,

                "risk_score":
                    None,

                "risk_level":
                    None,

                "review_required":
                    True,

                "operational_status":
                    "CRAWL_UNAVAILABLE_REVIEW",

                "reason":
                    (
                        "The website did not produce a successful "
                        "rendered page. No phishing classification was "
                        "made because crawl failure is not treated as "
                        "phishing evidence."
                    ),

                "crawl_error_type":
                    crawl_result.get(
                        "error_type",
                        "",
                    ),

                "crawl_error":
                    crawl_result.get(
                        "error",
                        "",
                    ),
            }

        # Reuse the exact collector flattening convention used when
        # the frozen tabular model's data was created.
        #
        # label=None is intentional for live inference. get_class_name()
        # falls back to "unknown"; the source label is not used by any
        # of the three branch predictors below.
        row = flatten_result(
            sample_id=1,
            source_url=str(
                crawl_result.get(
                    "requested_url",
                    url,
                )
            ),
            label=None,  # type: ignore[arg-type]
            result=crawl_result,
            source_metadata=None,
        )

        screenshot_path = Path(
            str(
                row.get(
                    "visual_screenshot_path",
                    "",
                )
            )
        )

        graph_path = Path(
            str(
                row.get(
                    "graph_artifact_path",
                    "",
                )
            )
        )

        if (
            not as_bool(
                row.get(
                    "visual_screenshot_saved"
                )
            )
            or not screenshot_path.exists()
        ):
            raise RuntimeError(
                "Successful live crawl has no usable screenshot artifact."
            )

        if (
            not as_bool(
                row.get(
                    "graph_artifact_saved"
                )
            )
            or not graph_path.exists()
        ):
            raise RuntimeError(
                "Successful live crawl has no usable graph artifact."
            )

        p_tabular = self._predict_tabular(
            row
        )

        p_visual = self._predict_visual(
            screenshot_path
        )

        p_graph = self._predict_graph(
            graph_path
        )

        decision = self._build_decision(
            p_tabular=p_tabular,
            p_visual=p_visual,
            p_graph=p_graph,
        )

        return {
            **base_output,

            "analysis_status":
                "SUCCESS",

            **decision,

            "observed_indicators":
                self._observed_indicators(
                    row
                ),

            "indicator_note":
                (
                    "Observed indicators are descriptive evidence "
                    "summaries, not causal explanations of the model."
                ),

            "artifacts": {
                "screenshot_path":
                    str(
                        screenshot_path
                    ),

                "graph_path":
                    str(
                        graph_path
                    ),
            },

            "models": {
                "tabular":
                    "LightGBM",

                "visual":
                    "ResNet18",

                "graph":
                    self.graph_architecture,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen final multimodal website phishing "
            "detector on one live URL."
        )
    )

    parser.add_argument(
        "url",
        help="Website URL to analyse.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional JSON output path. "
            "If omitted, result is only printed."
        ),
    )

    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact JSON instead of indented JSON.",
    )

    args = parser.parse_args()

    detector = FinalWebsiteDetector()

    result = detector.analyze(
        args.url
    )

    text = json.dumps(
        result,
        indent=(
            None
            if args.compact
            else 2
        ),
        ensure_ascii=False,
    )

    print(
        text
    )

    if args.output:
        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
