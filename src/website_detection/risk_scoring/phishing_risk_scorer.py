from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_WEIGHTS = {
    "tabular": 0.80,
    "visual": 0.10,
    "graph": 0.10,
}

DECISION_THRESHOLD = 0.50


@dataclass
class RiskResult:
    phishing_prediction: bool
    prediction_label: str
    risk_score: int
    risk_level: str
    fusion_probability: float
    decision_threshold: float
    tabular_probability: float
    visual_probability: float
    graph_probability: float
    tabular_weight: float
    visual_weight: float
    graph_weight: float
    tabular_contribution: float
    visual_contribution: float
    graph_contribution: float
    branch_phishing_votes: int
    branch_legitimate_votes: int
    modality_agreement: str
    majority_branch_label: str
    majority_disagrees_with_fusion: bool
    probability_spread: float
    review_required: bool
    operational_status: str
    score_source: str
    risk_band_note: str


def validate_probability(
    name: str,
    value: float,
) -> float:
    value = float(value)

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1, got {value}."
        )

    return value


def validate_weights(
    weights: dict,
) -> dict:
    required = {
        "tabular",
        "visual",
        "graph",
    }

    missing = required - set(weights)

    if missing:
        raise ValueError(
            "Missing fusion weights: "
            + ", ".join(sorted(missing))
        )

    normalized = {
        key: float(weights[key])
        for key in required
    }

    total = sum(normalized.values())

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Fusion weights must sum to 1.0, got {total}."
        )

    if any(
        value < 0.0
        for value in normalized.values()
    ):
        raise ValueError(
            "Fusion weights cannot be negative."
        )

    return normalized


def load_multimodal_weights(
    fusion_manifest: Path | None,
) -> dict:
    if (
        fusion_manifest is not None
        and fusion_manifest.exists()
    ):
        with fusion_manifest.open(
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(file)

        weights = manifest.get(
            "validation_selected_multimodal_weights"
        )

        if isinstance(weights, dict):
            return validate_weights(weights)

    return validate_weights(DEFAULT_WEIGHTS)


def risk_level_from_score(
    risk_score: int,
) -> str:
    """
    Communication bands only.

    These bands are NOT separately learned decision thresholds.
    The binary phishing decision remains based on 0.50.
    """

    if risk_score < 20:
        return "LOW"

    if risk_score < 40:
        return "MODERATE_LOW"

    if risk_score < 60:
        return "UNCERTAIN"

    if risk_score < 80:
        return "HIGH"

    return "VERY_HIGH"


def calculate_risk(
    tabular_probability: float,
    visual_probability: float,
    graph_probability: float,
    weights: dict,
    decision_threshold: float = DECISION_THRESHOLD,
) -> RiskResult:
    tabular_probability = validate_probability(
        "tabular_probability",
        tabular_probability,
    )

    visual_probability = validate_probability(
        "visual_probability",
        visual_probability,
    )

    graph_probability = validate_probability(
        "graph_probability",
        graph_probability,
    )

    decision_threshold = validate_probability(
        "decision_threshold",
        decision_threshold,
    )

    weights = validate_weights(weights)

    tabular_contribution = (
        weights["tabular"]
        * tabular_probability
    )

    visual_contribution = (
        weights["visual"]
        * visual_probability
    )

    graph_contribution = (
        weights["graph"]
        * graph_probability
    )

    fusion_probability = (
        tabular_contribution
        + visual_contribution
        + graph_contribution
    )

    phishing_prediction = (
        fusion_probability
        >= decision_threshold
    )

    # Raw multimodal fusion is intentionally used.
    #
    # Dataset V2 Platt calibration did not consistently improve
    # calibration quality:
    # - multimodal Brier score, log loss and ECE became worse
    #   on validation and test,
    # - classification performance also dropped on validation.
    #
    # Therefore calibration was evaluated but rejected for the
    # current final development pipeline.
    risk_score = int(
        round(
            100.0
            * fusion_probability
        )
    )

    risk_score = max(
        0,
        min(
            100,
            risk_score,
        ),
    )

    branch_probabilities = [
        tabular_probability,
        visual_probability,
        graph_probability,
    ]

    branch_phishing_votes = sum(
        probability >= decision_threshold
        for probability in branch_probabilities
    )

    branch_legitimate_votes = (
        len(branch_probabilities)
        - branch_phishing_votes
    )

    if (
        branch_phishing_votes == 3
        or branch_legitimate_votes == 3
    ):
        modality_agreement = "FULL_AGREEMENT"
    else:
        modality_agreement = "MIXED_EVIDENCE"

    majority_branch_label = (
        "PHISHING"
        if branch_phishing_votes >= 2
        else "LEGITIMATE"
    )

    majority_disagrees_with_fusion = (
        majority_branch_label
        != (
            "PHISHING"
            if phishing_prediction
            else "LEGITIMATE"
        )
    )

    probability_spread = (
        max(branch_probabilities)
        - min(branch_probabilities)
    )

    # IMPORTANT:
    # This flag does NOT change the trained model decision or threshold.
    # It only prevents a low weighted score from being presented as
    # confidently safe when the independent modalities strongly disagree.
    review_required = bool(
        majority_disagrees_with_fusion
        or modality_agreement == "MIXED_EVIDENCE"
    )

    operational_status = (
        "REVIEW_REQUIRED_MIXED_EVIDENCE"
        if review_required
        else (
            "PHISHING"
            if phishing_prediction
            else "LEGITIMATE"
        )
    )

    return RiskResult(
        phishing_prediction=
            phishing_prediction,

        prediction_label=(
            "PHISHING"
            if phishing_prediction
            else "LEGITIMATE"
        ),

        risk_score=
            risk_score,

        risk_level=
            risk_level_from_score(
                risk_score
            ),

        fusion_probability=
            float(
                fusion_probability
            ),

        decision_threshold=
            float(
                decision_threshold
            ),

        tabular_probability=
            tabular_probability,

        visual_probability=
            visual_probability,

        graph_probability=
            graph_probability,

        tabular_weight=
            weights["tabular"],

        visual_weight=
            weights["visual"],

        graph_weight=
            weights["graph"],

        tabular_contribution=
            float(
                tabular_contribution
            ),

        visual_contribution=
            float(
                visual_contribution
            ),

        graph_contribution=
            float(
                graph_contribution
            ),

        branch_phishing_votes=
            int(
                branch_phishing_votes
            ),

        branch_legitimate_votes=
            int(
                branch_legitimate_votes
            ),

        modality_agreement=
            modality_agreement,

        majority_branch_label=
            majority_branch_label,

        majority_disagrees_with_fusion=
            bool(
                majority_disagrees_with_fusion
            ),

        probability_spread=
            float(
                probability_spread
            ),

        review_required=
            review_required,

        operational_status=
            operational_status,

        score_source=
            "raw_validation_selected_multimodal_fusion",

        risk_band_note=(
            "Risk levels are communication bands only. "
            "The binary decision threshold remains 0.50. "
            "Mixed-modality evidence is surfaced as a review flag "
            "and does not alter the trained classifier decision."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Dataset V2 multimodal branch probabilities "
            "into a final phishing prediction and 0-100 risk score."
        )
    )

    parser.add_argument(
        "--tabular",
        type=float,
        required=True,
        help="LightGBM phishing probability.",
    )

    parser.add_argument(
        "--visual",
        type=float,
        required=True,
        help="ResNet18 phishing probability.",
    )

    parser.add_argument(
        "--graph",
        type=float,
        required=True,
        help="GraphSAGE phishing probability.",
    )

    parser.add_argument(
        "--fusion-manifest",
        default=(
            "results/multimodal_fusion_v2/"
            "fusion_manifest.json"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DECISION_THRESHOLD,
    )

    parser.add_argument(
        "--json-output",
        default=None,
        help=(
            "Optional path for saving the result as JSON."
        ),
    )

    args = parser.parse_args()

    manifest_path = Path(
        args.fusion_manifest
    )

    weights = load_multimodal_weights(
        manifest_path
    )

    result = calculate_risk(
        tabular_probability=
            args.tabular,

        visual_probability=
            args.visual,

        graph_probability=
            args.graph,

        weights=
            weights,

        decision_threshold=
            args.threshold,
    )

    payload = asdict(
        result
    )

    print()
    print("=" * 72)
    print("MULTIMODAL WEBSITE PHISHING RISK SCORE")
    print("=" * 72)
    print(
        f"Prediction: {result.prediction_label}"
    )
    print(
        f"Risk score: {result.risk_score}/100"
    )
    print(
        f"Risk level: {result.risk_level}"
    )
    print(
        f"Fusion p(phishing): "
        f"{result.fusion_probability:.6f}"
    )
    print(
        f"Decision threshold: "
        f"{result.decision_threshold:.2f}"
    )
    print()
    print("BRANCH EVIDENCE")
    print(
        f"LightGBM:   p={result.tabular_probability:.6f} "
        f"x w={result.tabular_weight:.2f} "
        f"= {result.tabular_contribution:.6f}"
    )
    print(
        f"ResNet18:   p={result.visual_probability:.6f} "
        f"x w={result.visual_weight:.2f} "
        f"= {result.visual_contribution:.6f}"
    )
    print(
        f"GraphSAGE:  p={result.graph_probability:.6f} "
        f"x w={result.graph_weight:.2f} "
        f"= {result.graph_contribution:.6f}"
    )
    print()
    print("MODALITY AGREEMENT")
    print(
        f"Phishing branch votes: "
        f"{result.branch_phishing_votes}/3"
    )
    print(
        f"Legitimate branch votes: "
        f"{result.branch_legitimate_votes}/3"
    )
    print(
        f"Agreement: {result.modality_agreement}"
    )
    print(
        f"Majority branch label: "
        f"{result.majority_branch_label}"
    )
    print(
        f"Majority disagrees with fusion: "
        f"{'YES' if result.majority_disagrees_with_fusion else 'NO'}"
    )
    print(
        f"Probability spread: "
        f"{result.probability_spread:.6f}"
    )
    print(
        f"Review required: "
        f"{'YES' if result.review_required else 'NO'}"
    )
    print(
        f"Operational status: "
        f"{result.operational_status}"
    )
    print()
    print(
        "Score source: raw validation-selected "
        "80/10/10 multimodal fusion"
    )
    print(
        "Risk bands are presentation categories, not "
        "separately learned classification thresholds."
    )
    print("=" * 72)

    if args.json_output:
        output_path = Path(
            args.json_output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
            )

        print(
            f"Saved JSON: {output_path}"
        )


if __name__ == "__main__":
    main()
