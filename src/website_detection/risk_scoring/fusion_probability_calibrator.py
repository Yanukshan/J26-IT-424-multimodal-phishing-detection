from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.calibration import calibration_curve


PRIMARY_STRATEGY = "multimodal"


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_branch_probabilities(
    fusion_dir: Path,
    split_name: str,
) -> pd.DataFrame:
    path = fusion_dir / f"{split_name}_branch_probabilities.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing branch probability file: {path}"
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


def fusion_probability(
    df: pd.DataFrame,
    weights: dict,
) -> np.ndarray:
    return (
        float(weights["tabular"])
        * pd.to_numeric(
            df["p_tabular"],
            errors="raise",
        ).to_numpy()
        +
        float(weights["visual"])
        * pd.to_numeric(
            df["p_visual"],
            errors="raise",
        ).to_numpy()
        +
        float(weights["graph"])
        * pd.to_numeric(
            df["p_graph"],
            errors="raise",
        ).to_numpy()
    )


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Simple equal-width Expected Calibration Error.

    This is a descriptive calibration metric.
    """

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

    total = len(y_true)

    if total == 0:
        return float("nan")

    ece = 0.0

    for index in range(bins):
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

        count = int(mask.sum())

        if count == 0:
            continue

        confidence = float(
            probabilities[mask].mean()
        )

        observed_rate = float(
            y_true[mask].mean()
        )

        ece += (
            count / total
        ) * abs(
            confidence
            - observed_rate
        )

    return float(ece)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "accuracy":
            accuracy_score(
                y_true,
                predictions,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                predictions,
            ),

        "precision_phishing":
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "recall_phishing":
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "f1_phishing":
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities,
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities,
            ),

        "brier_score":
            brier_score_loss(
                y_true,
                probabilities,
            ),

        "log_loss":
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            ),

        "ece_10":
            expected_calibration_error(
                y_true,
                probabilities,
                bins=10,
            ),
    }


def fit_platt_calibrator(
    probabilities: np.ndarray,
    y_true: np.ndarray,
) -> LogisticRegression:
    """
    Platt / sigmoid calibration.

    A one-dimensional logistic-regression model is fitted on the
    VALIDATION fusion probability only.

    The test set is never used to fit this model.
    """

    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )

    calibrator.fit(
        probabilities.reshape(
            -1,
            1,
        ),
        y_true,
    )

    return calibrator


def calibrated_probability(
    calibrator: LogisticRegression,
    raw_probability: np.ndarray,
) -> np.ndarray:
    return calibrator.predict_proba(
        raw_probability.reshape(
            -1,
            1,
        )
    )[:, 1]


def save_calibration_plot(
    y_true: np.ndarray,
    raw_probability: np.ndarray,
    calibrated_probability_values: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    raw_true, raw_pred = calibration_curve(
        y_true,
        raw_probability,
        n_bins=10,
        strategy="uniform",
    )

    calibrated_true, calibrated_pred = calibration_curve(
        y_true,
        calibrated_probability_values,
        n_bins=10,
        strategy="uniform",
    )

    figure = plt.figure(
        figsize=(8, 7)
    )

    axis = figure.add_subplot(
        1,
        1,
        1,
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    axis.plot(
        raw_pred,
        raw_true,
        marker="o",
        label="Raw fusion",
    )

    axis.plot(
        calibrated_pred,
        calibrated_true,
        marker="o",
        label="Platt calibrated",
    )

    axis.set_xlabel(
        "Predicted phishing probability"
    )

    axis.set_ylabel(
        "Observed phishing rate"
    )

    axis.set_title(
        title
    )

    axis.set_xlim(
        0.0,
        1.0,
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def build_prediction_output(
    df: pd.DataFrame,
    raw_probability: np.ndarray,
    calibrated_probability_values: np.ndarray,
    strategy: str,
) -> pd.DataFrame:
    output = df[
        [
            "split",
            "source_url",
            "actual_phishing",
            "p_tabular",
            "p_visual",
            "p_graph",
        ]
    ].copy()

    if "source_dataset" in df.columns:
        output["source_dataset"] = df[
            "source_dataset"
        ]

    output[
        "strategy"
    ] = strategy

    output[
        "raw_fusion_probability"
    ] = raw_probability

    output[
        "raw_prediction"
    ] = (
        raw_probability >= 0.5
    ).astype(int)

    output[
        "calibrated_phishing_probability"
    ] = calibrated_probability_values

    output[
        "calibrated_prediction"
    ] = (
        calibrated_probability_values >= 0.5
    ).astype(int)

    output[
        "raw_correct"
    ] = (
        output[
            "raw_prediction"
        ]
        == output[
            "actual_phishing"
        ].astype(int)
    )

    output[
        "calibrated_correct"
    ] = (
        output[
            "calibrated_prediction"
        ]
        == output[
            "actual_phishing"
        ].astype(int)
    )

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit validation-only Platt calibration for Dataset V2 "
            "late-fusion phishing probabilities."
        )
    )

    parser.add_argument(
        "--fusion-dir",
        default="results/multimodal_fusion_v2",
    )

    parser.add_argument(
        "--output-dir",
        default="results/calibration_v2",
    )

    parser.add_argument(
        "--model-dir",
        default=(
            "models/website_detection/"
            "calibration_v2"
        ),
    )

    args = parser.parse_args()

    fusion_dir = Path(
        args.fusion_dir
    )

    output_dir = Path(
        args.output_dir
    )

    model_dir = Path(
        args.model_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest(
        fusion_dir / "fusion_manifest.json"
    )

    strategies = {
        "unconstrained":
            manifest[
                "validation_selected_unconstrained_weights"
            ],

        "multimodal":
            manifest[
                "validation_selected_multimodal_weights"
            ],
    }

    validation = load_branch_probabilities(
        fusion_dir,
        "validation",
    )

    test = load_branch_probabilities(
        fusion_dir,
        "test",
    )

    y_validation = (
        validation[
            "actual_phishing"
        ]
        .astype(int)
        .to_numpy()
    )

    y_test = (
        test[
            "actual_phishing"
        ]
        .astype(int)
        .to_numpy()
    )

    metric_rows = []
    prediction_frames = []
    saved_calibrators = {}

    print()
    print("=" * 96)
    print("DATASET V2 FUSION PROBABILITY CALIBRATION")
    print("=" * 96)
    print(
        "Calibration method: Platt / sigmoid "
        "(LogisticRegression)"
    )
    print(
        "Calibration fit data: VALIDATION ONLY"
    )
    print(
        "Decision threshold: fixed at 0.5"
    )
    print(
        "Test data is evaluated only after the calibrator is fitted."
    )
    print("=" * 96)

    for strategy, weights in strategies.items():
        raw_validation = fusion_probability(
            validation,
            weights,
        )

        raw_test = fusion_probability(
            test,
            weights,
        )

        calibrator = fit_platt_calibrator(
            raw_validation,
            y_validation,
        )

        calibrated_validation = calibrated_probability(
            calibrator,
            raw_validation,
        )

        calibrated_test = calibrated_probability(
            calibrator,
            raw_test,
        )

        saved_calibrators[
            strategy
        ] = {
            "calibrator":
                calibrator,

            "weights":
                weights,
        }

        for split_name, y_true, raw, calibrated in [
            (
                "validation",
                y_validation,
                raw_validation,
                calibrated_validation,
            ),
            (
                "test",
                y_test,
                raw_test,
                calibrated_test,
            ),
        ]:
            raw_metrics = classification_metrics(
                y_true,
                raw,
                threshold=0.5,
            )

            calibrated_metrics = classification_metrics(
                y_true,
                calibrated,
                threshold=0.5,
            )

            metric_rows.append(
                {
                    "strategy":
                        strategy,

                    "split":
                        split_name,

                    "probability_type":
                        "raw",

                    **raw_metrics,
                }
            )

            metric_rows.append(
                {
                    "strategy":
                        strategy,

                    "split":
                        split_name,

                    "probability_type":
                        "platt_calibrated",

                    **calibrated_metrics,
                }
            )

        prediction_frames.append(
            build_prediction_output(
                validation,
                raw_validation,
                calibrated_validation,
                strategy,
            )
        )

        prediction_frames.append(
            build_prediction_output(
                test,
                raw_test,
                calibrated_test,
                strategy,
            )
        )

        save_calibration_plot(
            y_true=
                y_validation,

            raw_probability=
                raw_validation,

            calibrated_probability_values=
                calibrated_validation,

            output_path=
                output_dir
                / (
                    f"{strategy}_"
                    "validation_calibration_curve.png"
                ),

            title=(
                f"{strategy.title()} fusion - "
                "Validation calibration"
            ),
        )

        save_calibration_plot(
            y_true=
                y_test,

            raw_probability=
                raw_test,

            calibrated_probability_values=
                calibrated_test,

            output_path=
                output_dir
                / (
                    f"{strategy}_"
                    "test_calibration_curve.png"
                ),

            title=(
                f"{strategy.title()} fusion - "
                "Test calibration"
            ),
        )

        print()
        print(
            f"STRATEGY: {strategy}"
        )

        print(
            "Weights: "
            f"tabular={weights['tabular']:.2f}, "
            f"visual={weights['visual']:.2f}, "
            f"graph={weights['graph']:.2f}"
        )

        print(
            "Platt coefficient: "
            f"{float(calibrator.coef_[0][0]):.6f}"
        )

        print(
            "Platt intercept: "
            f"{float(calibrator.intercept_[0]):.6f}"
        )

    metrics = pd.DataFrame(
        metric_rows
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    metrics.to_csv(
        output_dir
        / "calibration_metrics.csv",
        index=False,
    )

    predictions.to_csv(
        output_dir
        / "calibrated_predictions.csv",
        index=False,
    )

    joblib.dump(
        saved_calibrators,
        model_dir
        / "fusion_platt_calibrators.joblib",
    )

    primary = saved_calibrators[
        PRIMARY_STRATEGY
    ]

    joblib.dump(
        primary,
        model_dir
        / "primary_multimodal_calibrator.joblib",
    )

    calibration_manifest = {
        "method":
            "Platt sigmoid calibration via LogisticRegression",

        "fit_split":
            "validation",

        "test_used_for_fitting":
            False,

        "decision_threshold":
            0.5,

        "primary_strategy":
            PRIMARY_STRATEGY,

        "primary_weights":
            strategies[
                PRIMARY_STRATEGY
            ],

        "calibrated_strategies":
            list(
                strategies.keys()
            ),

        "validation_rows":
            len(
                validation
            ),

        "test_rows":
            len(
                test
            ),

        "positive_class":
            "phishing",

        "risk_score_note":
            (
                "Calibrated probability can be used as the basis of a "
                "user-facing risk score, but final communication bands "
                "should be treated as presentation categories rather "
                "than separately learned class thresholds."
            ),

        "methodology_note":
            (
                "No fusion weights or thresholds are changed by this "
                "script. Calibration is fitted on validation only. "
                "The existing V2 test set is used only for descriptive "
                "evaluation. Final claims should be confirmed on a "
                "fresh later-collected external dataset."
            ),
    }

    with (
        output_dir
        / "calibration_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            calibration_manifest,
            file,
            indent=2,
        )

    print()
    print("=" * 96)
    print("CALIBRATION RESULTS")
    print("=" * 96)

    display_columns = [
        "strategy",
        "split",
        "probability_type",
        "accuracy",
        "balanced_accuracy",
        "precision_phishing",
        "recall_phishing",
        "f1_phishing",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "log_loss",
        "ece_10",
    ]

    print(
        metrics[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Primary final-system probability source: "
        f"{PRIMARY_STRATEGY} fusion"
    )

    print(
        "Primary multimodal weights: "
        f"tabular={strategies[PRIMARY_STRATEGY]['tabular']:.2f}, "
        f"visual={strategies[PRIMARY_STRATEGY]['visual']:.2f}, "
        f"graph={strategies[PRIMARY_STRATEGY]['graph']:.2f}"
    )

    print()
    print(
        f"Outputs: {output_dir}"
    )

    print(
        f"Saved calibrators: {model_dir}"
    )

    print("=" * 96)


if __name__ == "__main__":
    main()
