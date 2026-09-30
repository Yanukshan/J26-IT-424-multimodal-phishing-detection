from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


TOP_FEATURES_PER_SAMPLE = 10


def build_target(df: pd.DataFrame) -> pd.Series:
    labels = pd.to_numeric(df["source_label"], errors="coerce")
    if labels.isna().any() or (~labels.isin([0, 1])).any():
        raise ValueError("source_label must contain only 0 and 1.")
    # 1 = phishing, 0 = legitimate
    return labels.eq(0).astype(int).rename("actual_phishing")


def get_feature_names(model: Any, manifest_path: Path) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(x) for x in names]

    if isinstance(model, Pipeline):
        for step in model.named_steps.values():
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                return [str(x) for x in names]

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        for key in [
            "features",
            "feature_columns",
            "usable_features",
            "selected_features",
            "kept_features",
        ]:
            value = manifest.get(key)
            if isinstance(value, list) and value:
                return [str(x) for x in value]

    raise RuntimeError(
        "Could not recover model feature names from the saved model "
        "or feature_manifest.json."
    )


def prepare_input(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing model features: "
            + ", ".join(missing[:20])
        )
    return df[features].copy()


def get_estimator_and_preprocessors(model: Any):
    if isinstance(model, Pipeline):
        estimator = (
            model.named_steps["model"]
            if "model" in model.named_steps
            else model.steps[-1][1]
        )
        return estimator, model.steps[:-1]
    return model, []


def transform_for_tree(
    X: pd.DataFrame,
    preprocessing_steps,
) -> np.ndarray:
    transformed: Any = X
    for _, transformer in preprocessing_steps:
        transformed = transformer.transform(transformed)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return np.asarray(transformed, dtype=np.float64)


def normalize_shap_values(raw_values: Any, feature_count: int) -> np.ndarray:
    if isinstance(raw_values, list):
        if len(raw_values) >= 2:
            return np.asarray(raw_values[1], dtype=np.float64)
        return np.asarray(raw_values[0], dtype=np.float64)

    values = np.asarray(raw_values)

    if values.ndim == 2:
        return values.astype(np.float64)

    if values.ndim == 3:
        # [samples, features, classes]
        if values.shape[1] == feature_count and values.shape[2] >= 2:
            return values[:, :, 1].astype(np.float64)

        # [classes, samples, features]
        if values.shape[0] >= 2 and values.shape[2] == feature_count:
            return values[1, :, :].astype(np.float64)

    raise RuntimeError(f"Unexpected SHAP value shape: {values.shape}")


def get_base_value(explainer: shap.TreeExplainer) -> float:
    expected = np.asarray(explainer.expected_value)

    if expected.ndim == 0:
        return float(expected)

    flat = expected.reshape(-1)

    if len(flat) >= 2:
        return float(flat[1])

    return float(flat[0])


def safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return (text or "sample")[:80]


def local_rows(
    df: pd.DataFrame,
    X: pd.DataFrame,
    shap_values: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    split_name: str,
    top_n: int,
) -> pd.DataFrame:
    target = build_target(df).to_numpy()
    rows = []

    for i in range(len(df)):
        order = np.argsort(np.abs(shap_values[i]))[::-1][:top_n]

        for rank, feature_index in enumerate(order, start=1):
            shap_value = float(shap_values[i, feature_index])
            feature_name = X.columns[feature_index]

            rows.append(
                {
                    "split": split_name,
                    "source_url": str(df.iloc[i]["source_url"]),
                    "source_dataset": str(
                        df.iloc[i].get("source_dataset", "")
                    ),
                    "actual_phishing": int(target[i]),
                    "predicted_phishing": int(predictions[i]),
                    "phishing_probability": float(probabilities[i]),
                    "feature_rank": rank,
                    "feature": feature_name,
                    "feature_value": X.iloc[i, feature_index],
                    "shap_value": shap_value,
                    "effect": (
                        "toward_phishing"
                        if shap_value > 0
                        else (
                            "toward_legitimate"
                            if shap_value < 0
                            else "neutral"
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


def save_false_negative_waterfalls(
    df: pd.DataFrame,
    X_transformed: np.ndarray,
    X_original: pd.DataFrame,
    shap_values: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    base_value: float,
    output_dir: Path,
    split_name: str,
) -> int:
    target = build_target(df).to_numpy()
    positions = np.where((target == 1) & (predictions == 0))[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for i in positions:
        source_url = str(df.iloc[i]["source_url"])
        source_dataset = str(df.iloc[i].get("source_dataset", ""))

        explanation = shap.Explanation(
            values=shap_values[i],
            base_values=base_value,
            data=X_transformed[i],
            feature_names=list(X_original.columns),
        )

        shap.plots.waterfall(
            explanation,
            max_display=15,
            show=False,
        )

        plt.title(
            f"{split_name.upper()} false negative\n"
            f"{source_dataset} | p(phishing)={probabilities[i]:.4f}",
            fontsize=10,
        )

        filename = (
            f"{split_name}_{i:03d}_"
            f"{safe_name(source_dataset)}_"
            f"{safe_name(source_url)}.png"
        )

        plt.tight_layout()
        plt.savefig(
            output_dir / filename,
            dpi=180,
            bbox_inches="tight",
        )
        plt.close()
        saved += 1

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain the Dataset V2 LightGBM model using SHAP."
    )

    parser.add_argument(
        "--model",
        default=(
            "models/website_detection/"
            "ml_baselines_v2/lightgbm.joblib"
        ),
    )
    parser.add_argument(
        "--feature-manifest",
        default=(
            "results/ml_baselines_v2/"
            "feature_manifest.json"
        ),
    )
    parser.add_argument(
        "--split-dir",
        default="data/processed/final_splits_v2",
    )
    parser.add_argument(
        "--output-dir",
        default="results/xai_v2/lightgbm",
    )
    parser.add_argument(
        "--top-features-per-sample",
        type=int,
        default=TOP_FEATURES_PER_SAMPLE,
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    manifest_path = Path(args.feature_manifest)
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    saved_model = joblib.load(model_path)
    feature_names = get_feature_names(saved_model, manifest_path)
    estimator, preprocessing_steps = get_estimator_and_preprocessors(
        saved_model
    )

    validation_df = pd.read_csv(split_dir / "validation.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    X_validation = prepare_input(validation_df, feature_names)
    X_test = prepare_input(test_df, feature_names)

    X_validation_t = transform_for_tree(
        X_validation,
        preprocessing_steps,
    )
    X_test_t = transform_for_tree(
        X_test,
        preprocessing_steps,
    )

    explainer = shap.TreeExplainer(estimator)

    validation_shap = normalize_shap_values(
        explainer.shap_values(X_validation_t),
        len(feature_names),
    )
    test_shap = normalize_shap_values(
        explainer.shap_values(X_test_t),
        len(feature_names),
    )

    validation_probability = saved_model.predict_proba(
        X_validation
    )[:, 1]
    test_probability = saved_model.predict_proba(
        X_test
    )[:, 1]

    validation_prediction = (
        validation_probability >= 0.5
    ).astype(int)
    test_prediction = (
        test_probability >= 0.5
    ).astype(int)

    # Global importance uses validation + test only for descriptive XAI.
    combined_shap = np.vstack([validation_shap, test_shap])
    combined_t = np.vstack([X_validation_t, X_test_t])

    global_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.mean(
                np.abs(combined_shap),
                axis=0,
            ),
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    global_importance.insert(
        0,
        "rank",
        np.arange(1, len(global_importance) + 1),
    )

    global_importance.to_csv(
        output_dir / "global_shap_importance.csv",
        index=False,
    )

    # Beeswarm plot.
    shap.summary_plot(
        combined_shap,
        combined_t,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "shap_summary_beeswarm.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    # Bar plot.
    shap.summary_plot(
        combined_shap,
        combined_t,
        feature_names=feature_names,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "shap_summary_bar.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    validation_local = local_rows(
        validation_df,
        X_validation,
        validation_shap,
        validation_probability,
        validation_prediction,
        "validation",
        args.top_features_per_sample,
    )

    test_local = local_rows(
        test_df,
        X_test,
        test_shap,
        test_probability,
        test_prediction,
        "test",
        args.top_features_per_sample,
    )

    validation_local.to_csv(
        output_dir / "validation_shap_explanations.csv",
        index=False,
    )
    test_local.to_csv(
        output_dir / "test_shap_explanations.csv",
        index=False,
    )

    validation_target = build_target(validation_df).to_numpy()
    test_target = build_target(test_df).to_numpy()

    validation_fn_urls = set(
        validation_df.loc[
            (validation_target == 1)
            & (validation_prediction == 0),
            "source_url",
        ].astype(str)
    )
    test_fn_urls = set(
        test_df.loc[
            (test_target == 1)
            & (test_prediction == 0),
            "source_url",
        ].astype(str)
    )

    missed = pd.concat(
        [
            validation_local[
                validation_local["source_url"].isin(
                    validation_fn_urls
                )
            ],
            test_local[
                test_local["source_url"].isin(
                    test_fn_urls
                )
            ],
        ],
        ignore_index=True,
    )

    missed.to_csv(
        output_dir / "missed_phishing_explanations.csv",
        index=False,
    )

    base_value = get_base_value(explainer)
    waterfall_dir = output_dir / "missed_phishing_waterfalls"

    saved_validation = save_false_negative_waterfalls(
        validation_df,
        X_validation_t,
        X_validation,
        validation_shap,
        validation_probability,
        validation_prediction,
        base_value,
        waterfall_dir,
        "validation",
    )

    saved_test = save_false_negative_waterfalls(
        test_df,
        X_test_t,
        X_test,
        test_shap,
        test_probability,
        test_prediction,
        base_value,
        waterfall_dir,
        "test",
    )

    prediction_rows = []

    for split_name, df, probability, prediction in [
        (
            "validation",
            validation_df,
            validation_probability,
            validation_prediction,
        ),
        (
            "test",
            test_df,
            test_probability,
            test_prediction,
        ),
    ]:
        target = build_target(df).to_numpy()

        for i in range(len(df)):
            prediction_rows.append(
                {
                    "split": split_name,
                    "source_url": str(df.iloc[i]["source_url"]),
                    "source_dataset": str(
                        df.iloc[i].get("source_dataset", "")
                    ),
                    "actual_phishing": int(target[i]),
                    "predicted_phishing": int(prediction[i]),
                    "phishing_probability": float(probability[i]),
                    "correct": bool(target[i] == prediction[i]),
                }
            )

    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "lightgbm_xai_predictions.csv",
        index=False,
    )

    xai_manifest = {
        "model": str(model_path),
        "underlying_estimator": type(estimator).__name__,
        "positive_class": "phishing",
        "feature_count": len(feature_names),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "validation_false_negatives": len(validation_fn_urls),
        "test_false_negatives": len(test_fn_urls),
        "waterfall_plots_saved": saved_validation + saved_test,
        "top_features_per_sample": args.top_features_per_sample,
        "note": (
            "This script performs descriptive explainability only. "
            "It does not retrain the model, change thresholds, "
            "or tune using the test set."
        ),
    }

    with (
        output_dir / "lightgbm_xai_manifest.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(xai_manifest, f, indent=2)

    print()
    print("=" * 86)
    print("LIGHTGBM SHAP EXPLAINABILITY - DATASET V2")
    print("=" * 86)
    print(f"Model: {model_path}")
    print(f"Underlying estimator: {type(estimator).__name__}")
    print(f"Feature count: {len(feature_names)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Test rows: {len(test_df)}")
    print()
    print("TOP 20 GLOBAL SHAP FEATURES")
    print(
        global_importance.head(20).to_string(index=False)
    )
    print()
    print(
        "Validation false-negative phishing pages: "
        f"{len(validation_fn_urls)}"
    )
    print(
        "Test false-negative phishing pages: "
        f"{len(test_fn_urls)}"
    )
    print(
        "Waterfall plots saved: "
        f"{saved_validation + saved_test}"
    )
    print()
    print(f"Outputs: {output_dir}")
    print("=" * 86)


if __name__ == "__main__":
    main()
