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
from PIL import Image
from torchvision import models


def safe_name(value: str, max_length: int = 70) -> str:
    text = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(value),
    ).strip("_")
    return (text or "sample")[:max_length]


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Any:
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


def extract_state_dict(checkpoint: Any) -> dict:
    if isinstance(checkpoint, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "model",
        ]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        if checkpoint and all(
            torch.is_tensor(value)
            for value in checkpoint.values()
        ):
            return checkpoint

    raise RuntimeError(
        "Could not find a model state_dict in the ResNet18 checkpoint."
    )


def normalize_state_dict_keys(state_dict: dict) -> dict:
    normalized = {}

    for key, value in state_dict.items():
        new_key = str(key)

        for prefix in [
            "module.",
            "model.",
        ]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        normalized[new_key] = value

    return normalized


def build_resnet18(
    checkpoint: Any,
    device: torch.device,
) -> nn.Module:
    """
    Rebuild the saved Dataset V2 ResNet18 architecture.

    IMPORTANT:
    The original trainer used ImageNet pretrained weights only as
    initialization. For inference we load the saved checkpoint, so the
    constructor itself uses weights=None.
    """

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        2,
    )

    state_dict = normalize_state_dict_keys(
        extract_state_dict(checkpoint)
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing:
        raise RuntimeError(
            "Missing checkpoint keys: "
            + ", ".join(missing[:20])
        )

    if unexpected:
        raise RuntimeError(
            "Unexpected checkpoint keys: "
            + ", ".join(unexpected[:20])
        )

    model = model.to(device)
    model.eval()

    return model


def original_eval_transform():
    """
    EXACT preprocessing used by visual_cnn_v2_trainer.py.

    The trainer returned:
        weights.transforms()

    where the model used ImageNet ResNet18 weights.

    ResNet18_Weights.DEFAULT.transforms() includes the correct torchvision
    resize/crop/interpolation/normalization pipeline. This must be used for
    XAI too; replacing it with Resize((224, 224)) changes model probabilities.
    """

    weights = models.ResNet18_Weights.DEFAULT
    return weights.transforms()


class GradCAM:
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ) -> None:
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ) -> None:
        self.activations = output.detach()

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output,
    ) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ):
        self.model.zero_grad(
            set_to_none=True
        )

        logits = self.model(
            input_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        score = logits[
            0,
            target_class,
        ]

        score.backward()

        if (
            self.activations is None
            or self.gradients is None
        ):
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients."
            )

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights
            * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = torch.relu(
            cam
        )

        cam = torch.nn.functional.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[
            0,
            0,
        ]

        cam_min = float(
            cam.min().item()
        )

        cam_max = float(
            cam.max().item()
        )

        if cam_max > cam_min:
            cam = (
                cam
                - cam_min
            ) / (
                cam_max
                - cam_min
            )
        else:
            cam = torch.zeros_like(
                cam
            )

        return (
            cam.detach().cpu().numpy(),
            logits.detach().cpu().numpy()[0],
            probabilities.detach().cpu().numpy()[0],
        )


def find_screenshot_column(
    dataframe: pd.DataFrame,
) -> str:
    candidates = [
        "visual_screenshot_path",
        "screenshot_path",
        "visual_artifact_path",
        "visual_screenshot_artifact_path",
        "screenshot_artifact_path",
        "visual_path",
    ]

    for column in candidates:
        if column in dataframe.columns:
            return column

    for column in dataframe.columns:
        lower = column.lower()

        if (
            "screenshot" in lower
            and (
                "path" in lower
                or "artifact" in lower
            )
        ):
            return column

    raise RuntimeError(
        "Could not find screenshot path column."
    )


def resolve_artifact_path(
    value: Any,
) -> Path:
    path = Path(str(value))

    if path.exists():
        return path

    normalized = Path(
        str(value).replace("\\", "/")
    )

    return normalized


def find_sample_row(
    split_dir: Path,
    source_url: str,
):
    for split_name in [
        "validation",
        "test",
        "train",
    ]:
        split_path = split_dir / f"{split_name}.csv"
        dataframe = pd.read_csv(split_path)

        matches = dataframe[
            dataframe["source_url"].astype(str)
            == str(source_url)
        ]

        if not matches.empty:
            return (
                split_name,
                dataframe,
                matches.index[0],
            )

    return None, None, None


def target_urls_from_lightgbm_misses(
    missed_path: Path,
) -> list[str]:
    dataframe = pd.read_csv(
        missed_path
    )

    if "source_url" not in dataframe.columns:
        raise ValueError(
            "missed_phishing_explanations.csv does not contain source_url."
        )

    return list(
        dict.fromkeys(
            dataframe["source_url"]
            .astype(str)
            .tolist()
        )
    )


def load_original_prediction_probabilities(
    results_dir: Path,
) -> dict[tuple[str, str], float]:
    """
    Reads the exact saved probabilities produced by visual_cnn_v2_trainer.py.
    Used only as a consistency check.

    Expected key:
        (split, source_url)
    """

    lookup: dict[tuple[str, str], float] = {}

    for split_name in [
        "validation",
        "test",
    ]:
        path = (
            results_dir
            / f"{split_name}_predictions.csv"
        )

        if not path.exists():
            continue

        df = pd.read_csv(path)

        if "source_url" not in df.columns:
            continue

        probability_column = None

        candidates = [
            "phishing_probability",
            "probability_phishing",
            "p_phishing",
            "probability",
        ]

        for candidate in candidates:
            if candidate in df.columns:
                probability_column = candidate
                break

        if probability_column is None:
            numeric_candidates = [
                c
                for c in df.columns
                if (
                    "prob" in c.lower()
                    and pd.api.types.is_numeric_dtype(df[c])
                )
            ]

            if len(numeric_candidates) == 1:
                probability_column = numeric_candidates[0]

        if probability_column is None:
            raise RuntimeError(
                f"Could not identify phishing-probability column in {path}. "
                f"Columns: {list(df.columns)}"
            )

        for _, row in df.iterrows():
            lookup[
                (
                    split_name,
                    str(row["source_url"]),
                )
            ] = float(
                row[probability_column]
            )

    return lookup


def resize_cam_to_original(
    cam: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    cam_uint8 = (
        np.clip(
            cam,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(
        np.uint8
    )

    resized = Image.fromarray(
        cam_uint8
    ).resize(
        (
            width,
            height,
        ),
        resample=Image.Resampling.BILINEAR,
    )

    return (
        np.asarray(
            resized,
            dtype=np.float32,
        )
        / 255.0
    )


def save_gradcam_figure(
    original_image: Image.Image,
    cam: np.ndarray,
    output_path: Path,
    source_url: str,
    target_probability: float,
    prediction_name: str,
) -> None:
    image_array = np.asarray(
        original_image.convert("RGB")
    )

    cam_original = resize_cam_to_original(
        cam,
        original_image.width,
        original_image.height,
    )

    fig = plt.figure(
        figsize=(16, 6)
    )

    ax1 = fig.add_subplot(
        1,
        3,
        1,
    )

    ax1.imshow(
        image_array
    )

    ax1.set_title(
        "Original screenshot"
    )

    ax1.axis(
        "off"
    )

    ax2 = fig.add_subplot(
        1,
        3,
        2,
    )

    heatmap = ax2.imshow(
        cam_original,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )

    ax2.set_title(
        "Grad-CAM target: phishing"
    )

    ax2.axis(
        "off"
    )

    fig.colorbar(
        heatmap,
        ax=ax2,
        fraction=0.046,
        pad=0.04,
    )

    ax3 = fig.add_subplot(
        1,
        3,
        3,
    )

    ax3.imshow(
        image_array
    )

    ax3.imshow(
        cam_original,
        cmap="jet",
        alpha=0.45,
        vmin=0.0,
        vmax=1.0,
    )

    ax3.set_title(
        "Grad-CAM overlay"
    )

    ax3.axis(
        "off"
    )

    fig.suptitle(
        (
            f"{source_url}\n"
            f"Model prediction: {prediction_name} | "
            f"p(phishing)={target_probability:.6f}"
        ),
        fontsize=11,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def top_cam_regions(
    cam: np.ndarray,
    grid_rows: int = 4,
    grid_cols: int = 4,
    top_n: int = 5,
) -> list[dict]:
    height, width = cam.shape
    rows = []

    for row in range(grid_rows):
        y0 = int(
            row
            * height
            / grid_rows
        )

        y1 = int(
            (row + 1)
            * height
            / grid_rows
        )

        for col in range(grid_cols):
            x0 = int(
                col
                * width
                / grid_cols
            )

            x1 = int(
                (col + 1)
                * width
                / grid_cols
            )

            score = float(
                cam[
                    y0:y1,
                    x0:x1
                ].mean()
            )

            vertical = (
                "top"
                if row == 0
                else (
                    "bottom"
                    if row
                    == grid_rows - 1
                    else "middle"
                )
            )

            horizontal = (
                "left"
                if col == 0
                else (
                    "right"
                    if col
                    == grid_cols - 1
                    else "center"
                )
            )

            rows.append(
                {
                    "grid_row": row,
                    "grid_col": col,
                    "region": f"{vertical}-{horizontal}",
                    "mean_cam_activation": score,
                }
            )

    rows.sort(
        key=lambda item: item[
            "mean_cam_activation"
        ],
        reverse=True,
    )

    return rows[:top_n]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ResNet18 Grad-CAM explanations using the exact "
            "Dataset V2 visual-CNN inference preprocessing."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "models/website_detection/"
            "visual_cnn_v2/resnet18_v2.pt"
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
        "--visual-results-dir",
        default=(
            "results/visual_cnn_v2"
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
            "results/xai_v2/resnet18_gradcam"
        ),
    )

    parser.add_argument(
        "--probability-tolerance",
        type=float,
        default=1e-5,
    )

    args = parser.parse_args()

    checkpoint_path = Path(
        args.checkpoint
    )

    split_dir = Path(
        args.split_dir
    )

    visual_results_dir = Path(
        args.visual_results_dir
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
            f"ResNet18 checkpoint not found: {checkpoint_path}"
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

    model = build_resnet18(
        checkpoint,
        device,
    )

    # Exact original trainer preprocessing:
    transform = original_eval_transform()

    original_probability_lookup = (
        load_original_prediction_probabilities(
            visual_results_dir
        )
    )

    gradcam = GradCAM(
        model=model,
        target_layer=model.layer4[-1],
    )

    target_urls = target_urls_from_lightgbm_misses(
        missed_path
    )

    summary_rows = []
    region_rows = []

    print()
    print("=" * 92)
    print("RESNET18 GRAD-CAM EXPLAINABILITY - DATASET V2")
    print("=" * 92)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Preprocessing: ResNet18_Weights.DEFAULT.transforms()")
    print(
        f"LightGBM missed-phishing URLs to explain: "
        f"{len(target_urls)}"
    )
    print("Grad-CAM target class: phishing")
    print("=" * 92)

    try:
        for source_url in target_urls:
            (
                split_name,
                split_dataframe,
                row_index,
            ) = find_sample_row(
                split_dir,
                source_url,
            )

            if split_dataframe is None:
                print(
                    f"SKIP: URL not found in splits: {source_url}"
                )
                continue

            screenshot_column = find_screenshot_column(
                split_dataframe
            )

            row = split_dataframe.loc[
                row_index
            ]

            screenshot_path = resolve_artifact_path(
                row[
                    screenshot_column
                ]
            )

            if not screenshot_path.exists():
                print(
                    f"SKIP: screenshot does not exist: {screenshot_path}"
                )
                continue

            image = Image.open(
                screenshot_path
            ).convert(
                "RGB"
            )

            input_tensor = transform(
                image
            ).unsqueeze(
                0
            ).to(
                device
            )

            cam, logits, probabilities = gradcam.generate(
                input_tensor=input_tensor,
                target_class=1,
            )

            predicted_class = int(
                np.argmax(probabilities)
            )

            phishing_probability = float(
                probabilities[1]
            )

            legitimate_probability = float(
                probabilities[0]
            )

            predicted_name = (
                "phishing"
                if predicted_class == 1
                else "legitimate"
            )

            actual_phishing = int(
                pd.to_numeric(
                    row["source_label"],
                    errors="raise",
                )
                == 0
            )

            expected_probability = (
                original_probability_lookup.get(
                    (
                        split_name,
                        source_url,
                    )
                )
            )

            if expected_probability is None:
                probability_difference = np.nan
                probability_match = False
            else:
                probability_difference = abs(
                    phishing_probability
                    - expected_probability
                )

                probability_match = (
                    probability_difference
                    <= args.probability_tolerance
                )

            slug = (
                f"{split_name}_"
                f"{safe_name(source_url)}"
            )

            figure_path = (
                output_dir
                / f"{slug}_phishing_gradcam.png"
            )

            save_gradcam_figure(
                original_image=image,
                cam=cam,
                output_path=figure_path,
                source_url=source_url,
                target_probability=phishing_probability,
                prediction_name=predicted_name,
            )

            regions = top_cam_regions(
                cam,
                grid_rows=4,
                grid_cols=4,
                top_n=5,
            )

            for rank, item in enumerate(
                regions,
                start=1,
            ):
                region_rows.append(
                    {
                        "split": split_name,
                        "source_url": source_url,
                        "source_dataset": str(
                            row.get(
                                "source_dataset",
                                "",
                            )
                        ),
                        "region_rank": rank,
                        **item,
                    }
                )

            summary_rows.append(
                {
                    "split": split_name,
                    "source_url": source_url,
                    "source_dataset": str(
                        row.get(
                            "source_dataset",
                            "",
                        )
                    ),
                    "actual_phishing": actual_phishing,
                    "resnet18_predicted_phishing": int(
                        predicted_class == 1
                    ),
                    "resnet18_phishing_probability":
                        phishing_probability,
                    "original_saved_phishing_probability":
                        expected_probability,
                    "probability_absolute_difference":
                        probability_difference,
                    "probability_matches_saved_result":
                        probability_match,
                    "resnet18_legitimate_probability":
                        legitimate_probability,
                    "visual_correct": bool(
                        predicted_class
                        == actual_phishing
                    ),
                    "screenshot_path": str(
                        screenshot_path
                    ),
                    "screenshot_width": int(
                        image.width
                    ),
                    "screenshot_height": int(
                        image.height
                    ),
                    "gradcam_target": "phishing",
                    "top_cam_region": (
                        regions[0]["region"]
                        if regions
                        else ""
                    ),
                    "top_cam_region_activation": (
                        float(
                            regions[0][
                                "mean_cam_activation"
                            ]
                        )
                        if regions
                        else 0.0
                    ),
                    "gradcam_figure": str(
                        figure_path
                    ),
                }
            )

            print()
            print(
                f"Explaining: {source_url}"
            )
            print(
                f"  split={split_name}"
            )
            print(
                f"  actual="
                f"{'phishing' if actual_phishing == 1 else 'legitimate'}"
            )
            print(
                f"  ResNet18 prediction={predicted_name}"
            )
            print(
                f"  p(phishing)={phishing_probability:.6f}"
            )

            if expected_probability is not None:
                print(
                    f"  saved original p(phishing)="
                    f"{expected_probability:.6f}"
                )
                print(
                    f"  absolute difference="
                    f"{probability_difference:.10f}"
                )
                print(
                    f"  probability match="
                    f"{'YES' if probability_match else 'NO'}"
                )

            print(
                "  Top 5 Grad-CAM regions:"
            )

            print(
                pd.DataFrame(
                    regions
                ).to_string(
                    index=False
                )
            )

    finally:
        gradcam.close()

    summary = pd.DataFrame(
        summary_rows
    )

    regions_frame = pd.DataFrame(
        region_rows
    )

    summary.to_csv(
        output_dir
        / "resnet18_gradcam_summary.csv",
        index=False,
    )

    regions_frame.to_csv(
        output_dir
        / "resnet18_gradcam_regions.csv",
        index=False,
    )

    all_probabilities_match = (
        bool(
            summary[
                "probability_matches_saved_result"
            ].all()
        )
        if not summary.empty
        else False
    )

    manifest = {
        "checkpoint":
            str(
                checkpoint_path
            ),

        "architecture":
            "ResNet18",

        "target_layer":
            "layer4[-1]",

        "gradcam_target_class":
            "phishing",

        "preprocessing":
            "torchvision.models.ResNet18_Weights.DEFAULT.transforms()",

        "class_mapping": {
            "0": "legitimate",
            "1": "phishing",
        },

        "images_requested":
            len(
                target_urls
            ),

        "images_explained":
            len(
                summary
            ),

        "probability_tolerance":
            args.probability_tolerance,

        "all_explained_probabilities_match_saved_predictions":
            all_probabilities_match,

        "note":
            (
                "Grad-CAM is descriptive visual explainability. "
                "The script verifies that inference probabilities match "
                "the original saved visual-CNN prediction files before "
                "the explanations are accepted."
            ),
    }

    with (
        output_dir
        / "resnet18_gradcam_manifest.json"
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
    print("=" * 92)
    print("RESNET18 GRAD-CAM SUMMARY")
    print("=" * 92)

    if summary.empty:
        print(
            "No screenshots were explained."
        )
    else:
        print(
            summary[
                [
                    "split",
                    "source_url",
                    "actual_phishing",
                    "resnet18_predicted_phishing",
                    "resnet18_phishing_probability",
                    "original_saved_phishing_probability",
                    "probability_absolute_difference",
                    "probability_matches_saved_result",
                    "visual_correct",
                    "top_cam_region",
                ]
            ].to_string(
                index=False
            )
        )

        print()
        print(
            "All explained probabilities match original saved "
            f"predictions: {'YES' if all_probabilities_match else 'NO'}"
        )

    print()
    print(
        f"Outputs: {output_dir}"
    )

    print("=" * 92)

    if not summary.empty and not all_probabilities_match:
        raise RuntimeError(
            "Grad-CAM inference probabilities still do not match the "
            "original visual-CNN saved predictions. Do not use the "
            "heatmaps until preprocessing/model consistency is fixed."
        )


if __name__ == "__main__":
    main()
