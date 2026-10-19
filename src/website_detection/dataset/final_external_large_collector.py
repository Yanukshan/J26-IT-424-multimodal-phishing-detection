from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.website_detection.dataset.collector import (
    LABEL_LEGITIMATE,
    LABEL_PHISHING,
    collect_dataset,
)


DEFAULT_INPUT = (
    "data/processed/final_external_2026/"
    "final_external_candidate_pool.csv"
)

DEFAULT_OUTPUT_DIR = (
    "data/processed/final_external_2026/crawled"
)

DEFAULT_SEED = 42


def first_present_value(
    row: pd.Series,
    candidates: list[str],
) -> str:
    for column in candidates:
        if column not in row.index:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        text = str(value).strip()

        if text:
            return text

    return ""


def load_candidate_pool(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Candidate pool not found: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "source_url",
        "crawler_label",
        "candidate_class",
        "source_dataset",
        "registrable_domain",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            "Candidate pool is missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    df = df.copy()

    df["source_url"] = (
        df["source_url"]
        .astype(str)
        .str.strip()
    )

    df["crawler_label"] = pd.to_numeric(
        df["crawler_label"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "source_url",
            "crawler_label",
        ]
    ).copy()

    df = df[
        df["source_url"].ne("")
    ].copy()

    df["crawler_label"] = (
        df["crawler_label"]
        .astype(int)
    )

    df = df[
        df["crawler_label"].isin(
            [
                LABEL_PHISHING,
                LABEL_LEGITIMATE,
            ]
        )
    ].copy()

    df = (
        df.drop_duplicates(
            subset=[
                "source_url",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    return df


def select_mode(
    df: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    if mode == "phishing":
        return df[
            df["crawler_label"]
            == LABEL_PHISHING
        ].copy()

    if mode == "legitimate":
        return df[
            df["crawler_label"]
            == LABEL_LEGITIMATE
        ].copy()

    if mode == "all":
        return df.copy()

    raise ValueError(
        f"Unsupported mode: {mode}"
    )


def deterministic_order(
    df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    return (
        df.sample(
            frac=1.0,
            random_state=seed,
        )
        .reset_index(
            drop=True
        )
    )


def to_collector_input(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        source_id = first_present_value(
            row,
            [
                "source_id",
                "phish_id",
                "rank",
            ],
        )

        source_verified = first_present_value(
            row,
            [
                "source_verified",
                "verified",
            ],
        )

        source_online = first_present_value(
            row,
            [
                "source_online",
                "online",
            ],
        )

        source_submission_time = first_present_value(
            row,
            [
                "source_submission_time",
                "submission_time",
            ],
        )

        source_verification_time = first_present_value(
            row,
            [
                "source_verification_time",
                "verification_time",
            ],
        )

        source_target = first_present_value(
            row,
            [
                "source_target",
                "target",
            ],
        )

        source_detail_url = first_present_value(
            row,
            [
                "source_detail_url",
                "phish_detail_url",
            ],
        )

        source_loaded_at_utc = first_present_value(
            row,
            [
                "source_loaded_at_utc",
                "pool_built_at_utc",
                "source_snapshot_date",
            ],
        )

        rows.append(
            {
                "URL":
                    str(
                        row["source_url"]
                    ),

                "label":
                    int(
                        row["crawler_label"]
                    ),

                "source_dataset":
                    str(
                        row["source_dataset"]
                    ),

                "source_id":
                    source_id,

                "source_domain":
                    str(
                        row[
                            "registrable_domain"
                        ]
                    ),

                "source_verified":
                    source_verified,

                "source_online":
                    source_online,

                "source_submission_time":
                    source_submission_time,

                "source_verification_time":
                    source_verification_time,

                "source_target":
                    source_target,

                "source_detail_url":
                    source_detail_url,

                "source_loaded_at_utc":
                    source_loaded_at_utc,
            }
        )

    return pd.DataFrame(
        rows
    )


def print_source_summary(
    df: pd.DataFrame,
) -> None:
    if df.empty:
        print("No selected candidates.")
        return

    summary = (
        df.groupby(
            [
                "candidate_class",
                "source_dataset",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="candidate_count"
        )
    )

    print()
    print("Selected source distribution:")
    print(
        summary.to_string(
            index=False
        )
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing frozen website collector over the "
            "large final external candidate pool with safe resume."
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--mode",
        choices=[
            "phishing",
            "legitimate",
            "all",
        ],
        default="phishing",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help=(
            "Number of candidates from the deterministic shuffled "
            "order. Use 0 to select every candidate in the mode."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output CSV. If omitted, a mode-specific file "
            "is created under data/processed/final_external_2026/crawled."
        ),
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Start a new output file even if one already exists. "
            "Normally existing output is resumed automatically."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.count < 0:
        raise ValueError(
            "--count must be 0 or greater."
        )

    input_path = Path(
        args.input
    )

    output_dir = Path(
        DEFAULT_OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.output:
        output_path = Path(
            args.output
        )
    else:
        output_path = (
            output_dir
            / f"final_external_{args.mode}_crawled.csv"
        )

    candidate_pool = load_candidate_pool(
        input_path
    )

    selected = select_mode(
        candidate_pool,
        args.mode,
    )

    # Fixed permutation means:
    # --count 1000, then later --count 5000 with the same seed,
    # uses the same first 1000 plus 4000 more.
    selected = deterministic_order(
        selected,
        args.seed,
    )

    if args.count > 0:
        selected = selected.head(
            args.count
        ).copy()

    print()
    print("=" * 88)
    print("FINAL EXTERNAL LARGE CRAWL")
    print("=" * 88)
    print(
        f"Candidate pool: {input_path}"
    )
    print(
        f"Mode: {args.mode}"
    )
    print(
        f"Selected candidates: {len(selected):,}"
    )
    print(
        f"Output: {output_path}"
    )

    print_source_summary(
        selected
    )

    if selected.empty:
        print()
        print(
            "Nothing to crawl."
        )
        return

    collector_input = to_collector_input(
        selected
    )

    # Safe default:
    # if an output already exists, resume it automatically.
    resume = (
        output_path.exists()
        and not args.fresh
    )

    print()
    print(
        "Resume existing output: "
        + (
            "YES"
            if resume
            else "NO"
        )
    )

    print()
    print(
        "The existing frozen crawler/collector is used unchanged."
    )
    print(
        "No forms are submitted and no credentials are entered."
    )
    print("=" * 88)

    collect_dataset(
        selected_samples=
            collector_input,

        output_file=
            output_path,

        resume=
            resume,
    )


if __name__ == "__main__":
    main()
