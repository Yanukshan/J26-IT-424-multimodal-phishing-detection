from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


LABEL_LEGITIMATE = 1


def normalize_domain(value: str) -> str:
    domain = str(value or "").strip().lower().rstrip(".")

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def build_url(domain: str) -> str:
    """
    Build a crawlable HTTPS URL from a ranked Tranco domain.
    """
    domain = normalize_domain(domain)

    if not domain:
        return ""

    return f"https://{domain}"


def load_tranco(
    input_file: Path,
) -> pd.DataFrame:
    """
    Load standard Tranco CSV.

    Expected format:
        rank,domain

    The standard downloadable Tranco file normally has no
    header, so we read it explicitly as two columns.
    """

    if not input_file.exists():
        raise FileNotFoundError(
            f"Tranco file not found: {input_file}"
        )

    dataframe = pd.read_csv(
        input_file,
        header=None,
        names=[
            "rank",
            "domain",
        ],
        usecols=[
            0,
            1,
        ],
        dtype={
            "rank": "Int64",
            "domain": str,
        },
        keep_default_na=False,
    )

    dataframe[
        "domain"
    ] = (
        dataframe[
            "domain"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    dataframe = dataframe[
        dataframe[
            "domain"
        ].ne("")
    ].copy()

    dataframe = dataframe.drop_duplicates(
        subset=[
            "domain",
        ],
        keep="first",
    ).reset_index(
        drop=True
    )

    return dataframe


def select_domains(
    dataframe: pd.DataFrame,
    count: int,
    start_rank: int,
    end_rank: int,
    seed: int,
) -> pd.DataFrame:
    """
    Select a random domain sample from a rank window.

    Using a window avoids relying only on the very top global
    sites while still selecting established popular domains.
    """

    selected = dataframe.copy()

    if start_rank > 0:

        selected = selected[
            pd.to_numeric(
                selected[
                    "rank"
                ],
                errors="coerce",
            )
            >= start_rank
        ].copy()

    if end_rank > 0:

        selected = selected[
            pd.to_numeric(
                selected[
                    "rank"
                ],
                errors="coerce",
            )
            <= end_rank
        ].copy()

    if selected.empty:
        raise ValueError(
            "No Tranco rows remain inside the selected rank window."
        )

    if count > 0:

        sample_count = min(
            count,
            len(
                selected
            ),
        )

        selected = selected.sample(
            n=sample_count,
            random_state=seed,
        )

    return selected.reset_index(
        drop=True
    )


def build_output(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    loaded_at = datetime.now(
        timezone.utc
    ).isoformat()

    output = pd.DataFrame()

    output[
        "URL"
    ] = dataframe[
        "domain"
    ].map(
        build_url
    )

    output[
        "label"
    ] = LABEL_LEGITIMATE

    output[
        "source_dataset"
    ] = "Tranco"

    output[
        "source_id"
    ] = dataframe[
        "rank"
    ].astype(str)

    output[
        "source_domain"
    ] = dataframe[
        "domain"
    ].map(
        normalize_domain
    )

    output[
        "source_verified"
    ] = ""

    output[
        "source_online"
    ] = ""

    output[
        "source_submission_time"
    ] = ""

    output[
        "source_verification_time"
    ] = ""

    output[
        "source_target"
    ] = ""

    output[
        "source_detail_url"
    ] = ""

    output[
        "source_loaded_at_utc"
    ] = loaded_at

    output = output[
        output[
            "URL"
        ].ne("")
    ].copy()

    return output.reset_index(
        drop=True
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize and sample Tranco domains for legitimate "
            "website phishing research."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help=(
            "Maximum domains to select. Use 0 to keep all rows "
            "inside the rank window."
        ),
    )

    parser.add_argument(
        "--start-rank",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--end-rank",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    dataframe = load_tranco(
        Path(
            args.input
        )
    )

    selected = select_domains(
        dataframe=
            dataframe,

        count=
            args.count,

        start_rank=
            args.start_rank,

        end_rank=
            args.end_rank,

        seed=
            args.seed,
    )

    output = build_output(
        selected
    )

    output_file = Path(
        args.output
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        "=" * 70
    )
    print(
        "TRANCO LEGITIMATE LOADER"
    )
    print(
        "=" * 70
    )
    print(
        f"Loaded unique domains: {len(dataframe):,}"
    )
    print(
        f"Selected rows: {len(output):,}"
    )
    print(
        f"Unique URLs: {output['URL'].nunique():,}"
    )
    print(
        f"Rank window: {args.start_rank:,} - {args.end_rank:,}"
    )
    print(
        f"Output: {output_file}"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
