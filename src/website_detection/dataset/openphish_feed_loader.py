from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


LABEL_PHISHING = 0


def normalize_hostname(
    url: str,
) -> str:
    value = str(
        url or ""
    ).strip()

    if not value:
        return ""

    if not value.lower().startswith(
        (
            "http://",
            "https://",
        )
    ):
        value = (
            "https://"
            + value
        )

    try:
        hostname = (
            urlparse(
                value
            ).hostname
            or ""
        ).lower().rstrip(".")

        if hostname.startswith(
            "www."
        ):
            hostname = hostname[
                4:
            ]

        return hostname

    except Exception:
        return ""


def clean_url(
    value: str,
) -> str:
    url = str(
        value or ""
    ).strip()

    if not url:
        return ""

    # Remove accidental spreadsheet whitespace/newlines only.
    url = url.replace(
        "\r",
        ""
    ).replace(
        "\n",
        ""
    ).strip()

    if not url.lower().startswith(
        (
            "http://",
            "https://",
        )
    ):
        return ""

    return url


def load_openphish(
    input_file: Path,
) -> pd.DataFrame:
    """
    Load OpenPhish from .xlsx, .csv or .txt.

    The uploaded workbook format is a single URL column with
    no header.
    """

    if not input_file.exists():
        raise FileNotFoundError(
            f"OpenPhish file not found: {input_file}"
        )

    suffix = input_file.suffix.lower()

    if suffix in {
        ".xlsx",
        ".xls",
    }:

        dataframe = pd.read_excel(
            input_file,
            header=None,
            usecols=[
                0,
            ],
            names=[
                "url",
            ],
            dtype=str,
        )

    elif suffix == ".csv":

        dataframe = pd.read_csv(
            input_file,
            header=None,
            usecols=[
                0,
            ],
            names=[
                "url",
            ],
            dtype=str,
            keep_default_na=False,
        )

    else:

        urls = input_file.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()

        dataframe = pd.DataFrame(
            {
                "url":
                    urls,
            }
        )

    dataframe[
        "url"
    ] = dataframe[
        "url"
    ].map(
        clean_url
    )

    dataframe = dataframe[
        dataframe[
            "url"
        ].ne("")
    ].copy()

    dataframe[
        "source_domain"
    ] = dataframe[
        "url"
    ].map(
        normalize_hostname
    )

    dataframe = dataframe[
        dataframe[
            "source_domain"
        ].ne("")
    ].copy()

    dataframe = dataframe.drop_duplicates(
        subset=[
            "url",
        ],
        keep="first",
    ).reset_index(
        drop=True
    )

    return dataframe


def diversify(
    dataframe: pd.DataFrame,
    count: int,
    max_per_domain: int,
    seed: int,
) -> pd.DataFrame:
    """
    Select a domain-diverse OpenPhish sample.
    """

    shuffled = dataframe.sample(
        frac=1,
        random_state=seed,
    ).reset_index(
        drop=True
    )

    if max_per_domain > 0:

        shuffled = (
            shuffled.groupby(
                "source_domain",
                group_keys=False,
            )
            .head(
                max_per_domain
            )
            .reset_index(
                drop=True
            )
        )

    if count > 0:

        shuffled = shuffled.head(
            count
        ).copy()

    return shuffled.reset_index(
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
        "url"
    ]

    output[
        "label"
    ] = LABEL_PHISHING

    output[
        "source_dataset"
    ] = "OpenPhish"

    output[
        "source_id"
    ] = [
        f"openphish_{index + 1}"
        for index
        in range(
            len(
                dataframe
            )
        )
    ]

    output[
        "source_domain"
    ] = dataframe[
        "source_domain"
    ]

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

    return output.reset_index(
        drop=True
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize and diversify OpenPhish URLs for website "
            "phishing research."
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
        default=0,
        help=(
            "Maximum URLs to keep. Use 0 to keep all eligible "
            "rows after domain limiting."
        ),
    )

    parser.add_argument(
        "--max-per-domain",
        type=int,
        default=2,
        help=(
            "Maximum URLs from one normalized hostname. "
            "Use 0 for no limit."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    dataframe = load_openphish(
        Path(
            args.input
        )
    )

    selected = diversify(
        dataframe=
            dataframe,

        count=
            args.count,

        max_per_domain=
            args.max_per_domain,

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
        "OPENPHISH FEED LOADER"
    )
    print(
        "=" * 70
    )
    print(
        f"Loaded unique URLs: {len(dataframe):,}"
    )
    print(
        f"Loaded unique domains: "
        f"{dataframe['source_domain'].nunique():,}"
    )
    print(
        f"Selected rows: {len(output):,}"
    )
    print(
        f"Selected unique domains: "
        f"{output['source_domain'].nunique():,}"
    )
    print(
        f"Maximum per domain: {args.max_per_domain}"
    )
    print(
        f"Output: {output_file}"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
