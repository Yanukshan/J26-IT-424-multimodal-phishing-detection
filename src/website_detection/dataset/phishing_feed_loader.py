import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PHISHING_LABEL = 0

SOURCE_DATASET = "PhishTank"

DEFAULT_INPUT = (
    "data/raw/"
    "phishtank_online_valid.csv"
)

DEFAULT_OUTPUT = (
    "data/processed/"
    "phishtank_phishing_urls.csv"
)

DEFAULT_COUNT = 100

# 0 = no domain limit.
DEFAULT_MAX_PER_DOMAIN = 0


# ============================================================
# EXPECTED PHISHTANK COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "phish_id",
    "url",
    "submission_time",
    "verified",
    "verification_time",
    "online",
}


OPTIONAL_COLUMNS = {
    "phish_detail_url",
    "target",
}


# ============================================================
# OUTPUT COLUMNS
# ============================================================

OUTPUT_COLUMNS = [
    # --------------------------------------------------------
    # Required by collector.py
    # --------------------------------------------------------
    "URL",
    "label",

    # --------------------------------------------------------
    # Research provenance
    # --------------------------------------------------------
    "source_dataset",
    "source_id",
    "source_domain",

    "source_verified",
    "source_online",

    "source_submission_time",
    "source_verification_time",

    "source_target",
    "source_detail_url",

    "source_loaded_at_utc",
]


# ============================================================
# BOOLEAN NORMALIZATION
# ============================================================

def is_yes(
    value: Any,
) -> bool:
    """
    Interpret common affirmative values.

    PhishTank normally uses:

        yes
    """

    if value is None:
        return False

    normalized = str(
        value
    ).strip().lower()

    return normalized in {
        "yes",
        "y",
        "true",
        "1",
    }


# ============================================================
# DOMAIN EXTRACTION
# ============================================================

def extract_domain(
    url: str,
) -> str:
    """
    Extract normalized source hostname.

    Example:

        https://www.example.com/login
            -> example.com

        https://login.example.com/page
            -> login.example.com

    This is intentionally hostname-level diversity for the
    current dataset collection stage.
    """

    if not url:
        return ""

    try:

        hostname = (
            urlparse(
                str(url).strip()
            ).hostname
            or ""
        ).lower()

        if hostname.startswith(
            "www."
        ):
            hostname = hostname[
                4:
            ]

        return hostname

    except Exception:
        return ""


# ============================================================
# URL VALIDATION
# ============================================================

def is_valid_web_url(
    url: str,
) -> bool:
    """
    Keep only HTTP/HTTPS URLs containing a hostname.

    This does not visit the URL.
    """

    if not url:
        return False

    try:

        parsed = urlparse(
            str(url).strip()
        )

        if parsed.scheme.lower() not in {
            "http",
            "https",
        }:
            return False

        if not parsed.hostname:
            return False

        return True

    except Exception:
        return False


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_column_names(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize source column names.
    """

    dataframe = (
        dataframe.copy()
    )

    dataframe.columns = [
        str(column)
        .strip()
        .lower()

        for column
        in dataframe.columns
    ]

    return dataframe


# ============================================================
# FEED VALIDATION
# ============================================================

def validate_columns(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate required PhishTank feed columns.
    """

    available = set(
        dataframe.columns
    )

    missing = (
        REQUIRED_COLUMNS
        - available
    )

    if missing:

        raise ValueError(
            "Missing required PhishTank "
            "column(s): "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )


# ============================================================
# LOAD PHISHTANK CSV
# ============================================================

def load_phishtank_csv(
    input_file: Path,
) -> pd.DataFrame:
    """
    Load the downloaded PhishTank CSV.

    This function does NOT visit phishing websites.
    """

    if not input_file.exists():

        raise FileNotFoundError(
            "PhishTank CSV not found: "
            f"{input_file}"
        )

    print()
    print(
        "=" * 70
    )

    print(
        "PHISHTANK FEED LOADER"
    )

    print(
        "=" * 70
    )

    print(
        f"Input: {input_file}"
    )

    dataframe = pd.read_csv(
        input_file,
        dtype=str,
        keep_default_na=False,
    )

    dataframe = (
        normalize_column_names(
            dataframe
        )
    )

    validate_columns(
        dataframe
    )

    print(
        f"Raw feed rows: "
        f"{len(dataframe):,}"
    )

    return dataframe


# ============================================================
# CLEAN / FILTER PHISHTANK FEED
# ============================================================

def clean_phishtank_feed(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare verified, online, valid PhishTank URLs.

    Important:
    PhishTank supplies the phishing label.

    Our crawler features do not generate or modify that label.
    """

    dataframe = (
        dataframe.copy()
    )

    # --------------------------------------------------------
    # Normalize URLs
    # --------------------------------------------------------

    dataframe[
        "url"
    ] = (
        dataframe[
            "url"
        ]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Empty URLs
    # --------------------------------------------------------

    before_empty = len(
        dataframe
    )

    dataframe = dataframe[
        dataframe[
            "url"
        ] != ""
    ].copy()

    removed_empty = (
        before_empty
        - len(
            dataframe
        )
    )

    # --------------------------------------------------------
    # Verified only
    # --------------------------------------------------------

    before_verified = len(
        dataframe
    )

    dataframe = dataframe[
        dataframe[
            "verified"
        ].apply(
            is_yes
        )
    ].copy()

    removed_unverified = (
        before_verified
        - len(
            dataframe
        )
    )

    # --------------------------------------------------------
    # Online only
    # --------------------------------------------------------

    before_online = len(
        dataframe
    )

    dataframe = dataframe[
        dataframe[
            "online"
        ].apply(
            is_yes
        )
    ].copy()

    removed_offline = (
        before_online
        - len(
            dataframe
        )
    )

    # --------------------------------------------------------
    # HTTP / HTTPS only
    # --------------------------------------------------------

    before_scheme = len(
        dataframe
    )

    dataframe = dataframe[
        dataframe[
            "url"
        ].apply(
            is_valid_web_url
        )
    ].copy()

    removed_invalid_urls = (
        before_scheme
        - len(
            dataframe
        )
    )

    # --------------------------------------------------------
    # Parse timestamps
    # --------------------------------------------------------

    dataframe[
        "_verification_datetime"
    ] = pd.to_datetime(
        dataframe[
            "verification_time"
        ],
        errors="coerce",
        utc=True,
    )

    dataframe[
        "_submission_datetime"
    ] = pd.to_datetime(
        dataframe[
            "submission_time"
        ],
        errors="coerce",
        utc=True,
    )

    # --------------------------------------------------------
    # Add normalized source domain
    # --------------------------------------------------------

    dataframe[
        "_source_domain"
    ] = dataframe[
        "url"
    ].apply(
        extract_domain
    )

    # Defensive cleanup.
    dataframe = dataframe[
        dataframe[
            "_source_domain"
        ] != ""
    ].copy()

    # --------------------------------------------------------
    # Newest verified records first
    # --------------------------------------------------------

    dataframe = dataframe.sort_values(
        by=[
            "_verification_datetime",
            "_submission_datetime",
        ],
        ascending=[
            False,
            False,
        ],
        na_position="last",
    )

    # --------------------------------------------------------
    # Remove exact URL duplicates.
    #
    # Because we sorted newest-first, the newest record for
    # that exact URL is retained.
    # --------------------------------------------------------

    before_duplicates = len(
        dataframe
    )

    dataframe = dataframe.drop_duplicates(
        subset=[
            "url"
        ],
        keep="first",
    ).copy()

    duplicate_count = (
        before_duplicates
        - len(
            dataframe
        )
    )

    dataframe = dataframe.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Cleaning summary
    # --------------------------------------------------------

    print()
    print(
        "Feed cleaning:"
    )

    print(
        f"  Empty URLs removed: "
        f"{removed_empty:,}"
    )

    print(
        f"  Unverified removed: "
        f"{removed_unverified:,}"
    )

    print(
        f"  Offline removed: "
        f"{removed_offline:,}"
    )

    print(
        f"  Invalid web URLs removed: "
        f"{removed_invalid_urls:,}"
    )

    print(
        f"  Duplicate URLs removed: "
        f"{duplicate_count:,}"
    )

    print(
        f"  Usable verified + online URLs: "
        f"{len(dataframe):,}"
    )

    print(
        f"  Unique source domains: "
        f"{dataframe['_source_domain'].nunique():,}"
    )

    return dataframe


# ============================================================
# DOMAIN-DIVERSE RECENT SELECTION
# ============================================================

def select_recent_diverse_records(
    dataframe: pd.DataFrame,
    count: int,
    max_per_domain: int,
) -> pd.DataFrame:
    """
    Select recent phishing URLs with an optional per-domain
    limit.

    Examples:

        count=500
        max_per_domain=2

    means:

        select at most 500 URLs
        with no hostname contributing more than 2 URLs.

    count=0
        keeps all eligible records after applying the
        domain cap.

    max_per_domain=0
        disables the domain cap.

    The input is already sorted newest-first, so the most
    recent entries from each domain are retained.
    """

    if count < 0:

        raise ValueError(
            "--count cannot be negative."
        )

    if max_per_domain < 0:

        raise ValueError(
            "--max-per-domain cannot be negative."
        )

    selected = (
        dataframe.copy()
    )

    # --------------------------------------------------------
    # Apply domain diversity cap
    # --------------------------------------------------------

    if max_per_domain > 0:

        # Since records are newest-first, cumcount=0 is the
        # newest record for that domain, cumcount=1 the next,
        # and so on.
        selected[
            "_domain_rank"
        ] = (
            selected.groupby(
                "_source_domain",
                sort=False,
            )
            .cumcount()
        )

        selected = selected[
            selected[
                "_domain_rank"
            ]
            < max_per_domain
        ].copy()

    else:

        selected[
            "_domain_rank"
        ] = 0

    eligible_after_domain_cap = len(
        selected
    )

    # --------------------------------------------------------
    # Apply requested overall count
    # --------------------------------------------------------

    if count > 0:

        selected = selected.head(
            count
        ).copy()

    selected = selected.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Selection statistics
    # --------------------------------------------------------

    print()
    print(
        "Domain-diverse selection:"
    )

    if max_per_domain > 0:

        print(
            f"  Max URLs per domain: "
            f"{max_per_domain}"
        )

    else:

        print(
            "  Max URLs per domain: "
            "unlimited"
        )

    print(
        "  Eligible after domain cap: "
        f"{eligible_after_domain_cap:,}"
    )

    print(
        f"  Selected URLs: "
        f"{len(selected):,}"
    )

    if not selected.empty:

        print(
            f"  Selected unique domains: "
            f"{selected['_source_domain'].nunique():,}"
        )

        maximum_domain_count = int(
            selected[
                "_source_domain"
            ]
            .value_counts()
            .max()
        )

        print(
            "  Maximum selected from one domain: "
            f"{maximum_domain_count}"
        )

    return selected


# ============================================================
# BUILD NORMALIZED OUTPUT
# ============================================================

def build_output_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert PhishTank source format into our collector input.

    Required collector fields:

        URL
        label

    Additional fields preserve research provenance.
    """

    loaded_at = datetime.now(
        timezone.utc
    ).isoformat()

    output = pd.DataFrame(
        index=dataframe.index
    )

    # --------------------------------------------------------
    # Collector fields
    # --------------------------------------------------------

    output[
        "URL"
    ] = dataframe[
        "url"
    ].values

    output[
        "label"
    ] = PHISHING_LABEL

    # --------------------------------------------------------
    # Source provenance
    # --------------------------------------------------------

    output[
        "source_dataset"
    ] = SOURCE_DATASET

    output[
        "source_id"
    ] = dataframe[
        "phish_id"
    ].values

    # Use the domain already calculated during filtering.
    if (
        "_source_domain"
        in dataframe.columns
    ):

        output[
            "source_domain"
        ] = dataframe[
            "_source_domain"
        ].values

    else:

        output[
            "source_domain"
        ] = dataframe[
            "url"
        ].apply(
            extract_domain
        ).values

    output[
        "source_verified"
    ] = dataframe[
        "verified"
    ].values

    output[
        "source_online"
    ] = dataframe[
        "online"
    ].values

    output[
        "source_submission_time"
    ] = dataframe[
        "submission_time"
    ].values

    output[
        "source_verification_time"
    ] = dataframe[
        "verification_time"
    ].values

    # --------------------------------------------------------
    # Optional source metadata
    # --------------------------------------------------------

    if (
        "target"
        in dataframe.columns
    ):

        output[
            "source_target"
        ] = dataframe[
            "target"
        ].values

    else:

        output[
            "source_target"
        ] = ""

    if (
        "phish_detail_url"
        in dataframe.columns
    ):

        output[
            "source_detail_url"
        ] = dataframe[
            "phish_detail_url"
        ].values

    else:

        output[
            "source_detail_url"
        ] = ""

    output[
        "source_loaded_at_utc"
    ] = loaded_at

    # --------------------------------------------------------
    # Stable column ordering
    # --------------------------------------------------------

    output = output[
        OUTPUT_COLUMNS
    ].reset_index(
        drop=True
    )

    return output


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_output(
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    Save normalized phishing feed.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )

    print()
    print(
        f"Output saved: "
        f"{output_file}"
    )


# ============================================================
# QA SUMMARY
# ============================================================

def print_summary(
    dataframe: pd.DataFrame,
    max_per_domain: int,
) -> None:
    """
    Display feed and diversity QA.
    """

    print()
    print(
        "=" * 70
    )

    print(
        "NORMALIZED PHISHING FEED"
    )

    print(
        "=" * 70
    )

    print(
        f"Selected phishing URLs: "
        f"{len(dataframe):,}"
    )

    if dataframe.empty:

        print(
            "No URLs available."
        )

        print(
            "=" * 70
        )

        return

    unique_urls = (
        dataframe[
            "URL"
        ].nunique()
    )

    unique_domains = (
        dataframe[
            "source_domain"
        ].nunique()
    )

    domain_counts = (
        dataframe[
            "source_domain"
        ].value_counts()
    )

    max_observed = int(
        domain_counts.max()
    )

    print(
        f"Unique URLs: "
        f"{unique_urls:,}"
    )

    print(
        f"Unique domains: "
        f"{unique_domains:,}"
    )

    print(
        "Maximum URLs from one domain: "
        f"{max_observed}"
    )

    if max_per_domain > 0:

        print(
            "Configured domain limit: "
            f"{max_per_domain}"
        )

    else:

        print(
            "Configured domain limit: "
            "unlimited"
        )

    print(
        "Label: "
        f"{PHISHING_LABEL} "
        "(phishing)"
    )

    print(
        f"Source: "
        f"{SOURCE_DATASET}"
    )

    # --------------------------------------------------------
    # Verification range
    # --------------------------------------------------------

    verification_times = pd.to_datetime(
        dataframe[
            "source_verification_time"
        ],
        errors="coerce",
        utc=True,
    )

    valid_times = (
        verification_times.dropna()
    )

    if not valid_times.empty:

        print(
            "Newest verification: "
            f"{valid_times.max()}"
        )

        print(
            "Oldest selected verification: "
            f"{valid_times.min()}"
        )

    # --------------------------------------------------------
    # Most represented domains
    # --------------------------------------------------------

    print()

    print(
        "Top selected domains:"
    )

    print(
        domain_counts
        .head(
            10
        )
        .to_string()
    )

    print(
        "=" * 70
    )


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Normalize and diversify the PhishTank "
            "online-valid feed for website phishing research."
        )
    )

    parser.add_argument(
        "--input",
        default=
            DEFAULT_INPUT,

        help=(
            "Path to downloaded PhishTank "
            "online-valid CSV."
        ),
    )

    parser.add_argument(
        "--output",
        default=
            DEFAULT_OUTPUT,

        help=(
            "Output normalized phishing CSV."
        ),
    )

    parser.add_argument(
        "--count",
        type=int,
        default=
            DEFAULT_COUNT,

        help=(
            "Maximum number of URLs to select. "
            "Use 0 to keep all eligible records."
        ),
    )

    parser.add_argument(
        "--max-per-domain",
        type=int,
        default=
            DEFAULT_MAX_PER_DOMAIN,

        help=(
            "Maximum URLs allowed from one normalized "
            "source hostname. Use 0 for no limit."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_arguments()

    # --------------------------------------------------------
    # Validate CLI parameters
    # --------------------------------------------------------

    if args.count < 0:

        raise ValueError(
            "--count cannot be negative."
        )

    if args.max_per_domain < 0:

        raise ValueError(
            "--max-per-domain cannot be negative."
        )

    input_file = Path(
        args.input
    )

    output_file = Path(
        args.output
    )

    # --------------------------------------------------------
    # Load feed
    # --------------------------------------------------------

    dataframe = (
        load_phishtank_csv(
            input_file
        )
    )

    # --------------------------------------------------------
    # Clean / sort feed
    # --------------------------------------------------------

    dataframe = (
        clean_phishtank_feed(
            dataframe
        )
    )

    # --------------------------------------------------------
    # NEW:
    # recent + domain-diverse selection
    # --------------------------------------------------------

    dataframe = (
        select_recent_diverse_records(
            dataframe=
                dataframe,

            count=
                args.count,

            max_per_domain=
                args.max_per_domain,
        )
    )

    # --------------------------------------------------------
    # Convert to collector-compatible format
    # --------------------------------------------------------

    output_dataframe = (
        build_output_dataframe(
            dataframe
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_output(
        dataframe=
            output_dataframe,

        output_file=
            output_file,
    )

    # --------------------------------------------------------
    # QA
    # --------------------------------------------------------

    print_summary(
        dataframe=
            output_dataframe,

        max_per_domain=
            args.max_per_domain,
    )


if __name__ == "__main__":
    main()