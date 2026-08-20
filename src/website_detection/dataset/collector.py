import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.website_detection.crawler.crawler import crawl_website


# ============================================================
# DATASET CONFIGURATION
# ============================================================

LABEL_PHISHING = 0
LABEL_LEGITIMATE = 1

DEFAULT_RANDOM_SEED = 42


# ============================================================
# CONTENT QUALITY CONFIGURATION
# ============================================================

# Extremely small rendered pages can indicate:
# - parking pages
# - placeholder pages
# - incomplete rendering
# - historical content drift
#
# This is only a REVIEW threshold.
# It is NOT a phishing threshold.
MINIMAL_HTML_LENGTH = 1500


# ============================================================
# DOM FEATURES
# ============================================================

DOM_FEATURES = [
    "links",
    "forms",
    "scripts",
    "images",
    "iframes",
    "inputs",
    "password_fields",
    "hidden_inputs",
    "external_links",
    "external_link_ratio",
    "javascript_links",
    "mailto_links",
    "external_scripts",
    "inline_scripts",
    "external_images",
    "external_iframes",
    "forms_without_action",
    "external_form_actions",
    "insecure_form_actions",
    "post_forms",
    "login_form_present",
]


# ============================================================
# NETWORK FEATURES
# ============================================================

NETWORK_FEATURES = [
    "total_requests",
    "get_requests",
    "post_requests",
    "put_requests",
    "delete_requests",
    "document_requests",
    "script_requests",
    "stylesheet_requests",
    "image_requests",
    "xhr_requests",
    "fetch_requests",
    "font_requests",
    "media_requests",
    "external_request_count",
    "external_request_ratio",
    "external_domain_count",
    "external_domains",
    "unique_domain_count",
    "insecure_http_requests",
]


# ============================================================
# GRAPH FEATURES
# ============================================================

GRAPH_FEATURES = [
    "node_count",
    "edge_count",
    "external_node_count",
    "external_node_ratio",
    "form_node_count",
    "form_target_node_count",
    "script_node_count",
    "iframe_node_count",
    "network_resource_nodes",
    "runtime_requested_nodes",
    "graph_density",
    "average_degree",
]


# ============================================================
# GRAPH OUTPUT COLUMN HELPER
# ============================================================

def get_graph_output_column(feature_name: str) -> str:
    """
    Prevent incorrect names such as:

        graph_graph_density

    Examples:

        node_count
        -> graph_node_count

        graph_density
        -> graph_density
    """

    if feature_name.startswith("graph_"):
        return feature_name

    return f"graph_{feature_name}"


# ============================================================
# BASE OUTPUT COLUMNS
# ============================================================

BASE_COLUMNS = [
    "sample_id",

    "source_url",
    "source_label",
    "class_name",

    "collected_at_utc",

    "crawl_status",

    "requested_url",
    "requested_domain",

    "final_url",
    "final_domain",

    "redirected",
    "cross_domain_redirect",
    "label_review_required",

    # --------------------------------------------------------
    # NEW DATASET QUALITY FIELDS
    # --------------------------------------------------------
    "content_review_required",
    "review_reasons",
    "training_candidate",

    "status_code",

    "title",
    "html_length",

    "error_type",
    "error",
]


# ============================================================
# COMPLETE OUTPUT COLUMNS
# ============================================================

OUTPUT_COLUMNS = (
    BASE_COLUMNS

    + [
        f"dom_{name}"
        for name in DOM_FEATURES
    ]

    + [
        f"net_{name}"
        for name in NETWORK_FEATURES
    ]

    + [
        get_graph_output_column(name)
        for name in GRAPH_FEATURES
    ]
)


# ============================================================
# LABEL HELPERS
# ============================================================

def get_class_name(label: int) -> str:
    """
    Convert dataset label into a readable class name.
    """

    if label == LABEL_LEGITIMATE:
        return "legitimate"

    if label == LABEL_PHISHING:
        return "phishing"

    return "unknown"


# ============================================================
# SAFE BOOLEAN HELPER
# ============================================================

def as_bool(value: Any) -> bool:
    """
    Safely convert common values into boolean.

    Prevents problems such as:

        bool("False") == True
    """

    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, (int, float)):
        return bool(value)

    value_text = str(value).strip().lower()

    return value_text in {
        "true",
        "1",
        "yes",
        "y",
    }


# ============================================================
# PARKING / LANDER DETECTION
# ============================================================

def detect_parking_page(
    title: str,
    final_url: str,
    external_domains: Any,
) -> bool:
    """
    Detect common signs that the current website has become
    a parking / domain holding / lander page.

    IMPORTANT:
    This does NOT classify a page as phishing.

    It only marks the sample for dataset-quality review.
    """

    title_lower = str(title or "").strip().lower()
    url_lower = str(final_url or "").strip().lower()

    # --------------------------------------------------------
    # Common parking-title indicators
    # --------------------------------------------------------

    parking_title_indicators = (
        "domain for sale",
        "this domain is for sale",
        "buy this domain",
        "domain parking",
        "parked domain",
        "domain parked",
    )

    if any(
        indicator in title_lower
        for indicator in parking_title_indicators
    ):
        return True

    # --------------------------------------------------------
    # Common lander URL
    # --------------------------------------------------------

    if (
        url_lower.endswith("/lander")
        or "/parking-lander" in url_lower
    ):
        return True

    # --------------------------------------------------------
    # Normalize external domain values
    # --------------------------------------------------------

    if isinstance(external_domains, list):
        domains = [
            str(domain).lower()
            for domain in external_domains
        ]

    elif external_domains:
        domains = [
            domain.strip().lower()
            for domain in str(external_domains).split("|")
            if domain.strip()
        ]

    else:
        domains = []

    # --------------------------------------------------------
    # Known parking infrastructure
    # --------------------------------------------------------

    parking_domain_indicators = (
        "parking.godaddy.com",
        "sedoparking.com",
        "sedo.com",
        "bodis.com",
        "parkingcrew.net",
    )

    for domain in domains:
        if any(
            indicator in domain
            for indicator in parking_domain_indicators
        ):
            return True

    return False


# ============================================================
# CONTENT QUALITY ASSESSMENT
# ============================================================

def assess_content_quality(
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Determine whether a collected record is suitable for
    automatic model training.

    This function performs DATASET QUALITY CONTROL only.

    It must never change:

        source_label

    and must never classify a website as phishing.
    """

    crawl_status = str(
        result.get(
            "crawl_status",
            "",
        )
    ).upper()

    # --------------------------------------------------------
    # Failed crawls are not training candidates
    # --------------------------------------------------------

    if crawl_status != "SUCCESS":
        return {
            "content_review_required": False,
            "review_reasons": "",
            "training_candidate": False,
        }

    reasons: list[str] = []

    # --------------------------------------------------------
    # Cross-domain historical drift
    # --------------------------------------------------------

    cross_domain_redirect = as_bool(
        result.get(
            "cross_domain_redirect",
            False,
        )
    )

    if cross_domain_redirect:
        reasons.append(
            "CROSS_DOMAIN_REDIRECT"
        )

    # --------------------------------------------------------
    # Current page information
    # --------------------------------------------------------

    title = str(
        result.get(
            "title",
            "",
        )
        or ""
    ).strip()

    final_url = str(
        result.get(
            "final_url",
            "",
        )
        or ""
    ).strip()

    html_length_raw = result.get(
        "html_length",
        0,
    )

    try:
        html_length = int(
            html_length_raw
            or 0
        )

    except (
        TypeError,
        ValueError,
    ):
        html_length = 0

    # --------------------------------------------------------
    # Minimal / empty rendered page
    # --------------------------------------------------------

    minimal_page = (
        html_length
        < MINIMAL_HTML_LENGTH
    )

    empty_title = (
        title == ""
    )

    # Be conservative:
    #
    # An empty title alone does not trigger review.
    # A small page alone does not automatically trigger review.
    #
    # Both occurring together are more useful evidence that
    # the historical URL may no longer represent normal content.
    if minimal_page and empty_title:

        reasons.append(
            "MINIMAL_PAGE"
        )

        reasons.append(
            "EMPTY_TITLE"
        )

    # --------------------------------------------------------
    # Parking / lander indicators
    # --------------------------------------------------------

    network_features = result.get(
        "network_features",
        {},
    )

    external_domains = (
        network_features.get(
            "external_domains",
            [],
        )
        if isinstance(
            network_features,
            dict,
        )
        else []
    )

    parking_page = detect_parking_page(
        title=title,
        final_url=final_url,
        external_domains=external_domains,
    )

    if parking_page:
        reasons.append(
            "PARKING_PAGE"
        )

    # --------------------------------------------------------
    # Remove duplicate reasons while preserving order
    # --------------------------------------------------------

    unique_reasons = list(
        dict.fromkeys(
            reasons
        )
    )

    content_review_required = (
        len(
            unique_reasons
        )
        > 0
    )

    training_candidate = (
        crawl_status == "SUCCESS"
        and not content_review_required
    )

    return {
        "content_review_required":
            content_review_required,

        "review_reasons":
            "|".join(
                unique_reasons
            ),

        "training_candidate":
            training_candidate,
    }


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    input_file: Path,
) -> pd.DataFrame:
    """
    Load URL and label from PhiUSIIL dataset.
    """

    if not input_file.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{input_file}"
        )

    print()
    print("Loading dataset...")

    print(
        f"File: {input_file}"
    )

    dataframe = pd.read_csv(
        input_file,
        usecols=[
            "URL",
            "label",
        ],
    )

    print(
        f"Original rows: "
        f"{len(dataframe):,}"
    )

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    dataframe = dataframe.dropna(
        subset=[
            "URL",
            "label",
        ]
    ).copy()

    # --------------------------------------------------------
    # Normalize URL strings
    # --------------------------------------------------------

    dataframe[
        "URL"
    ] = (
        dataframe[
            "URL"
        ]
        .astype(str)
        .str.strip()
    )

    dataframe = dataframe[
        dataframe[
            "URL"
        ] != ""
    ].copy()

    # --------------------------------------------------------
    # Normalize labels
    # --------------------------------------------------------

    dataframe[
        "label"
    ] = pd.to_numeric(
        dataframe[
            "label"
        ],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=[
            "label"
        ]
    ).copy()

    dataframe[
        "label"
    ] = (
        dataframe[
            "label"
        ].astype(int)
    )

    # --------------------------------------------------------
    # Valid labels only
    # --------------------------------------------------------

    dataframe = dataframe[
        dataframe[
            "label"
        ].isin(
            [
                LABEL_PHISHING,
                LABEL_LEGITIMATE,
            ]
        )
    ].copy()

    # --------------------------------------------------------
    # Remove duplicate URLs
    # --------------------------------------------------------

    before_duplicates = len(
        dataframe
    )

    dataframe = (
        dataframe
        .drop_duplicates(
            subset=[
                "URL"
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    removed_duplicates = (
        before_duplicates
        - len(
            dataframe
        )
    )

    print(
        f"Removed duplicate URLs: "
        f"{removed_duplicates:,}"
    )

    print(
        f"Usable unique URLs: "
        f"{len(dataframe):,}"
    )

    legitimate_count = len(
        dataframe[
            dataframe[
                "label"
            ]
            == LABEL_LEGITIMATE
        ]
    )

    phishing_count = len(
        dataframe[
            dataframe[
                "label"
            ]
            == LABEL_PHISHING
        ]
    )

    print()

    print(
        f"Legitimate URLs: "
        f"{legitimate_count:,}"
    )

    print(
        f"Phishing URLs: "
        f"{phishing_count:,}"
    )

    return dataframe


# ============================================================
# SAMPLE SELECTION
# ============================================================

def select_samples(
    dataframe: pd.DataFrame,
    mode: str,
    count: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Select legitimate, phishing or balanced samples.

    Balanced mode:
        --count 20

    means:
        20 legitimate
        20 phishing
        40 total
    """

    legitimate = dataframe[
        dataframe[
            "label"
        ]
        == LABEL_LEGITIMATE
    ]

    phishing = dataframe[
        dataframe[
            "label"
        ]
        == LABEL_PHISHING
    ]

    # --------------------------------------------------------
    # Legitimate
    # --------------------------------------------------------

    if mode == "legitimate":

        sample_count = min(
            count,
            len(
                legitimate
            ),
        )

        selected = legitimate.sample(
            n=sample_count,
            random_state=
                random_seed,
        )

    # --------------------------------------------------------
    # Phishing
    # --------------------------------------------------------

    elif mode == "phishing":

        sample_count = min(
            count,
            len(
                phishing
            ),
        )

        selected = phishing.sample(
            n=sample_count,
            random_state=
                random_seed,
        )

    # --------------------------------------------------------
    # Balanced
    # --------------------------------------------------------

    elif mode == "balanced":

        legitimate_count = min(
            count,
            len(
                legitimate
            ),
        )

        phishing_count = min(
            count,
            len(
                phishing
            ),
        )

        legitimate_sample = (
            legitimate.sample(
                n=legitimate_count,
                random_state=
                    random_seed,
            )
        )

        phishing_sample = (
            phishing.sample(
                n=phishing_count,
                random_state=
                    random_seed,
            )
        )

        selected = pd.concat(
            [
                legitimate_sample,
                phishing_sample,
            ],
            ignore_index=True,
        )

    else:

        raise ValueError(
            f"Unsupported mode: "
            f"{mode}"
        )

    # --------------------------------------------------------
    # Shuffle selected records
    # --------------------------------------------------------

    selected = (
        selected.sample(
            frac=1,
            random_state=
                random_seed,
        )
        .reset_index(
            drop=True
        )
    )

    return selected


# ============================================================
# FLATTEN RESULT
# ============================================================

def flatten_result(
    sample_id: int,
    source_url: str,
    label: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert crawler result into one flat CSV row.
    """

    row: dict[str, Any] = {
        column: ""
        for column in OUTPUT_COLUMNS
    }

    # --------------------------------------------------------
    # Dataset source metadata
    # --------------------------------------------------------

    row[
        "sample_id"
    ] = sample_id

    row[
        "source_url"
    ] = source_url

    row[
        "source_label"
    ] = label

    row[
        "class_name"
    ] = get_class_name(
        label
    )

    row[
        "collected_at_utc"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    # --------------------------------------------------------
    # Crawl metadata
    # --------------------------------------------------------

    row[
        "crawl_status"
    ] = result.get(
        "crawl_status",
        "",
    )

    row[
        "requested_url"
    ] = result.get(
        "requested_url",
        "",
    )

    row[
        "requested_domain"
    ] = result.get(
        "requested_domain",
        "",
    )

    row[
        "final_url"
    ] = result.get(
        "final_url",
        "",
    )

    row[
        "final_domain"
    ] = result.get(
        "final_domain",
        "",
    )

    row[
        "redirected"
    ] = result.get(
        "redirected",
        "",
    )

    row[
        "cross_domain_redirect"
    ] = result.get(
        "cross_domain_redirect",
        "",
    )

    row[
        "label_review_required"
    ] = result.get(
        "label_review_required",
        "",
    )

    # --------------------------------------------------------
    # NEW CONTENT QUALITY ASSESSMENT
    # --------------------------------------------------------

    quality = assess_content_quality(
        result
    )

    row[
        "content_review_required"
    ] = quality[
        "content_review_required"
    ]

    row[
        "review_reasons"
    ] = quality[
        "review_reasons"
    ]

    row[
        "training_candidate"
    ] = quality[
        "training_candidate"
    ]

    # --------------------------------------------------------
    # Response information
    # --------------------------------------------------------

    row[
        "status_code"
    ] = result.get(
        "status_code",
        "",
    )

    row[
        "title"
    ] = result.get(
        "title",
        "",
    )

    row[
        "html_length"
    ] = result.get(
        "html_length",
        "",
    )

    row[
        "error_type"
    ] = result.get(
        "error_type",
        "",
    )

    row[
        "error"
    ] = result.get(
        "error",
        "",
    )

    # ========================================================
    # DOM FEATURES
    # ========================================================

    dom_features = result.get(
        "dom_features",
        {},
    )

    for feature in DOM_FEATURES:

        row[
            f"dom_{feature}"
        ] = dom_features.get(
            feature,
            "",
        )

    # ========================================================
    # NETWORK FEATURES
    # ========================================================

    network_features = result.get(
        "network_features",
        {},
    )

    for feature in NETWORK_FEATURES:

        value = network_features.get(
            feature,
            "",
        )

        # ----------------------------------------------------
        # Convert domain list into CSV-friendly string
        # ----------------------------------------------------

        if isinstance(
            value,
            list,
        ):

            value = "|".join(
                str(item)
                for item in value
            )

        row[
            f"net_{feature}"
        ] = value

    # ========================================================
    # GRAPH FEATURES
    # ========================================================

    graph_features = result.get(
        "graph_features",
        {},
    )

    for feature in GRAPH_FEATURES:

        output_column = (
            get_graph_output_column(
                feature
            )
        )

        row[
            output_column
        ] = graph_features.get(
            feature,
            "",
        )

    return row


# ============================================================
# INITIALIZE OUTPUT CSV
# ============================================================

def initialize_output_file(
    output_file: Path,
) -> None:
    """
    Create output CSV and write the header.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                OUTPUT_COLUMNS,
        )

        writer.writeheader()


# ============================================================
# APPEND ONE RESULT
# ============================================================

def append_result(
    output_file: Path,
    row: dict[str, Any],
) -> None:
    """
    Append immediately so results already collected are not
    lost if the collection is interrupted.
    """

    with output_file.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                OUTPUT_COLUMNS,
        )

        writer.writerow(
            row
        )


# ============================================================
# DATA COLLECTION
# ============================================================

def collect_dataset(
    selected_samples: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    Crawl selected URLs and persist feature records.
    """

    initialize_output_file(
        output_file
    )

    total = len(
        selected_samples
    )

    success_count = 0
    failure_count = 0

    label_review_count = 0
    content_review_count = 0
    training_candidate_count = 0

    print()
    print(
        "=" * 60
    )

    print(
        "WEBSITE DATASET COLLECTION"
    )

    print(
        "=" * 60
    )

    print(
        f"Samples selected: "
        f"{total}"
    )

    print(
        f"Output: "
        f"{output_file}"
    )

    print(
        "=" * 60
    )

    print()

    # ========================================================
    # PROCESS EACH SAMPLE
    # ========================================================

    for index, sample in (
        selected_samples.iterrows()
    ):

        sample_id = (
            index + 1
        )

        url = str(
            sample[
                "URL"
            ]
        )

        label = int(
            sample[
                "label"
            ]
        )

        class_name = (
            get_class_name(
                label
            )
        )

        print(
            f"[{sample_id}/{total}] "
            f"{class_name.upper()}"
        )

        print(
            f"URL: {url}"
        )

        try:

            # ------------------------------------------------
            # Crawl website
            # ------------------------------------------------

            result = crawl_website(
                url
            )

            status = str(
                result.get(
                    "crawl_status",
                    "UNKNOWN",
                )
            )

            # ------------------------------------------------
            # Flatten + quality assessment
            # ------------------------------------------------

            row = flatten_result(
                sample_id=
                    sample_id,

                source_url=
                    url,

                label=
                    label,

                result=
                    result,
            )

            # ------------------------------------------------
            # Save immediately
            # ------------------------------------------------

            append_result(
                output_file,
                row,
            )

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            if status == "SUCCESS":
                success_count += 1

            else:
                failure_count += 1

            if as_bool(
                result.get(
                    "label_review_required",
                    False,
                )
            ):

                label_review_count += 1

            if as_bool(
                row.get(
                    "content_review_required",
                    False,
                )
            ):

                content_review_count += 1

            if as_bool(
                row.get(
                    "training_candidate",
                    False,
                )
            ):

                training_candidate_count += 1

            # ------------------------------------------------
            # Console status
            # ------------------------------------------------

            print(
                f"Status: {status}"
            )

            if as_bool(
                result.get(
                    "cross_domain_redirect",
                    False,
                )
            ):

                print(
                    "Cross-domain redirect: "
                    f"{result.get('requested_domain', '')}"
                    " -> "
                    f"{result.get('final_domain', '')}"
                )

            if as_bool(
                row.get(
                    "content_review_required",
                    False,
                )
            ):

                print(
                    "Content review: REQUIRED"
                )

                print(
                    "Review reasons: "
                    f"{row.get('review_reasons', '')}"
                )

            if as_bool(
                row.get(
                    "training_candidate",
                    False,
                )
            ):

                print(
                    "Training candidate: YES"
                )

            else:

                print(
                    "Training candidate: NO"
                )

        # ====================================================
        # USER INTERRUPT
        # ====================================================

        except KeyboardInterrupt:

            print()
            print(
                "Collection stopped by user."
            )

            print(
                "Already collected results "
                "remain saved."
            )

            break

        # ====================================================
        # COLLECTOR-LEVEL FAILURE
        # ====================================================

        except Exception as exc:

            failure_count += 1

            result = {
                "crawl_status":
                    "COLLECTOR_ERROR",

                "requested_url":
                    url,

                "error_type":
                    type(
                        exc
                    ).__name__,

                "error":
                    str(
                        exc
                    ),
            }

            row = flatten_result(
                sample_id=
                    sample_id,

                source_url=
                    url,

                label=
                    label,

                result=
                    result,
            )

            append_result(
                output_file,
                row,
            )

            print(
                f"Collector error: "
                f"{exc}"
            )

            print(
                "Training candidate: NO"
            )

        print(
            "-" * 60
        )

    # ========================================================
    # FINAL COLLECTION SUMMARY
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        "COLLECTION FINISHED"
    )

    print(
        "=" * 60
    )

    print(
        f"Successful crawls: "
        f"{success_count}"
    )

    print(
        f"Other/failed crawls: "
        f"{failure_count}"
    )

    print(
        f"Label reviews required: "
        f"{label_review_count}"
    )

    print(
        f"Content reviews required: "
        f"{content_review_count}"
    )

    print(
        f"Training candidates: "
        f"{training_candidate_count}"
    )

    print(
        f"Output saved to: "
        f"{output_file}"
    )

    print(
        "=" * 60
    )


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line configuration.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Collect dynamic website phishing "
            "research features."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "data/raw/"
            "PhiUSIIL_Phishing_URL_Dataset.csv"
        ),
        help=(
            "Path to PhiUSIIL dataset CSV."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/processed/"
            "website_pilot_dataset.csv"
        ),
        help=(
            "Output CSV path."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "legitimate",
            "phishing",
            "balanced",
        ],
        default="legitimate",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=
            DEFAULT_RANDOM_SEED,
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Dataset collector command-line entry point.
    """

    args = parse_arguments()

    if args.count <= 0:

        raise ValueError(
            "--count must be "
            "greater than zero."
        )

    # --------------------------------------------------------
    # Load source URLs
    # --------------------------------------------------------

    dataframe = load_dataset(
        Path(
            args.input
        )
    )

    # --------------------------------------------------------
    # Select sample
    # --------------------------------------------------------

    selected_samples = select_samples(
        dataframe=
            dataframe,

        mode=
            args.mode,

        count=
            args.count,

        random_seed=
            args.seed,
    )

    # --------------------------------------------------------
    # Print distribution
    # --------------------------------------------------------

    print()

    print(
        "Selected sample distribution:"
    )

    distribution = (
        selected_samples[
            "label"
        ]
        .value_counts()
        .sort_index()
    )

    for label, count in (
        distribution.items()
    ):

        print(
            f"  "
            f"{get_class_name(int(label))}: "
            f"{count}"
        )

    # --------------------------------------------------------
    # Start collection
    # --------------------------------------------------------

    collect_dataset(
        selected_samples=
            selected_samples,

        output_file=
            Path(
                args.output
            ),
    )


if __name__ == "__main__":
    main()