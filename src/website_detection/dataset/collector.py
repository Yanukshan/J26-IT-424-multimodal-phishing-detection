import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.website_detection.crawler.crawler import (
    crawl_website,
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

LABEL_PHISHING = 0
LABEL_LEGITIMATE = 1

DEFAULT_RANDOM_SEED = 42


# ============================================================
# DOM FEATURE COLUMNS
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
# NETWORK FEATURE COLUMNS
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
# GRAPH FEATURE COLUMNS
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
# GRAPH COLUMN NAME HELPER
# ============================================================

def get_graph_output_column(
    feature_name: str,
) -> str:
    """
    Create a clean output column name for graph features.

    Examples:
        node_count
            -> graph_node_count

        edge_count
            -> graph_edge_count

        graph_density
            -> graph_density

    This prevents:
        graph_graph_density
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
    "final_url",
    "redirected",
    "status_code",
    "title",
    "html_length",

    "error_type",
    "error",
]


# ============================================================
# COMPLETE CSV COLUMN LIST
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

def get_class_name(
    label: int,
) -> str:
    """
    Convert PhiUSIIL numeric labels into readable classes.

    PhiUSIIL mapping:
        1 = legitimate
        0 = phishing
    """

    if label == LABEL_LEGITIMATE:
        return "legitimate"

    if label == LABEL_PHISHING:
        return "phishing"

    return "unknown"


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    input_file: Path,
) -> pd.DataFrame:
    """
    Load URL and label columns from the PhiUSIIL dataset.

    Processing:
        1. Read URL + label only
        2. Remove missing values
        3. Clean URLs
        4. Validate labels
        5. Remove duplicate URLs
    """

    if not input_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_file}"
        )

    print()
    print("Loading dataset...")
    print(f"File: {input_file}")

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
    # Remove missing URLs / labels
    # --------------------------------------------------------

    dataframe = dataframe.dropna(
        subset=[
            "URL",
            "label",
        ]
    ).copy()

    # --------------------------------------------------------
    # Normalize URLs
    # --------------------------------------------------------

    dataframe["URL"] = (
        dataframe["URL"]
        .astype(str)
        .str.strip()
    )

    dataframe = dataframe[
        dataframe["URL"] != ""
    ].copy()

    # --------------------------------------------------------
    # Convert labels to numeric
    # --------------------------------------------------------

    dataframe["label"] = pd.to_numeric(
        dataframe["label"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=["label"]
    ).copy()

    dataframe["label"] = (
        dataframe["label"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Keep only valid PhiUSIIL labels
    # --------------------------------------------------------

    dataframe = dataframe[
        dataframe["label"].isin(
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
            subset=["URL"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    removed_duplicates = (
        before_duplicates
        - len(dataframe)
    )

    print(
        f"Removed duplicate URLs: "
        f"{removed_duplicates:,}"
    )

    print(
        f"Usable unique URLs: "
        f"{len(dataframe):,}"
    )

    print()

    legitimate_count = len(
        dataframe[
            dataframe["label"]
            == LABEL_LEGITIMATE
        ]
    )

    phishing_count = len(
        dataframe[
            dataframe["label"]
            == LABEL_PHISHING
        ]
    )

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
    Select a controlled sample from the dataset.

    Available modes:
        legitimate
        phishing
        balanced

    For balanced mode:
        --count 20

    means:
        20 legitimate
        20 phishing
        40 total
    """

    legitimate = dataframe[
        dataframe["label"]
        == LABEL_LEGITIMATE
    ]

    phishing = dataframe[
        dataframe["label"]
        == LABEL_PHISHING
    ]

    # --------------------------------------------------------
    # Legitimate-only mode
    # --------------------------------------------------------

    if mode == "legitimate":

        sample_count = min(
            count,
            len(legitimate),
        )

        selected = legitimate.sample(
            n=sample_count,
            random_state=random_seed,
        )

    # --------------------------------------------------------
    # Phishing-only mode
    # --------------------------------------------------------

    elif mode == "phishing":

        sample_count = min(
            count,
            len(phishing),
        )

        selected = phishing.sample(
            n=sample_count,
            random_state=random_seed,
        )

    # --------------------------------------------------------
    # Balanced mode
    # --------------------------------------------------------

    elif mode == "balanced":

        legitimate_count = min(
            count,
            len(legitimate),
        )

        phishing_count = min(
            count,
            len(phishing),
        )

        legitimate_sample = (
            legitimate.sample(
                n=legitimate_count,
                random_state=random_seed,
            )
        )

        phishing_sample = (
            phishing.sample(
                n=phishing_count,
                random_state=random_seed,
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
            f"Unsupported mode: {mode}"
        )

    # --------------------------------------------------------
    # Shuffle selected records
    # --------------------------------------------------------

    selected = selected.sample(
        frac=1,
        random_state=random_seed,
    ).reset_index(drop=True)

    return selected


# ============================================================
# FLATTEN CRAWLER RESULT
# ============================================================

def flatten_result(
    sample_id: int,
    source_url: str,
    label: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert nested crawler results into one flat CSV row.
    """

    row: dict[str, Any] = {
        column: ""
        for column in OUTPUT_COLUMNS
    }

    # --------------------------------------------------------
    # Source information
    # --------------------------------------------------------

    row["sample_id"] = sample_id

    row["source_url"] = (
        source_url
    )

    row["source_label"] = (
        label
    )

    row["class_name"] = (
        get_class_name(label)
    )

    row["collected_at_utc"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    # --------------------------------------------------------
    # Crawler metadata
    # --------------------------------------------------------

    row["crawl_status"] = result.get(
        "crawl_status",
        "",
    )

    row["requested_url"] = result.get(
        "requested_url",
        "",
    )

    row["final_url"] = result.get(
        "final_url",
        "",
    )

    row["redirected"] = result.get(
        "redirected",
        "",
    )

    row["status_code"] = result.get(
        "status_code",
        "",
    )

    row["title"] = result.get(
        "title",
        "",
    )

    row["html_length"] = result.get(
        "html_length",
        "",
    )

    row["error_type"] = result.get(
        "error_type",
        "",
    )

    row["error"] = result.get(
        "error",
        "",
    )

    # --------------------------------------------------------
    # DOM features
    # --------------------------------------------------------

    dom_features = result.get(
        "dom_features",
        {},
    )

    for feature in DOM_FEATURES:

        output_column = (
            f"dom_{feature}"
        )

        row[
            output_column
        ] = dom_features.get(
            feature,
            "",
        )

    # --------------------------------------------------------
    # Network features
    # --------------------------------------------------------

    network_features = result.get(
        "network_features",
        {},
    )

    for feature in NETWORK_FEATURES:

        value = network_features.get(
            feature,
            "",
        )

        # Lists cannot be stored directly in a normal CSV cell.
        # Convert them to pipe-separated text.
        if isinstance(
            value,
            list,
        ):
            value = "|".join(
                str(item)
                for item in value
            )

        output_column = (
            f"net_{feature}"
        )

        row[
            output_column
        ] = value

    # --------------------------------------------------------
    # Graph features
    # --------------------------------------------------------

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
# INITIALIZE CSV
# ============================================================

def initialize_output_file(
    output_file: Path,
) -> None:
    """
    Create the result CSV and write its header.
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
            fieldnames=OUTPUT_COLUMNS,
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
    Immediately append one processed website to the CSV.

    This prevents an entire run from being lost if a later
    website crashes, times out or cannot be reached.
    """

    with output_file.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_COLUMNS,
        )

        writer.writerow(
            row
        )


# ============================================================
# DATASET COLLECTION
# ============================================================

def collect_dataset(
    selected_samples: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    Crawl all selected URLs and save their features.
    """

    initialize_output_file(
        output_file
    )

    total = len(
        selected_samples
    )

    success_count = 0
    failure_count = 0

    print()
    print("=" * 60)
    print(
        "WEBSITE DATASET COLLECTION"
    )
    print("=" * 60)

    print(
        f"Samples selected: {total}"
    )

    print(
        f"Output: {output_file}"
    )

    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Process websites one by one
    # --------------------------------------------------------

    for index, sample in (
        selected_samples.iterrows()
    ):

        sample_id = (
            index + 1
        )

        url = str(
            sample["URL"]
        )

        label = int(
            sample["label"]
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
            # Run complete website-analysis pipeline
            # ------------------------------------------------

            result = crawl_website(
                url
            )

            status = result.get(
                "crawl_status",
                "UNKNOWN",
            )

            # ------------------------------------------------
            # Convert nested result into CSV row
            # ------------------------------------------------

            row = flatten_result(
                sample_id=sample_id,
                source_url=url,
                label=label,
                result=result,
            )

            # ------------------------------------------------
            # Save immediately
            # ------------------------------------------------

            append_result(
                output_file,
                row,
            )

            if status == "SUCCESS":
                success_count += 1

            else:
                failure_count += 1

            print(
                f"Status: {status}"
            )

        # ----------------------------------------------------
        # Manual interruption
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Unexpected collector-level error
        # ----------------------------------------------------

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
                sample_id=sample_id,
                source_url=url,
                label=label,
                result=result,
            )

            append_result(
                output_file,
                row,
            )

            print(
                f"Collector error: {exc}"
            )

        print(
            "-" * 60
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)

    print(
        "COLLECTION FINISHED"
    )

    print("=" * 60)

    print(
        f"Successful crawls: "
        f"{success_count}"
    )

    print(
        f"Other/failed crawls: "
        f"{failure_count}"
    )

    print(
        f"Output saved to: "
        f"{output_file}"
    )

    print("=" * 60)


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse dataset collector command-line parameters.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Collect dynamic website phishing "
            "research features."
        )
    )

    # --------------------------------------------------------
    # Input dataset
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Output dataset
    # --------------------------------------------------------

    parser.add_argument(
        "--output",
        default=(
            "data/processed/"
            "website_pilot_dataset.csv"
        ),
        help=(
            "Path for collected result CSV."
        ),
    )

    # --------------------------------------------------------
    # Collection mode
    # --------------------------------------------------------

    parser.add_argument(
        "--mode",
        choices=[
            "legitimate",
            "phishing",
            "balanced",
        ],
        default="legitimate",
        help=(
            "Select legitimate, phishing "
            "or balanced URLs."
        ),
    )

    # --------------------------------------------------------
    # Number of samples
    # --------------------------------------------------------

    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help=(
            "Number of samples. "
            "In balanced mode this is "
            "the number per class."
        ),
    )

    # --------------------------------------------------------
    # Random seed
    # --------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "Random sampling seed."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Main command-line execution.
    """

    args = parse_arguments()

    input_file = Path(
        args.input
    )

    output_file = Path(
        args.output
    )

    # --------------------------------------------------------
    # Validate sample count
    # --------------------------------------------------------

    if args.count <= 0:
        raise ValueError(
            "--count must be greater than zero."
        )

    # --------------------------------------------------------
    # Load PhiUSIIL dataset
    # --------------------------------------------------------

    dataframe = load_dataset(
        input_file
    )

    # --------------------------------------------------------
    # Select requested sample
    # --------------------------------------------------------

    selected_samples = select_samples(
        dataframe=dataframe,
        mode=args.mode,
        count=args.count,
        random_seed=args.seed,
    )

    # --------------------------------------------------------
    # Display selected classes
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
            output_file,
    )


if __name__ == "__main__":
    main()