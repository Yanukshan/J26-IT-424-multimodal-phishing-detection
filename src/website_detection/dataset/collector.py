import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.website_detection.behavioral_analysis.network_features import (
    extract_behavioral_features,
)

from src.website_detection.crawler.crawler import (
    crawl_website,
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

LABEL_PHISHING = 0
LABEL_LEGITIMATE = 1

DEFAULT_RANDOM_SEED = 42

MINIMAL_HTML_LENGTH = 1500


# ============================================================
# OPTIONAL SOURCE / PROVENANCE COLUMNS
# ============================================================

SOURCE_METADATA_COLUMNS = [
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
# URL / DOMAIN FEATURES
# ============================================================

URL_FEATURES = [
    "length",
    "hostname_length",
    "path_length",
    "query_length",
    "fragment_length",
    "https",
    "has_query",
    "has_fragment",
    "hostname_label_count",
    "subdomain_depth",
    "path_depth",
    "query_parameter_count",
    "digit_count",
    "digit_ratio",
    "letter_count",
    "letter_ratio",
    "dot_count",
    "hyphen_count",
    "underscore_count",
    "at_count",
    "percent_count",
    "ampersand_count",
    "equals_count",
    "special_character_count",
    "has_ip_hostname",
    "has_punycode",
    "has_explicit_port",
    "has_nonstandard_port",
    "has_username",
    "has_password_in_url",
    "double_slash_in_path",
    "entropy",
    "hostname_entropy",
    "path_entropy",
    "token_count",
    "suspicious_token_count",
    "has_login_token",
    "has_verify_token",
    "has_account_token",
    "has_security_token",
    "has_payment_token",
    "redirected",
    "hostname_changed",
    "scheme_changed",
    "final_length",
    "final_hostname_length",
    "final_https",
    "final_subdomain_depth",
    "final_entropy",
]


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
# CREDENTIAL INTENT FEATURES
# ============================================================

CREDENTIAL_FEATURES = [
    "form_count",
    "input_count",
    "email_fields",
    "password_fields",
    "otp_fields",
    "card_fields",
    "cvv_fields",
    "expiry_fields",
    "pin_fields",
    "bank_fields",
    "identity_fields",
    "sensitive_field_count",
    "financial_field_count",
    "credential_forms",
    "external_form_actions",
    "insecure_form_actions",
    "empty_form_actions",
    "external_credential_forms",
    "insecure_credential_forms",
    "login_intent_present",
    "payment_intent_present",
    "otp_intent_present",
    "identity_intent_present",
    "external_sensitive_submission",
    "insecure_sensitive_submission",
]


# ============================================================
# VISUAL SCREENSHOT METADATA
# ============================================================

VISUAL_FEATURES = [
    "screenshot_saved",
    "screenshot_path",
    "screenshot_width",
    "screenshot_height",
    "screenshot_file_size",
    "screenshot_reused",
    "screenshot_error_type",
    "screenshot_error",
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
# NEW BEHAVIORAL FEATURES
# ============================================================

BEHAVIORAL_FEATURES = [
    "response_count",
    "main_document_response_count",
    "response_3xx_count",
    "response_4xx_count",
    "response_5xx_count",
    "failed_request_count",
    "failed_document_request_count",
    "failed_request_detail_count",
    "failed_request_domain_count",
    "server_redirect_hops",
    "server_redirect_chain",
    "initial_response_url",
    "late_navigation_detected",
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
# FULL GRAPH ARTIFACT METADATA
# ============================================================

GRAPH_ARTIFACT_FEATURES = [
    "artifact_saved",
    "artifact_path",
    "artifact_file_size",
    "artifact_reused",
    "artifact_schema_version",
    "artifact_error_type",
    "artifact_error",
]


# ============================================================
# GRAPH COLUMN HELPER
# ============================================================

def get_graph_output_column(
    feature_name: str,
) -> str:

    if feature_name.startswith(
        "graph_"
    ):
        return feature_name

    return f"graph_{feature_name}"


# ============================================================
# OUTPUT COLUMNS
# ============================================================

BASE_COLUMNS = [
    "sample_id",

    "source_url",
    "source_label",
    "class_name",

    # Original dataset provenance
    *SOURCE_METADATA_COLUMNS,

    "collected_at_utc",

    "crawl_status",

    "requested_url",
    "requested_domain",

    "final_url",
    "final_domain",

    "redirected",
    "cross_domain_redirect",
    "label_review_required",

    "content_review_required",
    "review_reasons",
    "training_candidate",

    "status_code",

    "title",
    "html_length",

    "error_type",
    "error",
]


OUTPUT_COLUMNS = (
    BASE_COLUMNS

    + [
        f"url_{feature}"
        for feature
        in URL_FEATURES
    ]

    + [
        f"dom_{feature}"
        for feature
        in DOM_FEATURES
    ]

    + [
        f"cred_{feature}"
        for feature
        in CREDENTIAL_FEATURES
    ]

    + [
        f"visual_{feature}"
        for feature
        in VISUAL_FEATURES
    ]

    + [
        f"net_{feature}"
        for feature
        in NETWORK_FEATURES
    ]

    + [
        f"beh_{feature}"
        for feature
        in BEHAVIORAL_FEATURES
    ]

    + [
        get_graph_output_column(
            feature
        )
        for feature
        in GRAPH_FEATURES
    ]

    + [
        f"graph_{feature}"
        for feature
        in GRAPH_ARTIFACT_FEATURES
    ]
)


# ============================================================
# LABEL
# ============================================================

def get_class_name(
    label: int,
) -> str:

    if label == LABEL_LEGITIMATE:
        return "legitimate"

    if label == LABEL_PHISHING:
        return "phishing"

    return "unknown"


# ============================================================
# BOOLEAN
# ============================================================

def as_bool(
    value: Any,
) -> bool:

    if isinstance(
        value,
        bool,
    ):
        return value

    if value is None:
        return False

    if isinstance(
        value,
        (int, float),
    ):
        return bool(
            value
        )

    return str(
        value
    ).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


# ============================================================
# METADATA CLEANING
# ============================================================

def clean_metadata_value(
    value: Any,
) -> Any:

    if value is None:
        return ""

    try:

        if pd.isna(
            value
        ):
            return ""

    except Exception:
        pass

    return value


# ============================================================
# PARKING / LANDER DETECTION
# ============================================================

def detect_parking_page(
    title: str,
    final_url: str,
    external_domains: Any,
) -> bool:

    title_lower = str(
        title or ""
    ).strip().lower()

    url_lower = str(
        final_url or ""
    ).strip().lower()

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
        for indicator
        in parking_title_indicators
    ):
        return True

    if (
        url_lower.endswith(
            "/lander"
        )
        or "/parking-lander"
        in url_lower
    ):
        return True

    if isinstance(
        external_domains,
        list,
    ):

        domains = [
            str(domain).lower()
            for domain
            in external_domains
        ]

    elif external_domains:

        domains = [
            domain.strip().lower()
            for domain
            in str(
                external_domains
            ).split("|")
            if domain.strip()
        ]

    else:
        domains = []

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
            for indicator
            in parking_domain_indicators
        ):
            return True

    return False


# ============================================================
# CONTENT QUALITY
# ============================================================

def assess_content_quality(
    result: dict[str, Any],
) -> dict[str, Any]:

    crawl_status = str(
        result.get(
            "crawl_status",
            "",
        )
    ).upper()

    # --------------------------------------------------------
    # Non-success records never enter training.
    # --------------------------------------------------------

    if crawl_status != "SUCCESS":

        return {
            "content_review_required":
                False,

            "review_reasons":
                "",

            "training_candidate":
                False,
        }

    reasons: list[str] = []

    # --------------------------------------------------------
    # Cross-domain drift
    # --------------------------------------------------------

    if as_bool(
        result.get(
            "cross_domain_redirect",
            False,
        )
    ):

        reasons.append(
            "CROSS_DOMAIN_REDIRECT"
        )

    # --------------------------------------------------------
    # Page information
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

    try:

        html_length = int(
            result.get(
                "html_length",
                0,
            )
            or 0
        )

    except (
        TypeError,
        ValueError,
    ):

        html_length = 0

    minimal_page = (
        html_length
        < MINIMAL_HTML_LENGTH
    )

    empty_title = (
        title == ""
    )

    # --------------------------------------------------------
    # Conservative minimal-content review
    # --------------------------------------------------------

    if (
        minimal_page
        and empty_title
    ):

        reasons.append(
            "MINIMAL_PAGE"
        )

        reasons.append(
            "EMPTY_TITLE"
        )

    # --------------------------------------------------------
    # Parking / lander
    # --------------------------------------------------------

    network_features = result.get(
        "network_features",
        {},
    )

    external_domains = []

    if isinstance(
        network_features,
        dict,
    ):

        external_domains = (
            network_features.get(
                "external_domains",
                [],
            )
        )

    if detect_parking_page(
        title=
            title,

        final_url=
            final_url,

        external_domains=
            external_domains,
    ):

        reasons.append(
            "PARKING_PAGE"
        )

    # --------------------------------------------------------
    # De-duplicate reasons
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

    return {
        "content_review_required":
            content_review_required,

        "review_reasons":
            "|".join(
                unique_reasons
            ),

        "training_candidate":
            not content_review_required,
    }


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(
    input_file: Path,
) -> pd.DataFrame:

    if not input_file.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{input_file}"
        )

    print()
    print(
        "Loading dataset..."
    )

    print(
        f"File: {input_file}"
    )

    # --------------------------------------------------------
    # Read header first.
    #
    # PhiUSIIL contains many unused columns, so we still avoid
    # loading the entire original dataset into memory.
    # --------------------------------------------------------

    header = pd.read_csv(
        input_file,
        nrows=0,
    )

    available_columns = set(
        header.columns
    )

    required_columns = {
        "URL",
        "label",
    }

    missing = (
        required_columns
        - available_columns
    )

    if missing:

        raise ValueError(
            "Missing required dataset column(s): "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    available_metadata = [
        column
        for column
        in SOURCE_METADATA_COLUMNS
        if column
        in available_columns
    ]

    use_columns = [
        "URL",
        "label",
        *available_metadata,
    ]

    dtype_map = {
        "URL":
            str,
    }

    for column in available_metadata:
        dtype_map[
            column
        ] = str

    dataframe = pd.read_csv(
        input_file,
        usecols=
            use_columns,

        dtype=
            dtype_map,

        keep_default_na=
            False,
    )

    print(
        f"Original rows: "
        f"{len(dataframe):,}"
    )

    if available_metadata:

        print(
            "Source metadata loaded: "
            + ", ".join(
                available_metadata
            )
        )

    # --------------------------------------------------------
    # Normalize URL
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

    # --------------------------------------------------------
    # Normalize labels
    # --------------------------------------------------------

    dataframe[
        "label"
    ] = pd.to_numeric(
        dataframe[
            "label"
        ],
        errors=
            "coerce",
    )

    dataframe = dataframe.dropna(
        subset=[
            "URL",
            "label",
        ]
    ).copy()

    dataframe = dataframe[
        dataframe[
            "URL"
        ] != ""
    ].copy()

    dataframe[
        "label"
    ] = dataframe[
        "label"
    ].astype(
        int
    )

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
    # Deduplicate URLs
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
            keep=
                "first",
        )
        .reset_index(
            drop=
                True
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

    legitimate_count = int(
        (
            dataframe[
                "label"
            ]
            == LABEL_LEGITIMATE
        ).sum()
    )

    phishing_count = int(
        (
            dataframe[
                "label"
            ]
            == LABEL_PHISHING
        ).sum()
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

    if mode == "legitimate":

        sample_count = min(
            count,
            len(
                legitimate
            ),
        )

        selected = legitimate.sample(
            n=
                sample_count,

            random_state=
                random_seed,
        )

    elif mode == "phishing":

        sample_count = min(
            count,
            len(
                phishing
            ),
        )

        selected = phishing.sample(
            n=
                sample_count,

            random_state=
                random_seed,
        )

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

        legitimate_sample = legitimate.sample(
            n=
                legitimate_count,

            random_state=
                random_seed,
        )

        phishing_sample = phishing.sample(
            n=
                phishing_count,

            random_state=
                random_seed,
        )

        selected = pd.concat(
            [
                legitimate_sample,
                phishing_sample,
            ],
            ignore_index=
                True,
        )

    else:

        raise ValueError(
            f"Unsupported mode: "
            f"{mode}"
        )

    return (
        selected.sample(
            frac=
                1,

            random_state=
                random_seed,
        )
        .reset_index(
            drop=
                True
        )
    )


# ============================================================
# SOURCE METADATA
# ============================================================

def extract_source_metadata(
    sample: pd.Series,
) -> dict[str, Any]:

    metadata = {}

    for column in SOURCE_METADATA_COLUMNS:

        value = ""

        if column in sample.index:
            value = sample[
                column
            ]

        metadata[
            column
        ] = clean_metadata_value(
            value
        )

    return metadata


# ============================================================
# FLATTEN RESULT
# ============================================================

def flatten_result(
    sample_id: int,
    source_url: str,
    label: int,
    result: dict[str, Any],
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:

    row: dict[str, Any] = {
        column:
            ""
        for column
        in OUTPUT_COLUMNS
    }

    # --------------------------------------------------------
    # Source information
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

    if source_metadata:

        for column in SOURCE_METADATA_COLUMNS:

            row[
                column
            ] = clean_metadata_value(
                source_metadata.get(
                    column,
                    "",
                )
            )

    row[
        "collected_at_utc"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    # --------------------------------------------------------
    # Crawl metadata
    # --------------------------------------------------------

    for column in [
        "crawl_status",
        "requested_url",
        "requested_domain",
        "final_url",
        "final_domain",
        "redirected",
        "cross_domain_redirect",
        "label_review_required",
        "status_code",
        "title",
        "html_length",
        "error_type",
        "error",
    ]:

        row[
            column
        ] = result.get(
            column,
            "",
        )

    # --------------------------------------------------------
    # Content quality
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

    # ========================================================
    # URL / DOMAIN
    # ========================================================

    url_features = result.get(
        "url_features",
        {},
    )

    if not isinstance(
        url_features,
        dict,
    ):
        url_features = {}

    for feature in URL_FEATURES:

        row[
            f"url_{feature}"
        ] = url_features.get(
            feature,
            "",
        )

    # ========================================================
    # DOM
    # ========================================================

    dom_features = result.get(
        "dom_features",
        {},
    )

    if not isinstance(
        dom_features,
        dict,
    ):
        dom_features = {}

    for feature in DOM_FEATURES:

        row[
            f"dom_{feature}"
        ] = dom_features.get(
            feature,
            "",
        )

    # ========================================================
    # CREDENTIAL INTENT
    # ========================================================

    credential_features = result.get(
        "credential_features",
        {},
    )

    if not isinstance(
        credential_features,
        dict,
    ):
        credential_features = {}

    for feature in CREDENTIAL_FEATURES:

        row[
            f"cred_{feature}"
        ] = credential_features.get(
            feature,
            "",
        )

    # ========================================================
    # VISUAL SCREENSHOT METADATA
    # ========================================================

    visual_features = result.get(
        "visual_features",
        {},
    )

    if not isinstance(
        visual_features,
        dict,
    ):
        visual_features = {}

    for feature in VISUAL_FEATURES:

        row[
            f"visual_{feature}"
        ] = visual_features.get(
            feature,
            "",
        )

    # ========================================================
    # NETWORK
    # ========================================================

    network_features = result.get(
        "network_features",
        {},
    )

    if not isinstance(
        network_features,
        dict,
    ):
        network_features = {}

    for feature in NETWORK_FEATURES:

        value = network_features.get(
            feature,
            "",
        )

        if isinstance(
            value,
            list,
        ):

            value = "|".join(
                str(item)
                for item
                in value
            )

        row[
            f"net_{feature}"
        ] = value

    # ========================================================
    # NEW BEHAVIORAL TELEMETRY
    # ========================================================

    behavioral_features = (
        extract_behavioral_features(
            behavioral_telemetry=
                result.get(
                    "behavioral_telemetry",
                    {},
                ),

            failed_requests=
                result.get(
                    "failed_requests",
                    [],
                ),
        )
    )

    for feature in BEHAVIORAL_FEATURES:

        value = behavioral_features.get(
            feature,
            "",
        )

        if isinstance(
            value,
            list,
        ):

            value = "|".join(
                str(item)
                for item
                in value
            )

        row[
            f"beh_{feature}"
        ] = value

    # ========================================================
    # GRAPH
    # ========================================================

    graph_features = result.get(
        "graph_features",
        {},
    )

    if not isinstance(
        graph_features,
        dict,
    ):
        graph_features = {}

    for feature in GRAPH_FEATURES:

        row[
            get_graph_output_column(
                feature
            )
        ] = graph_features.get(
            feature,
            "",
        )

    # ========================================================
    # FULL GRAPH ARTIFACT
    # ========================================================

    graph_artifact = result.get(
        "graph_artifact",
        {},
    )

    if not isinstance(
        graph_artifact,
        dict,
    ):
        graph_artifact = {}

    for feature in GRAPH_ARTIFACT_FEATURES:

        row[
            f"graph_{feature}"
        ] = graph_artifact.get(
            feature,
            "",
        )

    return row


# ============================================================
# OUTPUT INITIALIZATION
# ============================================================

def validate_existing_output_schema(
    output_file: Path,
) -> None:

    with output_file.open(
        "r",
        encoding=
            "utf-8",
        newline="",
    ) as file:

        reader = csv.reader(
            file
        )

        try:
            header = next(
                reader
            )

        except StopIteration:

            raise ValueError(
                "Resume file exists but has no CSV header."
            )

    if header != OUTPUT_COLUMNS:

        raise ValueError(
            "Cannot resume because the existing output CSV "
            "uses a different schema. Use a new output filename."
        )


def initialize_output_file(
    output_file: Path,
    resume: bool,
) -> None:

    output_file.parent.mkdir(
        parents=
            True,

        exist_ok=
            True,
    )

    if (
        resume
        and output_file.exists()
    ):

        validate_existing_output_schema(
            output_file
        )

        return

    with output_file.open(
        "w",
        newline=
            "",

        encoding=
            "utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                OUTPUT_COLUMNS,
        )

        writer.writeheader()


# ============================================================
# RESUME STATE
# ============================================================

def load_resume_state(
    output_file: Path,
) -> tuple[
    set[str],
    int,
    int,
]:

    if not output_file.exists():

        return (
            set(),
            0,
            0,
        )

    dataframe = pd.read_csv(
        output_file,
        usecols=[
            "sample_id",
            "source_url",
        ],
    )

    if dataframe.empty:

        return (
            set(),
            0,
            0,
        )

    existing_urls = set(
        dataframe[
            "source_url"
        ]
        .dropna()
        .astype(str)
    )

    sample_ids = pd.to_numeric(
        dataframe[
            "sample_id"
        ],
        errors=
            "coerce",
    )

    max_sample_id = (
        int(
            sample_ids.max()
        )
        if sample_ids.notna().any()
        else 0
    )

    return (
        existing_urls,
        max_sample_id,
        len(
            dataframe
        ),
    )


# ============================================================
# APPEND RESULT
# ============================================================

def append_result(
    output_file: Path,
    row: dict[str, Any],
) -> None:

    with output_file.open(
        "a",
        newline=
            "",

        encoding=
            "utf-8",
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
# COLLECTION
# ============================================================

def collect_dataset(
    selected_samples: pd.DataFrame,
    output_file: Path,
    resume: bool = False,
) -> None:

    initialize_output_file(
        output_file=
            output_file,

        resume=
            resume,
    )

    existing_urls: set[str] = set()
    last_sample_id = 0
    existing_row_count = 0

    if resume:

        (
            existing_urls,
            last_sample_id,
            existing_row_count,
        ) = load_resume_state(
            output_file
        )

    pending_samples = (
        selected_samples[
            ~selected_samples[
                "URL"
            ]
            .astype(str)
            .isin(
                existing_urls
            )
        ]
        .reset_index(
            drop=
                True
        )
    )

    total_selected = len(
        selected_samples
    )

    total_pending = len(
        pending_samples
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
        f"{total_selected}"
    )

    if resume:

        print(
            f"Already collected: "
            f"{existing_row_count}"
        )

        print(
            f"Remaining this run: "
            f"{total_pending}"
        )

    print(
        f"Output: "
        f"{output_file}"
    )

    print(
        "=" * 60
    )

    print()

    if total_pending == 0:

        print(
            "No remaining URLs to collect."
        )

        return

    # ========================================================
    # PROCESS
    # ========================================================

    for index, sample in (
        pending_samples.iterrows()
    ):

        sample_id = (
            last_sample_id
            + index
            + 1
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

        class_name = get_class_name(
            label
        )

        source_metadata = (
            extract_source_metadata(
                sample
            )
        )

        print(
            f"[{index + 1}/{total_pending}] "
            f"{class_name.upper()}"
        )

        print(
            f"URL: {url}"
        )

        try:

            result = crawl_website(
                url=
                    url,

                label=
                    label,
            )

            status = str(
                result.get(
                    "crawl_status",
                    "UNKNOWN",
                )
            )

            row = flatten_result(
                sample_id=
                    sample_id,

                source_url=
                    url,

                label=
                    label,

                result=
                    result,

                source_metadata=
                    source_metadata,
            )

            append_result(
                output_file=
                    output_file,

                row=
                    row,
            )

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

            print(
                f"Status: "
                f"{status}"
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

            print(
                "Training candidate: "
                + (
                    "YES"
                    if as_bool(
                        row.get(
                            "training_candidate",
                            False,
                        )
                    )
                    else "NO"
                )
            )

        except KeyboardInterrupt:

            print()
            print(
                "Collection stopped by user."
            )

            print(
                "Already collected results remain saved."
            )

            break

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

                source_metadata=
                    source_metadata,
            )

            append_result(
                output_file=
                    output_file,

                row=
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
    # SUMMARY
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
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:

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
    )

    parser.add_argument(
        "--output",
        default=(
            "data/processed/"
            "website_pilot_dataset.csv"
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "legitimate",
            "phishing",
            "balanced",
        ],
        default=
            "legitimate",
    )

    parser.add_argument(
        "--count",
        type=
            int,

        default=
            10,
    )

    parser.add_argument(
        "--seed",
        type=
            int,

        default=
            DEFAULT_RANDOM_SEED,
    )

    parser.add_argument(
        "--resume",
        action=
            "store_true",

        help=(
            "Resume an interrupted collection using the "
            "same output CSV and sampling configuration."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_arguments()

    if args.count <= 0:

        raise ValueError(
            "--count must be greater than zero."
        )

    dataframe = load_dataset(
        Path(
            args.input
        )
    )

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

    collect_dataset(
        selected_samples=
            selected_samples,

        output_file=
            Path(
                args.output
            ),

        resume=
            args.resume,
    )


if __name__ == "__main__":
    main()