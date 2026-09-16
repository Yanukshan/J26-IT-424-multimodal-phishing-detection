from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import pandas as pd


# ============================================================
# LABELS
# ============================================================

LABEL_PHISHING = 0
LABEL_LEGITIMATE = 1


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_RANDOM_SEED = 42

DEFAULT_LEGITIMATE_INPUT = (
    "data/processed/legitimate_500_full_collected.csv"
)

DEFAULT_PHISHING_INPUT = (
    "data/processed/phishtank_phishing_500_collected.csv"
)

DEFAULT_OUTPUT = (
    "data/processed/clean_balanced_full_dataset.csv"
)

# 0 means unlimited.
DEFAULT_MAX_PER_DOMAIN = 2


# ============================================================
# BOOLEAN MODEL FEATURES
# ============================================================

BOOLEAN_FEATURES = {
    "dom_login_form_present",
    "beh_late_navigation_detected",
}


# ============================================================
# DOM FEATURES
# ============================================================

DOM_FEATURES = [
    "dom_links",
    "dom_forms",
    "dom_scripts",
    "dom_images",
    "dom_iframes",
    "dom_inputs",
    "dom_password_fields",
    "dom_hidden_inputs",

    "dom_external_links",
    "dom_external_link_ratio",

    "dom_javascript_links",
    "dom_mailto_links",

    "dom_external_scripts",
    "dom_inline_scripts",

    "dom_external_images",
    "dom_external_iframes",

    "dom_forms_without_action",
    "dom_external_form_actions",
    "dom_insecure_form_actions",

    "dom_post_forms",
    "dom_login_form_present",
]


# ============================================================
# NETWORK FEATURES
# ============================================================

NETWORK_FEATURES = [
    "net_total_requests",

    "net_get_requests",
    "net_post_requests",
    "net_put_requests",
    "net_delete_requests",

    "net_document_requests",
    "net_script_requests",
    "net_stylesheet_requests",
    "net_image_requests",

    "net_xhr_requests",
    "net_fetch_requests",

    "net_font_requests",
    "net_media_requests",

    "net_external_request_count",
    "net_external_request_ratio",

    "net_external_domain_count",
    "net_unique_domain_count",

    "net_insecure_http_requests",
]


# ============================================================
# GRAPH FEATURES
# ============================================================

GRAPH_FEATURES = [
    "graph_node_count",
    "graph_edge_count",

    "graph_external_node_count",
    "graph_external_node_ratio",

    "graph_form_node_count",
    "graph_form_target_node_count",

    "graph_script_node_count",
    "graph_iframe_node_count",

    "graph_network_resource_nodes",
    "graph_runtime_requested_nodes",

    "graph_density",
    "graph_average_degree",
]


# ============================================================
# BEHAVIORAL FEATURES
# ============================================================

BEHAVIORAL_FEATURES = [
    "beh_response_count",

    "beh_main_document_response_count",

    "beh_response_3xx_count",
    "beh_response_4xx_count",
    "beh_response_5xx_count",

    "beh_failed_request_count",
    "beh_failed_document_request_count",

    "beh_failed_request_detail_count",
    "beh_failed_request_domain_count",

    "beh_server_redirect_hops",

    "beh_late_navigation_detected",
]


# ============================================================
# FEATURE PROFILES
# ============================================================

CORE_FEATURES = (
    DOM_FEATURES
    + NETWORK_FEATURES
    + GRAPH_FEATURES
)

FULL_FEATURES = (
    CORE_FEATURES
    + BEHAVIORAL_FEATURES
)


# ============================================================
# OUTPUT METADATA
# ============================================================

METADATA_COLUMNS = [
    "label",
    "class_name",

    "source_label",

    "source_url",
    "requested_url",
    "final_url",

    "effective_domain",

    "source_dataset",
    "source_id",

    "source_verified",
    "source_online",
    "source_verification_time",

    "collected_at_utc",
]


# ============================================================
# SAFE STRING
# ============================================================

def safe_string(value: Any) -> str:
    """
    Convert nullable values into clean strings.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


# ============================================================
# SAFE BOOLEAN
# ============================================================

def as_bool(value: Any) -> bool:
    """
    Convert common boolean representations to bool.
    """

    if isinstance(value, bool):
        return value

    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass

    if isinstance(value, (int, float)):
        return bool(value)

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
        }
    )


# ============================================================
# STRING COLUMN NORMALIZATION
# ============================================================

def ensure_string_column(
    dataframe: pd.DataFrame,
    column: str,
    default_value: str = "",
) -> None:
    """
    Ensure a metadata column uses pandas string dtype.

    This is important because completely empty CSV columns
    can be inferred by pandas as float64.

    Example:
        source_dataset = NaN

    If we later assign:
        "PhiUSIIL"

    pandas may raise:

        TypeError:
        Invalid value 'PhiUSIIL' for dtype 'float64'

    Converting metadata columns to string dtype prevents that.
    """

    if column not in dataframe.columns:

        dataframe[column] = pd.Series(
            default_value,
            index=dataframe.index,
            dtype="string",
        )

        return

    dataframe[column] = (
        dataframe[column]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    if default_value:

        empty_mask = (
            dataframe[column]
            == ""
        )

        dataframe.loc[
            empty_mask,
            column,
        ] = default_value


# ============================================================
# DOMAIN NORMALIZATION
# ============================================================

def normalize_domain(domain: Any) -> str:
    """
    Normalize hostname.

    Example:

        www.example.com
            ->
        example.com

    NOTE:
    This is currently hostname-level normalization.

    For the final ML train/validation/test split, a
    registrable-domain/eTLD+1 grouping step should later be
    used to prevent subdomain leakage.
    """

    domain_string = (
        safe_string(domain)
        .lower()
    )

    if domain_string.startswith("www."):
        domain_string = domain_string[4:]

    return domain_string


# ============================================================
# DOMAIN FROM URL
# ============================================================

def domain_from_url(url: Any) -> str:
    """
    Extract normalized hostname from a URL.
    """

    url_string = safe_string(url)

    if not url_string:
        return ""

    try:

        parsed = urlparse(
            url_string
        )

        hostname = (
            parsed.hostname
            or ""
        )

        return normalize_domain(
            hostname
        )

    except Exception:
        return ""


# ============================================================
# EFFECTIVE DOMAIN
# ============================================================

def derive_effective_domain(
    row: pd.Series,
) -> str:
    """
    Determine the hostname represented by the collected page.

    Priority:

        final_domain
        requested_domain
        source_domain
        source_url

    Training candidates with cross-domain drift should
    normally already have been excluded by the collector.
    """

    for column in (
        "final_domain",
        "requested_domain",
        "source_domain",
    ):

        if column not in row.index:
            continue

        domain = normalize_domain(
            row[column]
        )

        if domain:
            return domain

    if "source_url" in row.index:

        return domain_from_url(
            row["source_url"]
        )

    return ""


# ============================================================
# NORMALIZE URL FOR DUPLICATE CHECKING
# ============================================================

def normalize_url_key(url: Any) -> str:
    """
    Create a normalized URL key for duplicate detection.

    - hostname lowercased
    - leading www removed
    - fragment removed
    - root path normalized
    - query preserved

    Query parameters are intentionally preserved because
    different phishing URLs can represent different pages.
    """

    url_string = safe_string(url)

    if not url_string:
        return ""

    try:

        parsed = urlparse(
            url_string
        )

        scheme = (
            parsed.scheme
            or ""
        ).lower()

        hostname = (
            parsed.hostname
            or ""
        ).lower()

        if hostname.startswith("www."):
            hostname = hostname[4:]

        if not hostname:
            return ""

        try:
            port = parsed.port
        except ValueError:
            port = None

        if port is not None:
            netloc = f"{hostname}:{port}"
        else:
            netloc = hostname

        path = (
            parsed.path
            or "/"
        )

        if path != "/":
            path = path.rstrip("/")

        normalized = urlunparse(
            (
                scheme,
                netloc,
                path,
                "",
                parsed.query,
                "",
            )
        )

        return normalized

    except Exception:

        return url_string.lower()


# ============================================================
# CLASS NAME
# ============================================================

def get_class_name(label: int) -> str:
    """
    Human-readable class name.
    """

    if label == LABEL_LEGITIMATE:
        return "legitimate"

    if label == LABEL_PHISHING:
        return "phishing"

    return "unknown"


# ============================================================
# BASIC COLUMN VALIDATION
# ============================================================

def validate_basic_columns(
    dataframe: pd.DataFrame,
    input_file: Path,
) -> None:
    """
    Validate columns required before cleaning.
    """

    required = {
        "source_url",
        "source_label",
        "training_candidate",
    }

    missing = (
        required
        - set(dataframe.columns)
    )

    if missing:

        raise ValueError(
            f"{input_file} is missing required column(s): "
            + ", ".join(
                sorted(missing)
            )
        )


# ============================================================
# FEATURE COLUMN VALIDATION
# ============================================================

def validate_feature_columns(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    input_name: str,
) -> None:
    """
    Verify that the selected feature profile exists.
    """

    missing = [
        feature
        for feature in feature_columns
        if feature not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            f"{input_name} does not contain all required "
            "features for the selected profile.\n\n"
            "Missing feature columns:\n  "
            + "\n  ".join(missing)
        )


# ============================================================
# LOAD COLLECTED DATASET
# ============================================================

def load_collected_dataset(
    input_file: Path,
    expected_label: int,
    default_source_dataset: str,
) -> pd.DataFrame:
    """
    Load one collected dataset.

    Only rows already marked as:

        training_candidate = True

    are retained.
    """

    if not input_file.exists():

        raise FileNotFoundError(
            f"Collected dataset not found: {input_file}"
        )

    dataframe = pd.read_csv(
        input_file,
        low_memory=False,
    )

    print()
    print(
        f"Loading: {input_file}"
    )

    print(
        f"Original rows: {len(dataframe):,}"
    )

    validate_basic_columns(
        dataframe=dataframe,
        input_file=input_file,
    )

    # --------------------------------------------------------
    # Normalize source_url early.
    # --------------------------------------------------------

    ensure_string_column(
        dataframe=dataframe,
        column="source_url",
    )

    # --------------------------------------------------------
    # Normalize source label.
    # --------------------------------------------------------

    dataframe["source_label"] = pd.to_numeric(
        dataframe["source_label"],
        errors="coerce",
    )

    missing_label_count = (
        dataframe["source_label"]
        .isna()
        .sum()
    )

    if missing_label_count > 0:

        print(
            "Rows removed because source_label "
            f"was missing/invalid: {missing_label_count:,}"
        )

    dataframe = dataframe.dropna(
        subset=[
            "source_label"
        ]
    ).copy()

    dataframe["source_label"] = (
        dataframe["source_label"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Ensure this input belongs to the expected class.
    # --------------------------------------------------------

    wrong_class_mask = (
        dataframe["source_label"]
        != expected_label
    )

    wrong_class_rows = int(
        wrong_class_mask.sum()
    )

    if wrong_class_rows > 0:

        print(
            "Rows removed because they had the "
            f"wrong source label: {wrong_class_rows:,}"
        )

        dataframe = dataframe[
            ~wrong_class_mask
        ].copy()

    # --------------------------------------------------------
    # Training candidates only.
    # --------------------------------------------------------

    candidate_mask = (
        dataframe[
            "training_candidate"
        ]
        .apply(as_bool)
    )

    dataframe = dataframe[
        candidate_mask
    ].copy()

    print(
        f"Training candidates: {len(dataframe):,}"
    )

    # --------------------------------------------------------
    # FIX:
    # Normalize source_dataset as STRING before assigning
    # "PhiUSIIL" or "PhishTank".
    #
    # Empty CSV columns are often read by pandas as float64.
    # --------------------------------------------------------

    ensure_string_column(
        dataframe=dataframe,
        column="source_dataset",
        default_value=default_source_dataset,
    )

    # --------------------------------------------------------
    # Normalize all other metadata columns as STRING.
    #
    # This prevents the same pandas float64 assignment problem
    # from occurring later with provenance fields.
    # --------------------------------------------------------

    optional_string_columns = [
        "source_id",
        "source_domain",

        "source_verified",
        "source_online",
        "source_verification_time",

        "requested_url",
        "final_url",

        "requested_domain",
        "final_domain",

        "collected_at_utc",
    ]

    for column in optional_string_columns:

        ensure_string_column(
            dataframe=dataframe,
            column=column,
        )

    # --------------------------------------------------------
    # Effective domain.
    # --------------------------------------------------------

    dataframe[
        "effective_domain"
    ] = dataframe.apply(
        derive_effective_domain,
        axis=1,
    )

    # --------------------------------------------------------
    # Normalized URL key.
    # --------------------------------------------------------

    dataframe[
        "_url_key"
    ] = (
        dataframe[
            "source_url"
        ]
        .apply(
            normalize_url_key
        )
    )

    # --------------------------------------------------------
    # Remove rows without resolvable URL/domain.
    # --------------------------------------------------------

    before_invalid = len(
        dataframe
    )

    valid_mask = (
        dataframe["_url_key"].ne("")
        &
        dataframe["effective_domain"].ne("")
    )

    dataframe = dataframe[
        valid_mask
    ].copy()

    removed_invalid = (
        before_invalid
        - len(dataframe)
    )

    if removed_invalid > 0:

        print(
            "Rows removed because URL/domain "
            f"could not be resolved: {removed_invalid:,}"
        )

    # --------------------------------------------------------
    # Remove duplicate URLs.
    # --------------------------------------------------------

    before_duplicates = len(
        dataframe
    )

    dataframe = dataframe.drop_duplicates(
        subset=[
            "_url_key"
        ],
        keep="first",
    ).copy()

    duplicates_removed = (
        before_duplicates
        - len(dataframe)
    )

    print(
        f"Duplicate URLs removed: {duplicates_removed:,}"
    )

    print(
        "Unique candidate domains: "
        f"{dataframe['effective_domain'].nunique():,}"
    )

    return dataframe.reset_index(
        drop=True
    )


# ============================================================
# FEATURE VALUE COERCION
# ============================================================

def coerce_feature_values(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Convert all selected model features to numeric values.

    Boolean features become:

        False -> 0
        True  -> 1
    """

    dataframe = dataframe.copy()

    for feature in feature_columns:

        if feature in BOOLEAN_FEATURES:

            dataframe[feature] = (
                dataframe[feature]
                .apply(
                    lambda value:
                    1 if as_bool(value) else 0
                )
                .astype(int)
            )

        else:

            dataframe[feature] = pd.to_numeric(
                dataframe[feature],
                errors="coerce",
            )

    return dataframe


# ============================================================
# REMOVE MISSING MODEL FEATURES
# ============================================================

def remove_missing_feature_rows(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    class_label: str,
) -> pd.DataFrame:
    """
    Remove rows missing model feature values.

    Missing behavioral observations are NOT filled with zero.

    Zero itself can be a real behavioral value, so imputing
    missing values as zero could create false observations.
    """

    before = len(
        dataframe
    )

    dataframe = dataframe.dropna(
        subset=feature_columns
    ).copy()

    removed = (
        before
        - len(dataframe)
    )

    print(
        f"{class_label} rows removed for "
        f"missing features: {removed:,}"
    )

    return dataframe


# ============================================================
# REMOVE CROSS-CLASS DOMAIN OVERLAP
# ============================================================

def remove_cross_class_domain_overlap(
    legitimate: pd.DataFrame,
    phishing: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    set[str],
]:
    """
    Remove hostname values appearing in both classes.

    Example:

        example.com -> legitimate
        example.com -> phishing

    We do not guess which label is correct.

    Instead, the overlapping hostname is excluded from both
    classes.
    """

    legitimate_domains = set(
        legitimate[
            "effective_domain"
        ]
    )

    phishing_domains = set(
        phishing[
            "effective_domain"
        ]
    )

    overlap = (
        legitimate_domains
        &
        phishing_domains
    )

    print()

    print(
        "Cross-class domain overlap: "
        f"{len(overlap):,}"
    )

    if not overlap:

        return (
            legitimate.reset_index(
                drop=True
            ),
            phishing.reset_index(
                drop=True
            ),
            overlap,
        )

    print(
        "Removing overlapping domains "
        "from both classes."
    )

    legitimate = legitimate[
        ~legitimate[
            "effective_domain"
        ].isin(
            overlap
        )
    ].copy()

    phishing = phishing[
        ~phishing[
            "effective_domain"
        ].isin(
            overlap
        )
    ].copy()

    return (
        legitimate.reset_index(
            drop=True
        ),
        phishing.reset_index(
            drop=True
        ),
        overlap,
    )


# ============================================================
# DOMAIN LIMIT
# ============================================================

def apply_domain_limit(
    dataframe: pd.DataFrame,
    max_per_domain: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Limit repeated records from one hostname.

    Rows are shuffled before applying the cap so that the same
    original CSV ordering is not always favored.
    """

    if max_per_domain <= 0:

        return dataframe.copy()

    shuffled = dataframe.sample(
        frac=1,
        random_state=random_seed,
    ).copy()

    limited = (
        shuffled.groupby(
            "effective_domain",
            sort=False,
            group_keys=False,
        )
        .head(
            max_per_domain
        )
        .reset_index(
            drop=True
        )
    )

    return limited


# ============================================================
# BALANCE CLASSES
# ============================================================

def balance_classes(
    legitimate: pd.DataFrame,
    phishing: pd.DataFrame,
    per_class: int,
    random_seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Balance legitimate and phishing classes.

    per_class = 0
        automatically uses the smaller available class.
    """

    available = min(
        len(legitimate),
        len(phishing),
    )

    if available <= 0:

        raise ValueError(
            "Cannot balance dataset because one class "
            "contains no usable records."
        )

    if per_class > 0:

        sample_size = min(
            available,
            per_class,
        )

    else:

        sample_size = available

    legitimate = legitimate.sample(
        n=sample_size,
        random_state=random_seed,
    ).copy()

    phishing = phishing.sample(
        n=sample_size,
        random_state=random_seed + 1,
    ).copy()

    return (
        legitimate.reset_index(
            drop=True
        ),
        phishing.reset_index(
            drop=True
        ),
    )


# ============================================================
# PREPARE FINAL CLASS OUTPUT
# ============================================================

def prepare_output_class(
    dataframe: pd.DataFrame,
    label: int,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Prepare one class for the final balanced CSV.
    """

    dataframe = dataframe.copy()

    dataframe["label"] = label

    dataframe["class_name"] = (
        get_class_name(
            label
        )
    )

    # --------------------------------------------------------
    # Ensure metadata columns exist.
    # --------------------------------------------------------

    string_metadata_columns = [
        "class_name",
        "source_url",
        "requested_url",
        "final_url",
        "effective_domain",
        "source_dataset",
        "source_id",
        "source_verified",
        "source_online",
        "source_verification_time",
        "collected_at_utc",
    ]

    for column in string_metadata_columns:

        ensure_string_column(
            dataframe=dataframe,
            column=column,
        )

    if "source_label" not in dataframe.columns:

        dataframe[
            "source_label"
        ] = label

    output_columns = (
        METADATA_COLUMNS
        + feature_columns
    )

    return dataframe[
        output_columns
    ].copy()


# ============================================================
# FINAL QA
# ============================================================

def run_final_qa(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """
    Run final integrity checks before saving.
    """

    missing_feature_values = int(
        dataframe[
            feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    duplicate_urls = int(
        dataframe[
            "source_url"
        ]
        .duplicated()
        .sum()
    )

    legitimate_domains = set(
        dataframe.loc[
            dataframe["label"]
            == LABEL_LEGITIMATE,
            "effective_domain",
        ]
    )

    phishing_domains = set(
        dataframe.loc[
            dataframe["label"]
            == LABEL_PHISHING,
            "effective_domain",
        ]
    )

    cross_class_overlap = (
        legitimate_domains
        &
        phishing_domains
    )

    if missing_feature_values > 0:

        raise ValueError(
            "Final dataset contains missing model "
            f"feature values: {missing_feature_values}"
        )

    if duplicate_urls > 0:

        raise ValueError(
            "Final dataset contains duplicate "
            f"source URLs: {duplicate_urls}"
        )

    if cross_class_overlap:

        raise ValueError(
            "Final dataset still contains cross-class "
            "domain overlap."
        )


# ============================================================
# BUILD BALANCED DATASET
# ============================================================

def build_balanced_dataset(
    legitimate_file: Path,
    phishing_file: Path,
    output_file: Path,
    feature_profile: str,
    per_class: int,
    max_per_domain: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Main dataset-building pipeline.
    """

    # --------------------------------------------------------
    # Select feature profile.
    # --------------------------------------------------------

    if feature_profile == "core":

        feature_columns = list(
            CORE_FEATURES
        )

    elif feature_profile == "full":

        feature_columns = list(
            FULL_FEATURES
        )

    else:

        raise ValueError(
            "Unsupported feature profile: "
            f"{feature_profile}"
        )

    print()
    print(
        "=" * 70
    )

    print(
        "BALANCED WEBSITE DATASET BUILDER"
    )

    print(
        "=" * 70
    )

    print(
        f"Feature profile: "
        f"{feature_profile.upper()}"
    )

    print(
        f"Number of model features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Load legitimate data.
    # --------------------------------------------------------

    legitimate = load_collected_dataset(
        input_file=legitimate_file,
        expected_label=LABEL_LEGITIMATE,
        default_source_dataset="PhiUSIIL",
    )

    # --------------------------------------------------------
    # Load phishing data.
    # --------------------------------------------------------

    phishing = load_collected_dataset(
        input_file=phishing_file,
        expected_label=LABEL_PHISHING,
        default_source_dataset="PhishTank",
    )

    # --------------------------------------------------------
    # Validate selected feature schema.
    # --------------------------------------------------------

    validate_feature_columns(
        dataframe=legitimate,
        feature_columns=feature_columns,
        input_name="Legitimate dataset",
    )

    validate_feature_columns(
        dataframe=phishing,
        feature_columns=feature_columns,
        input_name="Phishing dataset",
    )

    # --------------------------------------------------------
    # Convert features to numeric.
    # --------------------------------------------------------

    legitimate = coerce_feature_values(
        dataframe=legitimate,
        feature_columns=feature_columns,
    )

    phishing = coerce_feature_values(
        dataframe=phishing,
        feature_columns=feature_columns,
    )

    # --------------------------------------------------------
    # Remove rows with missing required model features.
    # --------------------------------------------------------

    legitimate = remove_missing_feature_rows(
        dataframe=legitimate,
        feature_columns=feature_columns,
        class_label="Legitimate",
    )

    phishing = remove_missing_feature_rows(
        dataframe=phishing,
        feature_columns=feature_columns,
        class_label="Phishing",
    )

    # --------------------------------------------------------
    # Remove hostname overlap between classes.
    # --------------------------------------------------------

    (
        legitimate,
        phishing,
        overlap_domains,
    ) = remove_cross_class_domain_overlap(
        legitimate=legitimate,
        phishing=phishing,
    )

    # --------------------------------------------------------
    # Apply within-class hostname diversity.
    # --------------------------------------------------------

    legitimate = apply_domain_limit(
        dataframe=legitimate,
        max_per_domain=max_per_domain,
        random_seed=random_seed,
    )

    phishing = apply_domain_limit(
        dataframe=phishing,
        max_per_domain=max_per_domain,
        random_seed=random_seed + 10,
    )

    print()
    print(
        "After domain diversity limit:"
    )

    print(
        f"  Legitimate rows: "
        f"{len(legitimate):,}"
    )

    print(
        f"  Legitimate domains: "
        f"{legitimate['effective_domain'].nunique():,}"
    )

    print(
        f"  Phishing rows: "
        f"{len(phishing):,}"
    )

    print(
        f"  Phishing domains: "
        f"{phishing['effective_domain'].nunique():,}"
    )

    # --------------------------------------------------------
    # Balance classes.
    # --------------------------------------------------------

    (
        legitimate,
        phishing,
    ) = balance_classes(
        legitimate=legitimate,
        phishing=phishing,
        per_class=per_class,
        random_seed=random_seed,
    )

    # --------------------------------------------------------
    # Prepare class outputs.
    # --------------------------------------------------------

    legitimate_output = prepare_output_class(
        dataframe=legitimate,
        label=LABEL_LEGITIMATE,
        feature_columns=feature_columns,
    )

    phishing_output = prepare_output_class(
        dataframe=phishing,
        label=LABEL_PHISHING,
        feature_columns=feature_columns,
    )

    # --------------------------------------------------------
    # Combine both classes.
    # --------------------------------------------------------

    combined = pd.concat(
        [
            legitimate_output,
            phishing_output,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Shuffle combined dataset.
    # --------------------------------------------------------

    combined = (
        combined.sample(
            frac=1,
            random_state=random_seed,
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Final QA before saving.
    # --------------------------------------------------------

    run_final_qa(
        dataframe=combined,
        feature_columns=feature_columns,
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    legitimate_count = int(
        (
            combined["label"]
            == LABEL_LEGITIMATE
        ).sum()
    )

    phishing_count = int(
        (
            combined["label"]
            == LABEL_PHISHING
        ).sum()
    )

    print()
    print(
        "=" * 70
    )

    print(
        "CLEAN BALANCED DATASET CREATED"
    )

    print(
        "=" * 70
    )

    print(
        f"Total rows: "
        f"{len(combined):,}"
    )

    print(
        f"Legitimate rows: "
        f"{legitimate_count:,}"
    )

    print(
        f"Phishing rows: "
        f"{phishing_count:,}"
    )

    print(
        f"Unique domains: "
        f"{combined['effective_domain'].nunique():,}"
    )

    print(
        "Cross-class domains removed: "
        f"{len(overlap_domains):,}"
    )

    print(
        f"Model features: "
        f"{len(feature_columns)}"
    )

    print(
        "Missing feature values: "
        f"{combined[feature_columns].isna().sum().sum():,}"
    )

    print(
        "Duplicate source URLs: "
        f"{combined['source_url'].duplicated().sum():,}"
    )

    print(
        f"Output: "
        f"{output_file}"
    )

    print(
        "=" * 70
    )

    return combined


# ============================================================
# CLI ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Build a clean balanced website phishing "
            "research dataset from collected legitimate "
            "and phishing records."
        )
    )

    parser.add_argument(
        "--legitimate-input",
        default=DEFAULT_LEGITIMATE_INPUT,
        help=(
            "Collected legitimate CSV file."
        ),
    )

    parser.add_argument(
        "--phishing-input",
        default=DEFAULT_PHISHING_INPUT,
        help=(
            "Collected phishing CSV file."
        ),
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            "Output clean balanced CSV."
        ),
    )

    parser.add_argument(
        "--feature-profile",
        choices=[
            "core",
            "full",
        ],
        default="full",
        help=(
            "core = DOM + Network + Graph. "
            "full = core + Behavioral telemetry."
        ),
    )

    parser.add_argument(
        "--per-class",
        type=int,
        default=0,
        help=(
            "Rows to keep per class. "
            "0 automatically uses the smaller class."
        ),
    )

    parser.add_argument(
        "--max-per-domain",
        type=int,
        default=DEFAULT_MAX_PER_DOMAIN,
        help=(
            "Maximum rows allowed from one hostname. "
            "0 disables the domain limit."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "Random seed for deterministic sampling."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    CLI entry point.
    """

    args = parse_arguments()

    if args.per_class < 0:

        raise ValueError(
            "--per-class cannot be negative."
        )

    if args.max_per_domain < 0:

        raise ValueError(
            "--max-per-domain cannot be negative."
        )

    build_balanced_dataset(
        legitimate_file=Path(
            args.legitimate_input
        ),

        phishing_file=Path(
            args.phishing_input
        ),

        output_file=Path(
            args.output
        ),

        feature_profile=args.feature_profile,

        per_class=args.per_class,

        max_per_domain=args.max_per_domain,

        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()