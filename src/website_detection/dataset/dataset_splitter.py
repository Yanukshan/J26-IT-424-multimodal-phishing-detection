from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

try:
    import tldextract
except ImportError as exc:
    raise RuntimeError(
        "tldextract is required for registrable-domain splitting. "
        "Install it with: pip install tldextract"
    ) from exc


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SEED = 42

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VALIDATION_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15

LABEL_PHISHING = 0
LABEL_LEGITIMATE = 1

# Use tldextract's bundled Public Suffix List snapshot.
# This avoids downloading suffix data during the experiment.
DOMAIN_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=()
)


# ============================================================
# BASIC HELPERS
# ============================================================

def as_bool(
    value: Any,
) -> bool:
    """
    Normalize common CSV boolean representations.
    """

    if isinstance(
        value,
        bool,
    ):
        return value

    if value is None:
        return False

    try:
        if pd.isna(
            value
        ):
            return False
    except Exception:
        pass

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


def normalize_hostname(
    hostname: str,
) -> str:
    """
    Normalize a hostname for grouping.
    """

    hostname = str(
        hostname or ""
    ).strip().lower().rstrip(".")

    if hostname.startswith(
        "www."
    ):
        hostname = hostname[
            4:
        ]

    return hostname


def hostname_from_url(
    url: str,
) -> str:
    """
    Extract a hostname from a URL.
    """

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

        return normalize_hostname(
            urlparse(
                value
            ).hostname
            or ""
        )

    except Exception:
        return ""


def registrable_domain(
    hostname: str,
) -> str:
    """
    Convert a hostname to its registrable domain / eTLD+1.

    Examples:
        login.example.com -> example.com
        portal.example.co.uk -> example.co.uk

    This is used to prevent related URLs/subdomains from being
    split across train, validation and test sets.
    """

    hostname = normalize_hostname(
        hostname
    )

    if not hostname:
        return ""

    extracted = DOMAIN_EXTRACTOR(
        hostname
    )

    domain = str(
        extracted.domain
        or ""
    ).strip().lower()

    suffix = str(
        extracted.suffix
        or ""
    ).strip().lower()

    if domain and suffix:

        return (
            f"{domain}.{suffix}"
        )

    if domain:

        return domain

    # IP addresses / unusual local hosts can fall back to the
    # normalized hostname itself.
    return hostname


# ============================================================
# EFFECTIVE DOMAIN
# ============================================================

def determine_effective_hostname(
    row: pd.Series,
) -> str:
    """
    Determine the best hostname for leakage-control grouping.

    Priority:
        1. final_domain
        2. requested_domain
        3. source_domain
        4. final_url
        5. requested_url
        6. source_url
    """

    for column in [
        "final_domain",
        "requested_domain",
        "source_domain",
    ]:

        if column not in row.index:
            continue

        value = normalize_hostname(
            row.get(
                column,
                "",
            )
        )

        if value:
            return value

    for column in [
        "final_url",
        "requested_url",
        "source_url",
    ]:

        if column not in row.index:
            continue

        value = hostname_from_url(
            row.get(
                column,
                "",
            )
        )

        if value:
            return value

    return ""


# ============================================================
# DATASET LOADING / FILTERING
# ============================================================

def load_and_filter_dataset(
    input_file: Path,
    require_artifacts: bool,
    check_artifact_files: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Load the finalized collection CSV and keep rows eligible for
    experiments.

    Failed crawls are not converted into negative evidence.
    Missing screenshots/graphs remain missing rather than zero.
    """

    if not input_file.exists():

        raise FileNotFoundError(
            f"Dataset not found: {input_file}"
        )

    dataframe = pd.read_csv(
        input_file,
        keep_default_na=True,
    )

    original_rows = len(
        dataframe
    )

    required_columns = {
        "source_label",
        "source_url",
        "crawl_status",
        "training_candidate",
    }

    missing = (
        required_columns
        - set(
            dataframe.columns
        )
    )

    if missing:

        raise ValueError(
            "Missing required column(s): "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    # --------------------------------------------------------
    # Valid binary labels only
    # --------------------------------------------------------

    dataframe[
        "source_label"
    ] = pd.to_numeric(
        dataframe[
            "source_label"
        ],
        errors="coerce",
    )

    dataframe = dataframe[
        dataframe[
            "source_label"
        ].isin(
            [
                LABEL_PHISHING,
                LABEL_LEGITIMATE,
            ]
        )
    ].copy()

    rows_after_label_filter = len(
        dataframe
    )

    dataframe[
        "source_label"
    ] = dataframe[
        "source_label"
    ].astype(
        int
    )

    # --------------------------------------------------------
    # Successful rendered pages only
    # --------------------------------------------------------

    dataframe = dataframe[
        dataframe[
            "crawl_status"
        ]
        .astype(str)
        .str.upper()
        .eq(
            "SUCCESS"
        )
    ].copy()

    rows_after_success_filter = len(
        dataframe
    )

    # --------------------------------------------------------
    # Existing collector quality decision
    # --------------------------------------------------------

    training_mask = dataframe[
        "training_candidate"
    ].map(
        as_bool
    )

    dataframe = dataframe[
        training_mask
    ].copy()

    rows_after_training_filter = len(
        dataframe
    )

    # --------------------------------------------------------
    # Multimodal artifact requirements
    # --------------------------------------------------------

    if require_artifacts:

        artifact_columns = {
            "visual_screenshot_saved",
            "visual_screenshot_path",
            "graph_artifact_saved",
            "graph_artifact_path",
        }

        missing_artifact_columns = (
            artifact_columns
            - set(
                dataframe.columns
            )
        )

        if missing_artifact_columns:

            raise ValueError(
                "Artifact filtering requested, but these "
                "columns are missing: "
                + ", ".join(
                    sorted(
                        missing_artifact_columns
                    )
                )
            )

        visual_mask = dataframe[
            "visual_screenshot_saved"
        ].map(
            as_bool
        )

        graph_mask = dataframe[
            "graph_artifact_saved"
        ].map(
            as_bool
        )

        dataframe = dataframe[
            visual_mask
            & graph_mask
        ].copy()

    rows_after_artifact_flags = len(
        dataframe
    )

    # --------------------------------------------------------
    # Optional physical-file verification
    # --------------------------------------------------------

    if (
        require_artifacts
        and check_artifact_files
    ):

        def artifact_files_exist(
            row: pd.Series,
        ) -> bool:

            screenshot_path = Path(
                str(
                    row.get(
                        "visual_screenshot_path",
                        "",
                    )
                    or ""
                )
            )

            graph_path = Path(
                str(
                    row.get(
                        "graph_artifact_path",
                        "",
                    )
                    or ""
                )
            )

            return (
                screenshot_path.is_file()
                and graph_path.is_file()
            )

        file_mask = dataframe.apply(
            artifact_files_exist,
            axis=1,
        )

        dataframe = dataframe[
            file_mask
        ].copy()

    rows_after_file_check = len(
        dataframe
    )

    # --------------------------------------------------------
    # Build leakage-control domains
    # --------------------------------------------------------

    dataframe[
        "effective_hostname"
    ] = dataframe.apply(
        determine_effective_hostname,
        axis=1,
    )

    dataframe[
        "registrable_domain"
    ] = dataframe[
        "effective_hostname"
    ].map(
        registrable_domain
    )

    dataframe = dataframe[
        dataframe[
            "registrable_domain"
        ].astype(str)
        .str.strip()
        .ne(
            ""
        )
    ].copy()

    rows_after_domain_filter = len(
        dataframe
    )

    # --------------------------------------------------------
    # De-duplicate exact source URLs
    # --------------------------------------------------------

    dataframe[
        "source_url"
    ] = dataframe[
        "source_url"
    ].astype(str).str.strip()

    before_duplicates = len(
        dataframe
    )

    dataframe = (
        dataframe
        .drop_duplicates(
            subset=[
                "source_url",
            ],
            keep="first",
        )
        .reset_index(
            drop=True,
        )
    )

    duplicates_removed = (
        before_duplicates
        - len(
            dataframe
        )
    )

    stats = {
        "original_rows":
            original_rows,

        "rows_after_label_filter":
            rows_after_label_filter,

        "rows_after_success_filter":
            rows_after_success_filter,

        "rows_after_training_filter":
            rows_after_training_filter,

        "rows_after_artifact_flags":
            rows_after_artifact_flags,

        "rows_after_file_check":
            rows_after_file_check,

        "rows_after_domain_filter":
            rows_after_domain_filter,

        "duplicate_source_urls_removed":
            duplicates_removed,

        "usable_rows_before_conflict_removal":
            len(
                dataframe
            ),
    }

    return (
        dataframe,
        stats,
    )


# ============================================================
# CROSS-CLASS DOMAIN CONFLICTS
# ============================================================

def remove_cross_class_domains(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Remove domains that occur with both phishing and legitimate
    labels.

    Keeping the same registrable domain under different class
    labels can create ambiguous training evidence and leakage.
    Conflicts are exported for manual review instead of being
    silently relabeled.
    """

    label_counts = (
        dataframe.groupby(
            "registrable_domain"
        )[
            "source_label"
        ]
        .nunique()
    )

    conflict_domains = set(
        label_counts[
            label_counts > 1
        ].index
    )

    if not conflict_domains:

        return (
            dataframe.copy(),
            dataframe.iloc[
                0:0
            ].copy(),
        )

    conflict_rows = dataframe[
        dataframe[
            "registrable_domain"
        ].isin(
            conflict_domains
        )
    ].copy()

    clean_rows = dataframe[
        ~dataframe[
            "registrable_domain"
        ].isin(
            conflict_domains
        )
    ].copy()

    return (
        clean_rows.reset_index(
            drop=True
        ),
        conflict_rows.reset_index(
            drop=True
        ),
    )


# ============================================================
# RATIO VALIDATION
# ============================================================

def validate_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:

    ratios = [
        train_ratio,
        validation_ratio,
        test_ratio,
    ]

    if any(
        ratio <= 0
        for ratio
        in ratios
    ):

        raise ValueError(
            "All split ratios must be greater than zero."
        )

    total = sum(
        ratios
    )

    if abs(
        total - 1.0
    ) > 1e-9:

        raise ValueError(
            "Split ratios must sum to 1.0. "
            f"Current total: {total}"
        )


# ============================================================
# DOMAIN-GROUP ASSIGNMENT
# ============================================================

def assign_domains_for_class(
    class_dataframe: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> dict[str, str]:
    """
    Assign whole registrable-domain groups to train/validation/
    test while approximately preserving row-level ratios.

    No domain is split between partitions.
    """

    group_sizes = (
        class_dataframe.groupby(
            "registrable_domain"
        )
        .size()
        .reset_index(
            name="row_count"
        )
    )

    if group_sizes.empty:

        return {}

    rng = random.Random(
        random_seed
    )

    groups = [
        (
            str(
                row[
                    "registrable_domain"
                ]
            ),
            int(
                row[
                    "row_count"
                ]
            ),
            rng.random(),
        )
        for _, row
        in group_sizes.iterrows()
    ]

    # Large domains first reduces ratio distortion.
    # Random tie-break makes the assignment deterministic with
    # respect to the supplied seed.
    groups.sort(
        key=lambda item: (
            -item[1],
            item[2],
        )
    )

    total_rows = sum(
        item[
            1
        ]
        for item
        in groups
    )

    ratios = {
        "train":
            train_ratio,

        "validation":
            validation_ratio,

        "test":
            test_ratio,
    }

    targets = {
        split:
            total_rows
            * ratio
        for split, ratio
        in ratios.items()
    }

    current = {
        "train":
            0,

        "validation":
            0,

        "test":
            0,
    }

    mapping: dict[
        str,
        str,
    ] = {}

    # --------------------------------------------------------
    # If enough domain groups exist, seed all three partitions.
    # This prevents tiny classes from accidentally producing an
    # empty validation/test set.
    # --------------------------------------------------------

    split_order = [
        "train",
        "validation",
        "test",
    ]

    seeded_groups = 0

    if len(
        groups
    ) >= 3:

        for split, group in zip(
            split_order,
            groups[
                :3
            ],
        ):

            domain, row_count, _ = group

            mapping[
                domain
            ] = split

            current[
                split
            ] += row_count

            seeded_groups += 1

    # --------------------------------------------------------
    # Greedy ratio-aware assignment
    # --------------------------------------------------------

    for domain, row_count, _ in groups[
        seeded_groups:
    ]:

        def deficit_score(
            split: str,
        ) -> float:

            target = targets[
                split
            ]

            if target <= 0:
                return float(
                    "-inf"
                )

            return (
                target
                - current[
                    split
                ]
            ) / target

        selected_split = max(
            split_order,
            key=deficit_score,
        )

        mapping[
            domain
        ] = selected_split

        current[
            selected_split
        ] += row_count

    return mapping


def build_domain_aware_split(
    dataframe: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> pd.DataFrame:
    """
    Build a class-stratified, domain-aware split.

    Phishing and legitimate domains are assigned independently
    so both classes remain represented in all partitions where
    enough domains are available.
    """

    mappings: dict[
        str,
        str,
    ] = {}

    for class_label in [
        LABEL_PHISHING,
        LABEL_LEGITIMATE,
    ]:

        class_dataframe = dataframe[
            dataframe[
                "source_label"
            ]
            == class_label
        ].copy()

        class_mapping = (
            assign_domains_for_class(
                class_dataframe=
                    class_dataframe,

                train_ratio=
                    train_ratio,

                validation_ratio=
                    validation_ratio,

                test_ratio=
                    test_ratio,

                random_seed=
                    (
                        random_seed
                        + class_label
                    ),
            )
        )

        for domain, split in class_mapping.items():

            if domain in mappings:

                raise RuntimeError(
                    "A registrable domain unexpectedly appeared "
                    "in multiple label groups after conflict "
                    f"removal: {domain}"
                )

            mappings[
                domain
            ] = split

    output = dataframe.copy()

    output[
        "dataset_split"
    ] = output[
        "registrable_domain"
    ].map(
        mappings
    )

    if output[
        "dataset_split"
    ].isna().any():

        missing_domains = sorted(
            output.loc[
                output[
                    "dataset_split"
                ].isna(),
                "registrable_domain",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "Split assignment missing for domain(s): "
            + ", ".join(
                missing_domains[
                    :10
                ]
            )
        )

    return output


# ============================================================
# LEAKAGE VALIDATION
# ============================================================

def validate_zero_domain_overlap(
    dataframe: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Verify that no registrable domain occurs in more than one
    partition.
    """

    domains = {
        split:
            set(
                dataframe.loc[
                    dataframe[
                        "dataset_split"
                    ]
                    == split,
                    "registrable_domain",
                ]
                .astype(str)
                .tolist()
            )
        for split
        in [
            "train",
            "validation",
            "test",
        ]
    }

    overlaps = {
        "train_validation":
            sorted(
                domains[
                    "train"
                ]
                & domains[
                    "validation"
                ]
            ),

        "train_test":
            sorted(
                domains[
                    "train"
                ]
                & domains[
                    "test"
                ]
            ),

        "validation_test":
            sorted(
                domains[
                    "validation"
                ]
                & domains[
                    "test"
                ]
            ),
    }

    if any(
        overlaps.values()
    ):

        raise RuntimeError(
            "Domain leakage detected between dataset splits."
        )

    return overlaps


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    dataframe: pd.DataFrame,
    filter_stats: dict[str, int],
    conflict_rows: pd.DataFrame,
    ratios: dict[str, float],
    random_seed: int,
) -> dict[str, Any]:

    split_summary = {}

    for split in [
        "train",
        "validation",
        "test",
    ]:

        subset = dataframe[
            dataframe[
                "dataset_split"
            ]
            == split
        ]

        split_summary[
            split
        ] = {
            "rows":
                len(
                    subset
                ),

            "domains":
                int(
                    subset[
                        "registrable_domain"
                    ].nunique()
                ),

            "phishing_rows":
                int(
                    (
                        subset[
                            "source_label"
                        ]
                        == LABEL_PHISHING
                    ).sum()
                ),

            "legitimate_rows":
                int(
                    (
                        subset[
                            "source_label"
                        ]
                        == LABEL_LEGITIMATE
                    ).sum()
                ),
        }

    return {
        "seed":
            random_seed,

        "requested_ratios":
            ratios,

        "filtering":
            filter_stats,

        "cross_class_conflict_rows_removed":
            len(
                conflict_rows
            ),

        "cross_class_conflict_domains_removed":
            int(
                conflict_rows[
                    "registrable_domain"
                ].nunique()
                if not conflict_rows.empty
                else 0
            ),

        "final_rows":
            len(
                dataframe
            ),

        "final_domains":
            int(
                dataframe[
                    "registrable_domain"
                ].nunique()
            ),

        "split_summary":
            split_summary,
    }


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    dataframe: pd.DataFrame,
    conflict_rows: pd.DataFrame,
    output_directory: Path,
    summary: dict[str, Any],
) -> None:

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_directory
        / "dataset_with_split.csv",
        index=False,
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        subset = dataframe[
            dataframe[
                "dataset_split"
            ]
            == split
        ].copy()

        subset.to_csv(
            output_directory
            / f"{split}.csv",
            index=False,
        )

    conflict_rows.to_csv(
        output_directory
        / "cross_class_domain_conflicts.csv",
        index=False,
    )

    with (
        output_directory
        / "split_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Create domain-aware train/validation/test splits "
            "for the finalized website phishing dataset."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Final collected/cleaned CSV."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/processed/"
            "final_splits"
        ),
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=
            DEFAULT_TRAIN_RATIO,
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=
            DEFAULT_VALIDATION_RATIO,
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=
            DEFAULT_TEST_RATIO,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=
            DEFAULT_SEED,
    )

    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help=(
            "Require both screenshot and full graph artifacts."
        ),
    )

    parser.add_argument(
        "--check-artifact-files",
        action="store_true",
        help=(
            "Also verify screenshot/graph paths physically "
            "exist. Use with --require-artifacts."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_arguments()

    validate_ratios(
        train_ratio=
            args.train_ratio,

        validation_ratio=
            args.validation_ratio,

        test_ratio=
            args.test_ratio,
    )

    if (
        args.check_artifact_files
        and not args.require_artifacts
    ):

        raise ValueError(
            "--check-artifact-files requires "
            "--require-artifacts."
        )

    input_file = Path(
        args.input
    )

    output_directory = Path(
        args.output_dir
    )

    print()
    print(
        "=" * 72
    )

    print(
        "DOMAIN-AWARE DATASET SPLITTER"
    )

    print(
        "=" * 72
    )

    print(
        f"Input: {input_file}"
    )

    dataframe, filter_stats = (
        load_and_filter_dataset(
            input_file=
                input_file,

            require_artifacts=
                args.require_artifacts,

            check_artifact_files=
                args.check_artifact_files,
        )
    )

    print(
        f"Usable rows before domain conflict removal: "
        f"{len(dataframe):,}"
    )

    clean_dataframe, conflict_rows = (
        remove_cross_class_domains(
            dataframe
        )
    )

    print(
        "Cross-class domain conflict rows removed: "
        f"{len(conflict_rows):,}"
    )

    if clean_dataframe.empty:

        raise ValueError(
            "No rows remain after filtering."
        )

    final_dataframe = (
        build_domain_aware_split(
            dataframe=
                clean_dataframe,

            train_ratio=
                args.train_ratio,

            validation_ratio=
                args.validation_ratio,

            test_ratio=
                args.test_ratio,

            random_seed=
                args.seed,
        )
    )

    validate_zero_domain_overlap(
        final_dataframe
    )

    summary = build_summary(
        dataframe=
            final_dataframe,

        filter_stats=
            filter_stats,

        conflict_rows=
            conflict_rows,

        ratios={
            "train":
                args.train_ratio,

            "validation":
                args.validation_ratio,

            "test":
                args.test_ratio,
        },

        random_seed=
            args.seed,
    )

    save_outputs(
        dataframe=
            final_dataframe,

        conflict_rows=
            conflict_rows,

        output_directory=
            output_directory,

        summary=
            summary,
    )

    print()
    print(
        "Final dataset:"
    )

    print(
        f"  Rows: "
        f"{len(final_dataframe):,}"
    )

    print(
        f"  Registrable domains: "
        f"{final_dataframe['registrable_domain'].nunique():,}"
    )

    print()

    for split in [
        "train",
        "validation",
        "test",
    ]:

        info = summary[
            "split_summary"
        ][
            split
        ]

        print(
            f"{split.upper():10s} "
            f"rows={info['rows']:,} | "
            f"domains={info['domains']:,} | "
            f"phishing={info['phishing_rows']:,} | "
            f"legitimate={info['legitimate_rows']:,}"
        )

    print()
    print(
        "Domain overlap validation: PASSED"
    )

    print(
        f"Outputs: {output_directory}"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
