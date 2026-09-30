from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd


DEFAULT_INPUTS = [
    (
        "data/processed/final_legitimate_collection.csv",
        "PhiUSIIL",
    ),
    (
        "data/processed/final_phishing_collection.csv",
        "PhiUSIIL",
    ),
    (
        "data/processed/final_phishtank_collection_1500.csv",
        "PhishTank",
    ),
    (
        "data/processed/final_phishtank_collection_next_1000.csv",
        "PhishTank",
    ),
    (
        "data/processed/final_tranco_collection.csv",
        "Tranco",
    ),
    (
        "data/processed/final_openphish_collection.csv",
        "OpenPhish",
    ),
]


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def normalize_source_dataset(
    series: pd.Series,
    fallback: str,
) -> pd.Series:
    values = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )

    invalid = values.str.lower().isin(
        {
            "",
            "nan",
            "none",
            "unknown",
            "unknown_or_legacy",
        }
    )

    values.loc[
        invalid
    ] = fallback

    return values


def canonical_url(value: str) -> str:
    """
    Create a conservative URL key for global duplicate removal.

    - strips whitespace
    - removes fragments
    - lowercases scheme and hostname
    - removes default ports
    - normalizes an empty path to "/"

    Query text is kept because different query URLs can represent
    different phishing pages.
    """

    raw = str(
        value
        or ""
    ).strip()

    if not raw:
        return ""

    try:
        parsed = urlsplit(
            raw
        )

        scheme = parsed.scheme.lower()

        hostname = (
            parsed.hostname
            or ""
        ).lower().rstrip(".")

        if not hostname:
            return raw

        port = parsed.port

        if (
            port is None
            or (
                scheme == "http"
                and port == 80
            )
            or (
                scheme == "https"
                and port == 443
            )
        ):
            netloc = hostname
        else:
            netloc = (
                f"{hostname}:{port}"
            )

        path = (
            parsed.path
            or "/"
        )

        return urlunsplit(
            (
                scheme,
                netloc,
                path,
                parsed.query,
                "",
            )
        )

    except Exception:
        return raw


def artifact_exists(
    value,
) -> bool:
    path_text = str(
        value
        or ""
    ).strip()

    if not path_text:
        return False

    return Path(
        path_text
    ).exists()


def load_collection(
    file_path: Path,
    fallback_source: str,
    check_artifact_files: bool,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Collection not found: {file_path}"
        )

    dataframe = pd.read_csv(
        file_path
    )

    required = [
        "crawl_status",
        "training_candidate",
        "visual_screenshot_saved",
        "graph_artifact_saved",
        "source_label",
        "source_url",
    ]

    missing = [
        column
        for column
        in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{file_path} is missing required columns: "
            + ", ".join(
                missing
            )
        )

    success = (
        dataframe[
            "crawl_status"
        ]
        .astype(str)
        .str.upper()
        .eq(
            "SUCCESS"
        )
    )

    training_candidate = dataframe[
        "training_candidate"
    ].map(
        as_bool
    )

    screenshot_saved = dataframe[
        "visual_screenshot_saved"
    ].map(
        as_bool
    )

    graph_saved = dataframe[
        "graph_artifact_saved"
    ].map(
        as_bool
    )

    eligible_mask = (
        success
        & training_candidate
        & screenshot_saved
        & graph_saved
    )

    eligible = dataframe.loc[
        eligible_mask
    ].copy()

    if check_artifact_files:
        if (
            "visual_screenshot_path"
            not in eligible.columns
        ):
            raise ValueError(
                f"{file_path} does not contain "
                "visual_screenshot_path."
            )

        if (
            "graph_artifact_path"
            not in eligible.columns
        ):
            raise ValueError(
                f"{file_path} does not contain "
                "graph_artifact_path."
            )

        screenshot_file_exists = (
            eligible[
                "visual_screenshot_path"
            ].map(
                artifact_exists
            )
        )

        graph_file_exists = (
            eligible[
                "graph_artifact_path"
            ].map(
                artifact_exists
            )
        )

        eligible = eligible.loc[
            screenshot_file_exists
            & graph_file_exists
        ].copy()

    # Make provenance explicit even for older PhiUSIIL rows.
    if (
        "source_dataset"
        not in eligible.columns
    ):
        eligible[
            "source_dataset"
        ] = fallback_source

    else:
        eligible[
            "source_dataset"
        ] = normalize_source_dataset(
            eligible[
                "source_dataset"
            ],
            fallback_source,
        )

    eligible[
        "source_collection_file"
    ] = file_path.name

    labels = pd.to_numeric(
        eligible[
            "source_label"
        ],
        errors="coerce",
    )

    valid_label = labels.isin(
        [
            0,
            1,
        ]
    )

    eligible = eligible.loc[
        valid_label
    ].copy()

    eligible[
        "source_label"
    ] = (
        pd.to_numeric(
            eligible[
                "source_label"
            ],
            errors="coerce",
        )
        .astype(
            int
        )
    )

    eligible[
        "class_name"
    ] = eligible[
        "source_label"
    ].map(
        {
            0:
                "phishing",

            1:
                "legitimate",
        }
    )

    eligible[
        "canonical_source_url"
    ] = eligible[
        "source_url"
    ].map(
        canonical_url
    )

    eligible = eligible[
        eligible[
            "canonical_source_url"
        ].ne(
            ""
        )
    ].copy()

    summary = {
        "file":
            str(
                file_path
            ),

        "fallback_source":
            fallback_source,

        "raw_rows":
            int(
                len(
                    dataframe
                )
            ),

        "success_rows":
            int(
                success.sum()
            ),

        "training_candidate_rows":
            int(
                training_candidate.sum()
            ),

        "screenshot_saved_rows":
            int(
                screenshot_saved.sum()
            ),

        "graph_saved_rows":
            int(
                graph_saved.sum()
            ),

        "eligible_rows_after_policy":
            int(
                eligible.shape[
                    0
                ]
            ),
    }

    return (
        eligible.reset_index(
            drop=True
        ),
        summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the clean multimodal Dataset V2 from "
            "PhiUSIIL, PhishTank, Tranco and OpenPhish "
            "collection outputs."
        )
    )

    parser.add_argument(
        "--output",
        default=(
            "data/processed/"
            "final_multimodal_v2_unbalanced.csv"
        ),
    )

    parser.add_argument(
        "--summary",
        default=(
            "data/processed/"
            "final_multimodal_v2_summary.json"
        ),
    )

    parser.add_argument(
        "--duplicates",
        default=(
            "data/processed/"
            "final_multimodal_v2_duplicate_urls.csv"
        ),
    )

    parser.add_argument(
        "--source-summary",
        default=(
            "data/processed/"
            "final_multimodal_v2_source_summary.csv"
        ),
    )

    parser.add_argument(
        "--check-artifact-files",
        action="store_true",
        help=(
            "Require screenshot PNG and graph JSON paths "
            "to exist on disk."
        ),
    )

    args = parser.parse_args()

    collections = []
    collection_summaries = []

    for file_text, fallback_source in DEFAULT_INPUTS:
        dataframe, summary = load_collection(
            file_path=Path(
                file_text
            ),
            fallback_source=fallback_source,
            check_artifact_files=
                args.check_artifact_files,
        )

        collections.append(
            dataframe
        )

        collection_summaries.append(
            summary
        )

    merged = pd.concat(
        collections,
        ignore_index=True,
        sort=False,
    )

    # --------------------------------------------------------
    # Global canonical URL duplicate report.
    # --------------------------------------------------------

    duplicate_mask = merged[
        "canonical_source_url"
    ].duplicated(
        keep=False
    )

    duplicate_rows = (
        merged.loc[
            duplicate_mask
        ]
        .sort_values(
            [
                "canonical_source_url",
                "source_dataset",
            ]
        )
        .copy()
    )

    # Keep one website observation per canonical URL.
    # Existing input order is intentional:
    # PhiUSIIL -> PhishTank -> Tranco -> OpenPhish.
    clean = (
        merged
        .drop_duplicates(
            subset=[
                "canonical_source_url",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Save outputs.
    # --------------------------------------------------------

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean.to_csv(
        output_path,
        index=False,
    )

    duplicates_path = Path(
        args.duplicates
    )

    duplicates_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    duplicate_rows.to_csv(
        duplicates_path,
        index=False,
    )

    source_summary = (
        clean.groupby(
            [
                "source_dataset",
                "class_name",
            ]
        )
        .size()
        .reset_index(
            name="rows"
        )
        .sort_values(
            [
                "class_name",
                "source_dataset",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    source_summary_path = Path(
        args.source_summary
    )

    source_summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_summary.to_csv(
        source_summary_path,
        index=False,
    )

    class_counts = (
        clean[
            "class_name"
        ]
        .value_counts()
        .to_dict()
    )

    summary_payload = {
        "collection_summaries":
            collection_summaries,

        "rows_before_global_url_dedup":
            int(
                len(
                    merged
                )
            ),

        "duplicate_rows_in_report":
            int(
                len(
                    duplicate_rows
                )
            ),

        "rows_removed_by_global_url_dedup":
            int(
                len(
                    merged
                )
                - len(
                    clean
                )
            ),

        "final_rows_before_domain_conflict_removal":
            int(
                len(
                    clean
                )
            ),

        "class_counts":
            {
                key:
                    int(
                        value
                    )
                for key, value
                in class_counts.items()
            },

        "source_counts":
            {
                str(
                    key
                ):
                    int(
                        value
                    )
                for key, value
                in clean[
                    "source_dataset"
                ]
                .value_counts()
                .to_dict()
                .items()
            },

        "note":
            (
                "Cross-class registrable-domain conflicts are "
                "not removed here. Run dataset_splitter on this "
                "output so conflict removal and domain-aware "
                "train/validation/test assignment happen in one "
                "controlled step."
            ),
    }

    summary_path = Path(
        args.summary
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary_payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "=" * 78
    )

    print(
        "MULTIMODAL DATASET V2 BUILDER"
    )

    print(
        "=" * 78
    )

    for item in collection_summaries:
        print(
            f"{item['fallback_source']:10s} "
            f"| file={Path(item['file']).name:45s} "
            f"| raw={item['raw_rows']:4d} "
            f"| eligible={item['eligible_rows_after_policy']:4d}"
        )

    print()

    print(
        f"Rows before global URL dedupe: "
        f"{len(merged)}"
    )

    print(
        f"Rows removed by global URL dedupe: "
        f"{len(merged) - len(clean)}"
    )

    print(
        f"Dataset V2 rows before domain conflict removal: "
        f"{len(clean)}"
    )

    print()

    print(
        "CLASS COUNTS"
    )

    print(
        clean[
            "class_name"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "SOURCE / CLASS COUNTS"
    )

    print(
        source_summary.to_string(
            index=False
        )
    )

    print()

    print(
        f"Dataset: {output_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print(
        f"Duplicate report: {duplicates_path}"
    )

    print(
        f"Source summary: {source_summary_path}"
    )

    print()

    print(
        "NEXT: run the existing domain-aware dataset_splitter "
        "on the V2 dataset."
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
