from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd

try:
    import tldextract
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: tldextract\n"
        "Install it with: pip install tldextract"
    ) from exc


RANDOM_SEED = 42

DEFAULT_TRANCO_COUNT = 15000
DEFAULT_CLOUDFLARE_COUNT = 5000
DEFAULT_PHISHTANK_COUNT = 25000

TRANCO_MIN_RANK = 100001
TRANCO_MAX_RANK = 300000

PHISHTANK_MAX_PER_DOMAIN = 2
OPENPHISH_MAX_PER_DOMAIN = 2

# Uses the packaged Public Suffix List snapshot.
# No live network fetch is required.
EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=None
)


def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def snapshot_date_from_name(
    path: Path,
) -> str | None:
    match = re.search(
        r"(\d{4}-\d{2}-\d{2})",
        path.name,
    )

    if match:
        return match.group(1)

    return None


def registrable_domain(
    value: str,
) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if "://" not in text:
        text = "https://" + text

    try:
        hostname = (
            urlsplit(text)
            .hostname
        )
    except Exception:
        return None

    if not hostname:
        return None

    hostname = (
        hostname
        .strip(".")
        .lower()
    )

    result = EXTRACT(
        hostname
    )

    if (
        not result.domain
        or not result.suffix
    ):
        return hostname

    return (
        f"{result.domain}."
        f"{result.suffix}"
    ).lower()


def normalize_url(
    value: str,
) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        text,
    ):
        text = "https://" + text

    try:
        parsed = urlsplit(
            text
        )
    except Exception:
        return None

    if not parsed.hostname:
        return None

    scheme = (
        parsed.scheme.lower()
        if parsed.scheme
        else "https"
    )

    hostname = (
        parsed.hostname
        .strip(".")
        .lower()
    )

    port = parsed.port

    if port is None:
        netloc = hostname
    else:
        netloc = (
            f"{hostname}:{port}"
        )

    path = parsed.path or ""

    normalized = urlunsplit(
        (
            scheme,
            netloc,
            path,
            parsed.query,
            "",
        )
    )

    return normalized


def domain_to_url(
    domain: str,
) -> str | None:
    domain = str(
        domain
    ).strip()

    if not domain:
        return None

    return normalize_url(
        "https://" + domain
    )


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    lowered = {
        str(column).strip().lower():
            column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.lower()

        if key in lowered:
            return lowered[key]

    return None


def read_tranco(
    path: Path,
) -> pd.DataFrame:
    # Official Tranco CSV is headerless:
    # rank,domain
    df = pd.read_csv(
        path,
        header=None,
        names=[
            "rank",
            "domain",
        ],
        usecols=[0, 1],
    )

    df["rank"] = pd.to_numeric(
        df["rank"],
        errors="coerce",
    )

    df["domain"] = (
        df["domain"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[
        df["rank"].notna()
        & df["domain"].ne("")
    ].copy()

    return df


def read_cloudflare(
    path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(
        path
    )

    domain_column = first_existing_column(
        df,
        [
            "domain",
            "hostname",
            "host",
        ],
    )

    if domain_column is None:
        if df.shape[1] != 1:
            raise ValueError(
                "Could not identify the Cloudflare domain column."
            )

        domain_column = df.columns[0]

    out = pd.DataFrame(
        {
            "domain":
                df[
                    domain_column
                ]
                .astype(str)
                .str.strip()
                .str.lower()
        }
    )

    out = out[
        out["domain"].ne("")
    ].copy()

    return out


def read_phishtank(
    path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "url",
        "verified",
        "online",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "PhishTank file missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    df = df.copy()

    df["verified_normalized"] = (
        df["verified"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["online_normalized"] = (
        df["online"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[
        df["verified_normalized"]
        .eq("yes")
        & df["online_normalized"]
        .eq("yes")
    ].copy()

    if "verification_time" in df.columns:
        df[
            "verification_time_parsed"
        ] = pd.to_datetime(
            df["verification_time"],
            errors="coerce",
            utc=True,
        )

        df = df.sort_values(
            "verification_time_parsed",
            ascending=False,
            na_position="last",
        )

    return df


def read_openphish(
    path: Path,
) -> pd.DataFrame:
    # The supplied OpenPhish XLSX is headerless.
    # Reading header=None preserves the first URL.
    df = pd.read_excel(
        path,
        header=None,
    )

    if df.shape[1] < 1:
        raise ValueError(
            "OpenPhish file has no columns."
        )

    out = pd.DataFrame(
        {
            "url":
                df.iloc[:, 0]
                .astype(str)
                .str.strip()
        }
    )

    out = out[
        out["url"].ne("")
    ].copy()

    return out


def load_v2_overlap(
    path: Path,
) -> tuple[set[str], set[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset V2 not found: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    url_column = first_existing_column(
        df,
        [
            "source_url",
            "requested_url",
            "url",
            "original_url",
            "final_url",
        ],
    )

    if url_column is None:
        raise ValueError(
            "Could not identify a URL column in Dataset V2."
        )

    normalized_urls: set[str] = set()
    domains: set[str] = set()

    for value in (
        df[url_column]
        .dropna()
        .astype(str)
    ):
        normalized = normalize_url(
            value
        )

        domain = registrable_domain(
            value
        )

        if normalized:
            normalized_urls.add(
                normalized
            )

        if domain:
            domains.add(
                domain
            )

    return (
        normalized_urls,
        domains,
    )


def prepare_common_columns(
    df: pd.DataFrame,
    source_dataset: str,
    raw_path: Path,
    actual_phishing: int,
) -> pd.DataFrame:
    out = df.copy()

    out[
        "source_dataset"
    ] = source_dataset

    out[
        "raw_source_file"
    ] = raw_path.name

    out[
        "source_snapshot_date"
    ] = snapshot_date_from_name(
        raw_path
    )

    out[
        "pool_built_at_utc"
    ] = utc_now_iso()

    out[
        "actual_phishing"
    ] = int(
        actual_phishing
    )

    # Existing crawler convention:
    # phishing = 0, legitimate = 1.
    out[
        "crawler_label"
    ] = (
        0
        if actual_phishing == 1
        else 1
    )

    out[
        "candidate_class"
    ] = (
        "phishing"
        if actual_phishing == 1
        else "legitimate"
    )

    return out


def remove_v2_overlap(
    df: pd.DataFrame,
    v2_urls: set[str],
    v2_domains: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlap_mask = (
        df["source_url"]
        .isin(v2_urls)
        | df[
            "registrable_domain"
        ].isin(
            v2_domains
        )
    )

    excluded = df[
        overlap_mask
    ].copy()

    kept = df[
        ~overlap_mask
    ].copy()

    return (
        kept,
        excluded,
    )


def deduplicate_candidates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return (
        df
        .drop_duplicates(
            subset=[
                "source_url",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )


def cap_per_domain(
    df: pd.DataFrame,
    max_per_domain: int,
) -> pd.DataFrame:
    if max_per_domain <= 0:
        return df

    return (
        df
        .groupby(
            "registrable_domain",
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


def deterministic_sample(
    df: pd.DataFrame,
    count: int,
) -> pd.DataFrame:
    if count <= 0:
        return df.iloc[0:0].copy()

    if len(df) <= count:
        return df.copy()

    return (
        df.sample(
            n=count,
            random_state=RANDOM_SEED,
        )
        .reset_index(
            drop=True
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build large fresh external candidate pools for "
            "multimodal phishing evaluation."
        )
    )

    parser.add_argument(
        "--tranco",
        default=(
            "data/raw/final_external_2026/"
            "tranco_2026-09-30.csv"
        ),
    )

    parser.add_argument(
        "--cloudflare",
        default=(
            "data/raw/final_external_2026/"
            "cloudflare_radar_2026-09-30.csv"
        ),
    )

    parser.add_argument(
        "--phishtank",
        default=(
            "data/raw/final_external_2026/"
            "phishtank_2026-09-30.csv"
        ),
    )

    parser.add_argument(
        "--openphish",
        default=(
            "data/raw/final_external_2026/"
            "openphish_2026-09-30.xlsx"
        ),
    )

    parser.add_argument(
        "--v2",
        default=(
            "data/processed/"
            "final_multimodal_v2_unbalanced.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/processed/"
            "final_external_2026"
        ),
    )

    parser.add_argument(
        "--tranco-count",
        type=int,
        default=DEFAULT_TRANCO_COUNT,
    )

    parser.add_argument(
        "--cloudflare-count",
        type=int,
        default=DEFAULT_CLOUDFLARE_COUNT,
    )

    parser.add_argument(
        "--phishtank-count",
        type=int,
        default=DEFAULT_PHISHTANK_COUNT,
    )

    args = parser.parse_args()

    tranco_path = Path(
        args.tranco
    )

    cloudflare_path = Path(
        args.cloudflare
    )

    phishtank_path = Path(
        args.phishtank
    )

    openphish_path = Path(
        args.openphish
    )

    v2_path = Path(
        args.v2
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        tranco_path,
        cloudflare_path,
        phishtank_path,
        openphish_path,
        v2_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    print()
    print("=" * 96)
    print("FINAL EXTERNAL LARGE CANDIDATE POOL BUILDER")
    print("=" * 96)

    v2_urls, v2_domains = load_v2_overlap(
        v2_path
    )

    print(
        f"Dataset V2 URLs loaded: {len(v2_urls):,}"
    )

    print(
        f"Dataset V2 registrable domains loaded: "
        f"{len(v2_domains):,}"
    )

    excluded_frames: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # TRANCO LEGITIMATE CANDIDATES
    # ------------------------------------------------------------------
    tranco = read_tranco(
        tranco_path
    )

    tranco = tranco[
        tranco["rank"].between(
            TRANCO_MIN_RANK,
            TRANCO_MAX_RANK,
            inclusive="both",
        )
    ].copy()

    tranco[
        "source_url"
    ] = tranco[
        "domain"
    ].map(
        domain_to_url
    )

    tranco[
        "registrable_domain"
    ] = tranco[
        "source_url"
    ].map(
        registrable_domain
    )

    tranco = tranco.dropna(
        subset=[
            "source_url",
            "registrable_domain",
        ]
    )

    tranco = prepare_common_columns(
        tranco,
        source_dataset=
            "Tranco_Final_External",

        raw_path=
            tranco_path,

        actual_phishing=
            0,
    )

    tranco, excluded = remove_v2_overlap(
        tranco,
        v2_urls,
        v2_domains,
    )

    if not excluded.empty:
        excluded[
            "exclusion_reason"
        ] = "dataset_v2_url_or_domain_overlap"

        excluded_frames.append(
            excluded
        )

    tranco = (
        tranco
        .drop_duplicates(
            subset=[
                "registrable_domain",
            ],
            keep="first",
        )
    )

    tranco = deterministic_sample(
        tranco,
        args.tranco_count,
    )

    # ------------------------------------------------------------------
    # CLOUDFLARE LEGITIMATE CANDIDATES
    # ------------------------------------------------------------------
    cloudflare = read_cloudflare(
        cloudflare_path
    )

    cloudflare[
        "source_url"
    ] = cloudflare[
        "domain"
    ].map(
        domain_to_url
    )

    cloudflare[
        "registrable_domain"
    ] = cloudflare[
        "source_url"
    ].map(
        registrable_domain
    )

    cloudflare = cloudflare.dropna(
        subset=[
            "source_url",
            "registrable_domain",
        ]
    )

    cloudflare = prepare_common_columns(
        cloudflare,
        source_dataset=
            "Cloudflare_Radar_Final_External",

        raw_path=
            cloudflare_path,

        actual_phishing=
            0,
    )

    cloudflare, excluded = remove_v2_overlap(
        cloudflare,
        v2_urls,
        v2_domains,
    )

    if not excluded.empty:
        excluded[
            "exclusion_reason"
        ] = "dataset_v2_url_or_domain_overlap"

        excluded_frames.append(
            excluded
        )

    tranco_domains = set(
        tranco[
            "registrable_domain"
        ]
    )

    cf_overlap = cloudflare[
        "registrable_domain"
    ].isin(
        tranco_domains
    )

    if cf_overlap.any():
        excluded = cloudflare[
            cf_overlap
        ].copy()

        excluded[
            "exclusion_reason"
        ] = "duplicate_legitimate_domain_across_sources"

        excluded_frames.append(
            excluded
        )

        cloudflare = cloudflare[
            ~cf_overlap
        ].copy()

    cloudflare = (
        cloudflare
        .drop_duplicates(
            subset=[
                "registrable_domain",
            ],
            keep="first",
        )
    )

    cloudflare = deterministic_sample(
        cloudflare,
        args.cloudflare_count,
    )

    legitimate = pd.concat(
        [
            tranco,
            cloudflare,
        ],
        ignore_index=True,
        sort=False,
    )

    legitimate = deduplicate_candidates(
        legitimate
    )

    # ------------------------------------------------------------------
    # PHISHTANK PHISHING CANDIDATES
    # ------------------------------------------------------------------
    phishtank = read_phishtank(
        phishtank_path
    )

    phishtank[
        "source_url"
    ] = phishtank[
        "url"
    ].map(
        normalize_url
    )

    phishtank[
        "registrable_domain"
    ] = phishtank[
        "source_url"
    ].map(
        registrable_domain
    )

    phishtank = phishtank.dropna(
        subset=[
            "source_url",
            "registrable_domain",
        ]
    )

    phishtank = prepare_common_columns(
        phishtank,
        source_dataset=
            "PhishTank_Final_External",

        raw_path=
            phishtank_path,

        actual_phishing=
            1,
    )

    phishtank, excluded = remove_v2_overlap(
        phishtank,
        v2_urls,
        v2_domains,
    )

    if not excluded.empty:
        excluded[
            "exclusion_reason"
        ] = "dataset_v2_url_or_domain_overlap"

        excluded_frames.append(
            excluded
        )

    phishtank = deduplicate_candidates(
        phishtank
    )

    phishtank = cap_per_domain(
        phishtank,
        PHISHTANK_MAX_PER_DOMAIN,
    )

    phishtank = deterministic_sample(
        phishtank,
        args.phishtank_count,
    )

    # ------------------------------------------------------------------
    # OPENPHISH PHISHING CANDIDATES
    # ------------------------------------------------------------------
    openphish = read_openphish(
        openphish_path
    )

    openphish[
        "source_url"
    ] = openphish[
        "url"
    ].map(
        normalize_url
    )

    openphish[
        "registrable_domain"
    ] = openphish[
        "source_url"
    ].map(
        registrable_domain
    )

    openphish = openphish.dropna(
        subset=[
            "source_url",
            "registrable_domain",
        ]
    )

    openphish = prepare_common_columns(
        openphish,
        source_dataset=
            "OpenPhish_Final_External",

        raw_path=
            openphish_path,

        actual_phishing=
            1,
    )

    openphish, excluded = remove_v2_overlap(
        openphish,
        v2_urls,
        v2_domains,
    )

    if not excluded.empty:
        excluded[
            "exclusion_reason"
        ] = "dataset_v2_url_or_domain_overlap"

        excluded_frames.append(
            excluded
        )

    phishtank_urls = set(
        phishtank[
            "source_url"
        ]
    )

    phishtank_domains = set(
        phishtank[
            "registrable_domain"
        ]
    )

    openphish_duplicate = (
        openphish[
            "source_url"
        ].isin(
            phishtank_urls
        )
    )

    if openphish_duplicate.any():
        excluded = openphish[
            openphish_duplicate
        ].copy()

        excluded[
            "exclusion_reason"
        ] = "duplicate_phishing_url_across_sources"

        excluded_frames.append(
            excluded
        )

        openphish = openphish[
            ~openphish_duplicate
        ].copy()

    openphish = deduplicate_candidates(
        openphish
    )

    openphish = cap_per_domain(
        openphish,
        OPENPHISH_MAX_PER_DOMAIN,
    )

    phishing = pd.concat(
        [
            phishtank,
            openphish,
        ],
        ignore_index=True,
        sort=False,
    )

    phishing = deduplicate_candidates(
        phishing
    )

    # ------------------------------------------------------------------
    # REMOVE CROSS-CLASS DOMAIN CONFLICTS
    # ------------------------------------------------------------------
    legitimate_domains = set(
        legitimate[
            "registrable_domain"
        ]
    )

    phishing_domains = set(
        phishing[
            "registrable_domain"
        ]
    )

    conflict_domains = (
        legitimate_domains
        & phishing_domains
    )

    conflicts = pd.DataFrame(
        {
            "registrable_domain":
                sorted(
                    conflict_domains
                )
        }
    )

    conflicts.to_csv(
        output_dir
        / "cross_class_domain_conflicts.csv",
        index=False,
    )

    if conflict_domains:
        legit_conflict = legitimate[
            legitimate[
                "registrable_domain"
            ].isin(
                conflict_domains
            )
        ].copy()

        legit_conflict[
            "exclusion_reason"
        ] = "cross_class_domain_conflict"

        excluded_frames.append(
            legit_conflict
        )

        phishing_conflict = phishing[
            phishing[
                "registrable_domain"
            ].isin(
                conflict_domains
            )
        ].copy()

        phishing_conflict[
            "exclusion_reason"
        ] = "cross_class_domain_conflict"

        excluded_frames.append(
            phishing_conflict
        )

        legitimate = legitimate[
            ~legitimate[
                "registrable_domain"
            ].isin(
                conflict_domains
            )
        ].copy()

        phishing = phishing[
            ~phishing[
                "registrable_domain"
            ].isin(
                conflict_domains
            )
        ].copy()

    # ------------------------------------------------------------------
    # FINAL OUTPUTS
    # ------------------------------------------------------------------
    legitimate = (
        legitimate
        .reset_index(
            drop=True
        )
    )

    phishing = (
        phishing
        .reset_index(
            drop=True
        )
    )

    combined = pd.concat(
        [
            legitimate,
            phishing,
        ],
        ignore_index=True,
        sort=False,
    )

    combined = combined.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(
        drop=True
    )

    legitimate.to_csv(
        output_dir
        / "legitimate_candidate_pool.csv",
        index=False,
    )

    phishing.to_csv(
        output_dir
        / "phishing_candidate_pool.csv",
        index=False,
    )

    combined.to_csv(
        output_dir
        / "final_external_candidate_pool.csv",
        index=False,
    )

    if excluded_frames:
        excluded_all = pd.concat(
            excluded_frames,
            ignore_index=True,
            sort=False,
        )
    else:
        excluded_all = pd.DataFrame()

    excluded_all.to_csv(
        output_dir
        / "excluded_candidates.csv",
        index=False,
    )

    summary = (
        combined
        .groupby(
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

    summary.to_csv(
        output_dir
        / "external_candidate_source_summary.csv",
        index=False,
    )

    manifest = {
        "generated_at_utc":
            utc_now_iso(),

        "purpose":
            "fresh_external_evaluation_candidate_pool",

        "evaluation_only":
            True,

        "random_seed":
            RANDOM_SEED,

        "v2_overlap_removed":
            True,

        "v2_exact_url_count":
            len(
                v2_urls
            ),

        "v2_registrable_domain_count":
            len(
                v2_domains
            ),

        "tranco_rank_window": {
            "minimum":
                TRANCO_MIN_RANK,

            "maximum":
                TRANCO_MAX_RANK,
        },

        "phishing_max_urls_per_domain": {
            "PhishTank":
                PHISHTANK_MAX_PER_DOMAIN,

            "OpenPhish":
                OPENPHISH_MAX_PER_DOMAIN,
        },

        "candidate_counts": {
            "legitimate":
                len(
                    legitimate
                ),

            "phishing":
                len(
                    phishing
                ),

            "total":
                len(
                    combined
                ),
        },

        "raw_files": {
            "tranco":
                tranco_path.name,

            "cloudflare":
                cloudflare_path.name,

            "phishtank":
                phishtank_path.name,

            "openphish":
                openphish_path.name,
        },

        "important_note":
            (
                "These are candidate labels from source feeds, not yet "
                "final evaluation samples. Existing crawl success, "
                "content/label QA, screenshot availability and graph "
                "artifact checks must still be applied. Cloudflare Radar "
                "domains are only legitimate candidates and require the "
                "same QA; popularity is not proof of legitimacy."
            ),
    }

    with (
        output_dir
        / "external_candidate_pool_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print()
    print("RAW FILE CHECK")
    print(
        f"Tranco rows:      {len(read_tranco(tranco_path)):,}"
    )
    print(
        f"Cloudflare rows:  {len(read_cloudflare(cloudflare_path)):,}"
    )
    print(
        f"PhishTank online verified rows: "
        f"{len(read_phishtank(phishtank_path)):,}"
    )
    print(
        f"OpenPhish rows:   {len(read_openphish(openphish_path)):,}"
    )

    print()
    print("FINAL CANDIDATE POOL")
    print(
        f"Legitimate: {len(legitimate):,}"
    )
    print(
        f"Phishing:   {len(phishing):,}"
    )
    print(
        f"Total:      {len(combined):,}"
    )

    print()
    print("SOURCE SUMMARY")
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Cross-class domain conflicts removed: "
        f"{len(conflict_domains):,}"
    )

    print(
        f"Excluded records saved: "
        f"{len(excluded_all):,}"
    )

    print()
    print(
        f"Outputs: {output_dir}"
    )
    print("=" * 96)


if __name__ == "__main__":
    main()
