from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import tldextract

RANDOM_SEED = 42
EXTRACT = tldextract.TLDExtract(suffix_list_urls=None)


def as_bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def registrable_domain(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    if "://" not in text:
        text = "https://" + text

    try:
        host = urlsplit(text).hostname
    except Exception:
        return ""

    if not host:
        return ""

    host = host.strip(".").lower()
    result = EXTRACT(host)

    if result.domain and result.suffix:
        return f"{result.domain}.{result.suffix}".lower()

    return host


def choose_file(directory: Path, kind: str) -> Path:
    fixed = directory / f"final_external_{kind}_crawled_fixed.csv"
    if fixed.exists():
        return fixed

    return directory / f"final_external_{kind}_crawled.csv"


def load_collection(path: Path, expected_class: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    required = {
        "source_url",
        "source_label",
        "class_name",
        "crawl_status",
        "training_candidate",
        "visual_screenshot_saved",
        "graph_artifact_saved",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} missing columns: "
            + ", ".join(sorted(missing))
        )

    df = df.copy()
    df["class_name"] = (
        df["class_name"].astype(str).str.strip().str.lower()
    )

    return df[df["class_name"].eq(expected_class)].copy()


def eligibility_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["crawl_status"].astype(str).str.upper().eq("SUCCESS")
        & as_bool_series(df["training_candidate"])
        & as_bool_series(df["visual_screenshot_saved"])
        & as_bool_series(df["graph_artifact_saved"])
    )


def load_v2_domains(path: Path) -> set[str]:
    df = pd.read_csv(path, low_memory=False)

    url_column = next(
        (
            c for c in [
                "source_url",
                "requested_url",
                "URL",
                "url",
                "final_url",
            ]
            if c in df.columns
        ),
        None,
    )

    if url_column is None:
        raise ValueError("No URL column found in Dataset V2.")

    domains = {
        registrable_domain(v)
        for v in df[url_column].dropna().astype(str)
    }
    domains.discard("")
    return domains


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crawl-dir",
        default="data/processed/final_external_2026/crawled",
    )
    parser.add_argument(
        "--v2",
        default="data/processed/final_multimodal_v2_unbalanced.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/final_external_2026/final",
    )
    args = parser.parse_args()

    crawl_dir = Path(args.crawl_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    legitimate_path = choose_file(crawl_dir, "legitimate")
    phishing_path = choose_file(crawl_dir, "phishing")

    legitimate = load_collection(legitimate_path, "legitimate")
    phishing = load_collection(phishing_path, "phishing")

    legitimate = legitimate[eligibility_mask(legitimate)].copy()
    phishing = phishing[eligibility_mask(phishing)].copy()

    for df in [legitimate, phishing]:
        df["final_eval_registrable_domain"] = (
            df["source_url"].map(registrable_domain)
        )
        df["actual_phishing"] = (
            df["class_name"].eq("phishing").astype(int)
        )

    legitimate = (
        legitimate.drop_duplicates("source_url")
        .reset_index(drop=True)
    )
    phishing = (
        phishing.drop_duplicates("source_url")
        .reset_index(drop=True)
    )

    v2_domains = load_v2_domains(Path(args.v2))

    legit_overlap = legitimate[
        "final_eval_registrable_domain"
    ].isin(v2_domains)

    phish_overlap = phishing[
        "final_eval_registrable_domain"
    ].isin(v2_domains)

    excluded_v2 = pd.concat(
        [
            legitimate[legit_overlap],
            phishing[phish_overlap],
        ],
        ignore_index=True,
        sort=False,
    )

    legitimate = legitimate[~legit_overlap].copy()
    phishing = phishing[~phish_overlap].copy()

    legit_domains = set(
        legitimate["final_eval_registrable_domain"]
    )
    phish_domains = set(
        phishing["final_eval_registrable_domain"]
    )
    conflicts = legit_domains & phish_domains

    if conflicts:
        legitimate = legitimate[
            ~legitimate[
                "final_eval_registrable_domain"
            ].isin(conflicts)
        ].copy()

        phishing = phishing[
            ~phishing[
                "final_eval_registrable_domain"
            ].isin(conflicts)
        ].copy()

    full = pd.concat(
        [legitimate, phishing],
        ignore_index=True,
        sort=False,
    ).sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    per_class = min(len(legitimate), len(phishing))

    balanced = pd.concat(
        [
            legitimate.sample(
                n=per_class,
                random_state=RANDOM_SEED,
            ),
            phishing.sample(
                n=per_class,
                random_state=RANDOM_SEED,
            ),
        ],
        ignore_index=True,
        sort=False,
    ).sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    legitimate.to_csv(
        output_dir / "final_external_legitimate_eligible.csv",
        index=False,
    )
    phishing.to_csv(
        output_dir / "final_external_phishing_eligible.csv",
        index=False,
    )
    full.to_csv(
        output_dir / "final_external_full.csv",
        index=False,
    )
    balanced.to_csv(
        output_dir / "final_external_balanced.csv",
        index=False,
    )
    excluded_v2.to_csv(
        output_dir / "excluded_dataset_v2_domain_overlap.csv",
        index=False,
    )

    source_summary = (
        full.groupby(
            ["class_name", "source_dataset"],
            dropna=False,
        )
        .size()
        .reset_index(name="sample_count")
    )

    source_summary.to_csv(
        output_dir / "final_external_source_summary.csv",
        index=False,
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True,
        "legitimate_input": str(legitimate_path),
        "phishing_input": str(phishing_path),
        "eligibility": [
            "crawl_status == SUCCESS",
            "training_candidate == True",
            "visual_screenshot_saved == True",
            "graph_artifact_saved == True",
            "exact URL deduplicated",
            "Dataset V2 registrable-domain overlap removed",
            "cross-class registrable-domain conflicts removed",
        ],
        "final_counts": {
            "legitimate": len(legitimate),
            "phishing": len(phishing),
            "full_total": len(full),
            "balanced_per_class": per_class,
            "balanced_total": len(balanced),
        },
        "dataset_v2_overlap_rows_excluded": len(excluded_v2),
        "cross_class_conflict_domains": len(conflicts),
        "model_target_convention": {
            "phishing": 1,
            "legitimate": 0,
        },
        "note": (
            "Final external evaluation only. Do not retrain models "
            "or tune features, fusion weights, threshold, calibration, "
            "or risk logic using these results."
        ),
    }

    with (
        output_dir / "final_external_manifest.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 88)
    print("FINAL EXTERNAL DATASET")
    print("=" * 88)
    print(f"Legitimate eligible: {len(legitimate):,}")
    print(f"Phishing eligible:   {len(phishing):,}")
    print(f"Full total:          {len(full):,}")
    print()
    print(f"Balanced per class:  {per_class:,}")
    print(f"Balanced total:      {len(balanced):,}")
    print()
    print(
        "Dataset V2 overlap rows excluded: "
        f"{len(excluded_v2):,}"
    )
    print(
        "Cross-class conflict domains: "
        f"{len(conflicts):,}"
    )
    print()
    print(source_summary.to_string(index=False))
    print()
    print(f"Outputs: {output_dir}")
    print("=" * 88)


if __name__ == "__main__":
    main()
