from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import socket
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.website_detection.inference.final_website_detector import FinalWebsiteDetector

APP_TITLE = "J26-IT-424 Website Phishing Detection API"
APP_VERSION = "1.2.0"

ANALYSIS_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()
DETECTOR: FinalWebsiteDetector | None = None

LIVE_LOG_DIR = Path("results/live_case_studies")
LIVE_LOG_CSV = LIVE_LOG_DIR / "live_analysis_log.csv"
LIVE_LOG_JSONL = LIVE_LOG_DIR / "live_analysis_log.jsonl"

LIVE_LOG_COLUMNS = [
    "timestamp_utc",
    "input_url",
    "requested_url",
    "final_url",
    "crawl_status",
    "status_code",
    "title",
    "analysis_status",
    "lightgbm_probability",
    "resnet18_probability",
    "graphsage_probability",
    "raw_fusion_probability",
    "raw_risk_score",
    "raw_risk_level",
    "raw_prediction",
    "majority_branch_prediction",
    "evidence_status",
    "review_required",
    "operational_decision",
    "operational_classification",
    "operational_action",
    "operational_status",
    "conflict_type",
    "navigation_wait_until",
    "navigation_timeout_recovered",
]


class AnalyzeWebsiteRequest(BaseModel):
    url: str = Field(
        ...,
        min_length=4,
        max_length=2048,
        description="Public HTTP/HTTPS website URL to analyse.",
    )


class HealthResponse(BaseModel):
    status: str
    detector_loaded: bool
    version: str
    live_log_csv: str


def normalize_candidate_url(value: str) -> str:
    url = str(value).strip()

    if not url:
        raise ValueError("URL is empty.")

    if "://" not in url:
        url = "https://" + url

    parsed = urlsplit(url)

    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs are allowed.")

    if not parsed.hostname:
        raise ValueError("URL does not contain a valid hostname.")

    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded credentials are not allowed.")

    return url


def safe_log_url(value: Any) -> str:
    """Strip query/fragment/credentials before persistent logging."""
    text = str(value or "").strip()
    if not text:
        return ""

    try:
        parsed = urlsplit(text)
        hostname = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        netloc = f"{hostname}{port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except Exception:
        return text


def is_forbidden_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return any(
        [
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_reserved,
            ip.is_unspecified,
        ]
    )


def validate_public_destination(url: str) -> None:
    parsed = urlsplit(url)
    hostname = parsed.hostname

    if not hostname:
        raise ValueError("URL has no hostname.")

    normalized_host = hostname.rstrip(".").lower()

    if normalized_host in {"localhost", "localhost.localdomain"}:
        raise ValueError("Localhost URLs are not allowed.")

    try:
        direct_ip = ipaddress.ip_address(normalized_host)
    except ValueError:
        direct_ip = None

    if direct_ip is not None:
        if is_forbidden_ip(str(direct_ip)):
            raise ValueError(
                "Private, loopback, link-local, reserved, multicast "
                "or unspecified IP addresses are not allowed."
            )
        return

    try:
        address_info = socket.getaddrinfo(
            normalized_host,
            parsed.port or (443 if parsed.scheme.lower() == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError(
            f"Hostname could not be resolved: {normalized_host}"
        ) from exc

    addresses = {
        item[4][0]
        for item in address_info
        if item and len(item) >= 5 and item[4]
    }

    if not addresses:
        raise ValueError("Hostname resolved to no usable network address.")

    forbidden = [address for address in addresses if is_forbidden_ip(address)]
    if forbidden:
        raise ValueError(
            "URL resolves to a non-public network address "
            f"({', '.join(sorted(forbidden))})."
        )


def get_detector() -> FinalWebsiteDetector:
    if DETECTOR is None:
        raise RuntimeError("Detector has not been initialized.")
    return DETECTOR


def apply_operational_decision(result: dict[str, Any]) -> dict[str, Any]:
    """Add safety-oriented conflict handling without changing model outputs."""
    enriched = dict(result)

    raw_prediction = result.get("prediction")
    raw_score = result.get("risk_score")
    raw_risk_level = result.get("risk_level")
    fusion_probability = result.get("fusion_probability_raw")

    analysis_succeeded = (
        result.get("analysis_status") == "SUCCESS" and raw_prediction is not None
    )

    if not analysis_succeeded:
        enriched["operational_result"] = {
            "decision": "REVIEW_REQUIRED",
            "classification": None,
            "action": "RETRY_OR_MANUAL_REVIEW",
            "status": "CRAWL_OR_INFERENCE_UNAVAILABLE",
            "conflict_type": "CRAWL_UNAVAILABLE",
            "reason": result.get("reason")
            or "No successful model inference was available.",
            "raw_fusion_prediction": raw_prediction,
            "raw_fusion_probability": fusion_probability,
            "raw_risk_score": raw_score,
            "raw_risk_level": raw_risk_level,
            "majority_branch_prediction": result.get("majority_branch_prediction"),
            "model_outputs_modified": False,
        }
        return enriched

    majority_prediction = result.get("majority_branch_prediction")
    evidence_status = str(result.get("evidence_status", ""))
    majority_disagrees = bool(
        result.get("majority_disagrees_with_fusion", False)
    )
    review_required = bool(result.get("review_required", False))

    if majority_disagrees:
        operational_result = {
            "decision": "REVIEW_REQUIRED",
            "classification": None,
            "action": "MANUAL_REVIEW",
            "status": "FUSION_MAJORITY_CONFLICT",
            "conflict_type": "FUSION_MAJORITY_CONFLICT",
            "reason": (
                "The weighted fusion prediction conflicts with the majority vote "
                "of the three model branches. The raw research-model output is "
                "preserved, but no automatic operational class is issued."
            ),
        }
    elif evidence_status == "MIXED_EVIDENCE" or review_required:
        operational_result = {
            "decision": str(raw_prediction),
            "classification": str(raw_prediction),
            "action": "MANUAL_REVIEW",
            "status": "CLASSIFICATION_WITH_REVIEW",
            "conflict_type": "MINORITY_BRANCH_DISAGREEMENT",
            "reason": (
                "The fusion prediction agrees with the branch majority, but at "
                "least one branch disagrees. The raw class is retained and "
                "analyst review is recommended."
            ),
        }
    else:
        operational_result = {
            "decision": str(raw_prediction),
            "classification": str(raw_prediction),
            "action": "AUTOMATED_RESULT",
            "status": "FULL_MODEL_AGREEMENT",
            "conflict_type": "NONE",
            "reason": (
                "The model branches are consistent with an automated "
                "operational result."
            ),
        }

    operational_result.update(
        {
            "raw_fusion_prediction": raw_prediction,
            "raw_fusion_probability": fusion_probability,
            "raw_risk_score": raw_score,
            "raw_risk_level": raw_risk_level,
            "majority_branch_prediction": majority_prediction,
            "evidence_status": evidence_status,
            "review_required": review_required,
            "model_outputs_modified": False,
        }
    )

    enriched["operational_result"] = operational_result
    return enriched


def _branch_probability(result: dict[str, Any], branch: str) -> Any:
    branch_scores = result.get("branch_scores")
    if not isinstance(branch_scores, dict):
        return None
    branch_data = branch_scores.get(branch)
    if not isinstance(branch_data, dict):
        return None
    return branch_data.get("probability")


def build_live_log_row(result: dict[str, Any]) -> dict[str, Any]:
    op = result.get("operational_result")
    if not isinstance(op, dict):
        op = {}

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_url": safe_log_url(result.get("input_url")),
        "requested_url": safe_log_url(result.get("requested_url")),
        "final_url": safe_log_url(result.get("final_url")),
        "crawl_status": result.get("crawl_status"),
        "status_code": result.get("status_code"),
        "title": result.get("title"),
        "analysis_status": result.get("analysis_status"),
        "lightgbm_probability": _branch_probability(result, "lightgbm"),
        "resnet18_probability": _branch_probability(result, "resnet18"),
        "graphsage_probability": _branch_probability(result, "graphsage"),
        "raw_fusion_probability": result.get("fusion_probability_raw"),
        "raw_risk_score": result.get("risk_score"),
        "raw_risk_level": result.get("risk_level"),
        "raw_prediction": result.get("prediction"),
        "majority_branch_prediction": result.get("majority_branch_prediction"),
        "evidence_status": result.get("evidence_status"),
        "review_required": result.get("review_required"),
        "operational_decision": op.get("decision"),
        "operational_classification": op.get("classification"),
        "operational_action": op.get("action"),
        "operational_status": op.get("status"),
        "conflict_type": op.get("conflict_type"),
        "navigation_wait_until": result.get("navigation_wait_until"),
        "navigation_timeout_recovered": result.get(
            "navigation_timeout_recovered"
        ),
    }


def append_live_case_study_log(result: dict[str, Any]) -> None:
    LIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    row = build_live_log_row(result)

    with LOG_LOCK:
        csv_exists = LIVE_LOG_CSV.exists()

        with LIVE_LOG_CSV.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=LIVE_LOG_COLUMNS)
            if not csv_exists:
                writer.writeheader()
            writer.writerow(row)

        with LIVE_LOG_JSONL.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_recent_live_logs(limit: int) -> list[dict[str, Any]]:
    if not LIVE_LOG_CSV.exists():
        return []

    with LOG_LOCK:
        with LIVE_LOG_CSV.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))

    return rows[-limit:][::-1]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global DETECTOR

    LIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    DETECTOR = FinalWebsiteDetector()
    yield
    DETECTOR = None


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "Frozen multimodal website phishing detector using LightGBM, "
        "ResNet18, GraphSAGE, fixed 80/10/10 fusion, operational conflict "
        "handling and persistent live case-study logs."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": APP_TITLE,
        "version": APP_VERSION,
        "status": "running",
        "endpoint": "POST /api/website/analyze",
        "recent_logs_endpoint": "GET /api/live-logs",
        "documentation": "/docs",
    }


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        detector_loaded=(DETECTOR is not None),
        version=APP_VERSION,
        live_log_csv=str(LIVE_LOG_CSV),
    )


@app.get("/api/live-logs")
def live_logs(
    limit: int = Query(default=20, ge=1, le=200),
) -> dict[str, Any]:
    rows = read_recent_live_logs(limit)
    return {
        "count": len(rows),
        "log_file": str(LIVE_LOG_CSV),
        "rows": rows,
    }


@app.post("/api/website/analyze")
def analyze_website(request: AnalyzeWebsiteRequest) -> dict[str, Any]:
    try:
        url = normalize_candidate_url(request.url)
        validate_public_destination(url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    detector = get_detector()

    try:
        with ANALYSIS_LOCK:
            result = detector.analyze(url)
            result = apply_operational_decision(result)
            append_live_case_study_log(result)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "A required frozen model or generated artifact was unavailable: "
                f"{exc}"
            ),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Website analysis failed: {type(exc).__name__}: {exc}",
        ) from exc

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the J26-IT-424 website phishing detection API."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. Keep 127.0.0.1 for local development.",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help=(
            "Enable Uvicorn reload for development. "
            "The detector reloads when source files change."
        ),
    )
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "src.website_detection.api.website_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
