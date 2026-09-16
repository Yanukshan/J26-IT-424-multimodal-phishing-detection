from collections import Counter
from typing import Any
from urllib.parse import urlparse


# ============================================================
# DOMAIN HELPERS
# ============================================================

def _hostname(url: str) -> str:
    """
    Extract a normalized hostname.
    """

    try:
        host = urlparse(url).hostname or ""
        host = host.lower()

        if host.startswith("www."):
            host = host[4:]

        return host

    except Exception:
        return ""


# ============================================================
# SAFE VALUE HELPERS
# ============================================================

def _safe_int(value: Any) -> int:
    """
    Safely convert telemetry values into integers.
    """

    try:
        return int(value or 0)

    except (TypeError, ValueError):
        return 0


def _safe_bool(value: Any) -> bool:
    """
    Safely convert common values into booleans.
    """

    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, (int, float)):
        return bool(value)

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


# ============================================================
# NORMAL NETWORK FEATURES
# ============================================================

def extract_network_features(
    network_requests: list[dict],
    page_url: str,
) -> dict:
    """
    Extract runtime request-level behavioral features.

    These features describe network behavior only.
    They are NOT phishing classifications.
    """

    page_host = _hostname(page_url)

    method_counter = Counter()
    resource_counter = Counter()

    external_domains: set[str] = set()
    unique_domains: set[str] = set()

    external_request_count = 0
    insecure_request_count = 0

    for request in network_requests:

        request_url = str(
            request.get(
                "url",
                "",
            )
        )

        method = str(
            request.get(
                "method",
                "",
            )
        ).upper()

        resource_type = str(
            request.get(
                "resource_type",
                "",
            )
        ).lower()

        # ----------------------------------------------------
        # HTTP methods
        # ----------------------------------------------------

        if method:
            method_counter[
                method
            ] += 1

        # ----------------------------------------------------
        # Resource types
        # ----------------------------------------------------

        if resource_type:
            resource_counter[
                resource_type
            ] += 1

        # ----------------------------------------------------
        # Domain information
        # ----------------------------------------------------

        request_host = _hostname(
            request_url
        )

        if request_host:
            unique_domains.add(
                request_host
            )

        # ----------------------------------------------------
        # External request
        # ----------------------------------------------------

        if (
            request_host
            and page_host
            and request_host != page_host
        ):
            external_request_count += 1

            external_domains.add(
                request_host
            )

        # ----------------------------------------------------
        # Insecure request
        # ----------------------------------------------------

        if request_url.lower().startswith(
            "http://"
        ):
            insecure_request_count += 1

    total_requests = len(
        network_requests
    )

    external_request_ratio = (
        external_request_count / total_requests
        if total_requests > 0
        else 0.0
    )

    return {
        "total_requests":
            total_requests,

        "get_requests":
            method_counter.get(
                "GET",
                0,
            ),

        "post_requests":
            method_counter.get(
                "POST",
                0,
            ),

        "put_requests":
            method_counter.get(
                "PUT",
                0,
            ),

        "delete_requests":
            method_counter.get(
                "DELETE",
                0,
            ),

        "document_requests":
            resource_counter.get(
                "document",
                0,
            ),

        "script_requests":
            resource_counter.get(
                "script",
                0,
            ),

        "stylesheet_requests":
            resource_counter.get(
                "stylesheet",
                0,
            ),

        "image_requests":
            resource_counter.get(
                "image",
                0,
            ),

        "xhr_requests":
            resource_counter.get(
                "xhr",
                0,
            ),

        "fetch_requests":
            resource_counter.get(
                "fetch",
                0,
            ),

        "font_requests":
            resource_counter.get(
                "font",
                0,
            ),

        "media_requests":
            resource_counter.get(
                "media",
                0,
            ),

        "external_request_count":
            external_request_count,

        "external_request_ratio":
            round(
                external_request_ratio,
                4,
            ),

        "external_domain_count":
            len(
                external_domains
            ),

        "external_domains":
            sorted(
                external_domains
            ),

        "unique_domain_count":
            len(
                unique_domains
            ),

        "insecure_http_requests":
            insecure_request_count,
    }


# ============================================================
# NEW BEHAVIORAL TELEMETRY FEATURES
# ============================================================

def extract_behavioral_features(
    behavioral_telemetry: dict[str, Any] | None,
    failed_requests: list[dict] | None,
) -> dict[str, Any]:
    """
    Normalize the additional runtime telemetry captured by
    crawler.py.

    This includes:

        response status behavior
        failed requests
        redirect chain information
        late/client-side navigation

    The returned information is suitable for flattening into
    the research CSV.

    IMPORTANT:
    These values remain model/research features.
    They do not independently classify phishing.
    """

    telemetry = (
        behavioral_telemetry
        if isinstance(
            behavioral_telemetry,
            dict,
        )
        else {}
    )

    failed_request_list = (
        failed_requests
        if isinstance(
            failed_requests,
            list,
        )
        else []
    )

    # --------------------------------------------------------
    # Redirect chain
    # --------------------------------------------------------

    redirect_chain = telemetry.get(
        "server_redirect_chain",
        [],
    )

    if not isinstance(
        redirect_chain,
        list,
    ):
        redirect_chain = []

    redirect_chain = [
        str(url)
        for url in redirect_chain
        if url
    ]

    # --------------------------------------------------------
    # Failed request domains
    # --------------------------------------------------------

    failed_request_domains: set[str] = set()

    for failed_request in failed_request_list:

        if not isinstance(
            failed_request,
            dict,
        ):
            continue

        failed_url = str(
            failed_request.get(
                "url",
                "",
            )
        )

        domain = _hostname(
            failed_url
        )

        if domain:
            failed_request_domains.add(
                domain
            )

    return {
        # ----------------------------------------------------
        # Response telemetry
        # ----------------------------------------------------

        "response_count":
            _safe_int(
                telemetry.get(
                    "response_count"
                )
            ),

        "main_document_response_count":
            _safe_int(
                telemetry.get(
                    "main_document_response_count"
                )
            ),

        "response_3xx_count":
            _safe_int(
                telemetry.get(
                    "response_3xx_count"
                )
            ),

        "response_4xx_count":
            _safe_int(
                telemetry.get(
                    "response_4xx_count"
                )
            ),

        "response_5xx_count":
            _safe_int(
                telemetry.get(
                    "response_5xx_count"
                )
            ),

        # ----------------------------------------------------
        # Failed requests
        # ----------------------------------------------------

        "failed_request_count":
            _safe_int(
                telemetry.get(
                    "failed_request_count"
                )
            ),

        "failed_document_request_count":
            _safe_int(
                telemetry.get(
                    "failed_document_request_count"
                )
            ),

        # Number of detailed failed requests stored by crawler.
        # This may be lower than failed_request_count because
        # crawler.py intentionally limits saved details.
        "failed_request_detail_count":
            len(
                failed_request_list
            ),

        "failed_request_domain_count":
            len(
                failed_request_domains
            ),

        # ----------------------------------------------------
        # Redirect behavior
        # ----------------------------------------------------

        "server_redirect_hops":
            _safe_int(
                telemetry.get(
                    "server_redirect_hops"
                )
            ),

        "server_redirect_chain":
            redirect_chain,

        # ----------------------------------------------------
        # Navigation behavior
        # ----------------------------------------------------

        "initial_response_url":
            str(
                telemetry.get(
                    "initial_response_url",
                    "",
                )
                or ""
            ),

        "late_navigation_detected":
            _safe_bool(
                telemetry.get(
                    "late_navigation_detected",
                    False,
                )
            ),
    }