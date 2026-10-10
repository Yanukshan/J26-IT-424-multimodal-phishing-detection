from collections import Counter
from urllib.parse import urlparse


def _hostname(url: str) -> str:
    """
    Extract and normalize a hostname.
    """
    try:
        host = urlparse(url).hostname or ""
        host = host.lower()

        if host.startswith("www."):
            host = host[4:]

        return host

    except Exception:
        return ""


def extract_network_features(
    network_requests: list[dict],
    page_url: str,
) -> dict:
    """
    Convert raw Playwright network requests into
    numerical behavioral features.
    """

    page_host = _hostname(
        page_url
    )

    method_counter = Counter()
    resource_counter = Counter()

    external_domains: set[str] = set()
    unique_domains: set[str] = set()

    external_request_count = 0
    insecure_request_count = 0

    # ---------------------------------------------------------
    # Process every request
    # ---------------------------------------------------------

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

        if method:
            method_counter[
                method
            ] += 1

        if resource_type:
            resource_counter[
                resource_type
            ] += 1

        request_host = _hostname(
            request_url
        )

        if request_host:
            unique_domains.add(
                request_host
            )

        # External network request
        if (
            request_host
            and page_host
            and request_host != page_host
        ):
            external_request_count += 1

            external_domains.add(
                request_host
            )

        # Insecure HTTP resource
        if request_url.lower().startswith(
            "http://"
        ):
            insecure_request_count += 1

    # ---------------------------------------------------------
    # Derived features
    # ---------------------------------------------------------

    total_requests = len(
        network_requests
    )

    external_request_ratio = (
        external_request_count
        / total_requests
        if total_requests > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # Final features
    # ---------------------------------------------------------

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