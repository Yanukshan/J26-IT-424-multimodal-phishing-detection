import json
import sys
from typing import Any
from urllib.parse import urlparse

from playwright.sync_api import (
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

from src.website_detection.behavioral_analysis.network_features import (
    extract_network_features,
)

from src.website_detection.dom_analysis.dom_features import (
    extract_dom_features,
)

from src.website_detection.graph_builder.resource_graph import (
    build_resource_graph,
    extract_graph_features,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TIMEOUT_MS = 30_000

STABLE_PAGE_RETRIES = 3
STABLE_PAGE_WAIT_MS = 5_000
STABLE_PAGE_RETRY_DELAY_MS = 500


# ============================================================
# URL NORMALIZATION
# ============================================================

def normalize_url(
    url: str,
) -> str:
    """
    Normalize a URL supplied to the crawler.

    If no HTTP/HTTPS scheme exists, HTTPS is used.
    """

    url = url.strip()

    if not url.lower().startswith(
        (
            "http://",
            "https://",
        )
    ):
        return f"https://{url}"

    return url


# ============================================================
# DOMAIN EXTRACTION
# ============================================================

def extract_domain(
    url: str,
) -> str:
    """
    Extract and normalize the hostname from a URL.

    Removes the common leading 'www.' prefix.
    """

    if not url:
        return ""

    try:
        hostname = (
            urlparse(url).hostname
            or ""
        ).lower()

        if hostname.startswith("www."):
            hostname = hostname[4:]

        return hostname

    except Exception:
        return ""


# ============================================================
# REDIRECT / DOMAIN INTEGRITY
# ============================================================

def build_redirect_metadata(
    requested_url: str,
    final_url: str,
) -> dict[str, Any]:
    """
    Compare the originally requested URL with the final URL.

    Cross-domain redirects are marked for later label review.

    This function never changes the original dataset label.
    """

    requested_domain = extract_domain(
        requested_url
    )

    final_domain = extract_domain(
        final_url
    )

    # --------------------------------------------------------
    # No final page was reached
    # --------------------------------------------------------

    if not final_url:

        return {
            "requested_domain":
                requested_domain,

            "final_domain":
                "",

            "redirected":
                False,

            "cross_domain_redirect":
                False,

            "label_review_required":
                False,
        }

    # --------------------------------------------------------
    # URL-level redirect
    # --------------------------------------------------------

    redirected = (
        final_url.rstrip("/")
        != requested_url.rstrip("/")
    )

    # --------------------------------------------------------
    # Domain-level redirect
    # --------------------------------------------------------

    cross_domain_redirect = bool(
        requested_domain
        and final_domain
        and requested_domain
        != final_domain
    )

    # --------------------------------------------------------
    # Historical label may no longer describe current content
    # --------------------------------------------------------

    label_review_required = (
        cross_domain_redirect
    )

    return {
        "requested_domain":
            requested_domain,

        "final_domain":
            final_domain,

        "redirected":
            redirected,

        "cross_domain_redirect":
            cross_domain_redirect,

        "label_review_required":
            label_review_required,
    }


# ============================================================
# HTTP STATUS CLASSIFICATION
# ============================================================

def determine_crawl_status(
    status_code: int | None,
) -> str:
    """
    Convert the main HTTP response into an operational
    collection status.

    These statuses describe crawling only.
    They are not phishing classifications.
    """

    if status_code is None:
        return "NO_RESPONSE"

    if status_code == 403:
        return "BLOCKED"

    if 200 <= status_code < 400:
        return "SUCCESS"

    return "HTTP_ERROR"


# ============================================================
# NAVIGATION ERROR CLASSIFICATION
# ============================================================

def classify_navigation_error(
    error_message: str,
) -> str:
    """
    Convert common Chromium / Playwright navigation errors
    into research-friendly collection statuses.
    """

    message = error_message.upper()

    # --------------------------------------------------------
    # Page continuously navigating while being captured
    # --------------------------------------------------------

    unstable_navigation_indicators = (
        "PAGE IS NAVIGATING AND CHANGING THE CONTENT",
        "EXECUTION CONTEXT WAS DESTROYED",
    )

    if any(
        indicator in message
        for indicator
        in unstable_navigation_indicators
    ):
        return "NAVIGATION_UNSTABLE"

    # --------------------------------------------------------
    # DNS
    # --------------------------------------------------------

    if (
        "ERR_NAME_NOT_RESOLVED"
        in message
        or
        "ERR_NAME_RESOLUTION_FAILED"
        in message
    ):
        return "DNS_ERROR"

    # --------------------------------------------------------
    # Connection timeout
    # --------------------------------------------------------

    if (
        "ERR_CONNECTION_TIMED_OUT"
        in message
    ):
        return "CONNECTION_TIMEOUT"

    # --------------------------------------------------------
    # Connection refused
    # --------------------------------------------------------

    if (
        "ERR_CONNECTION_REFUSED"
        in message
    ):
        return "CONNECTION_REFUSED"

    # --------------------------------------------------------
    # Connection reset / closed
    # --------------------------------------------------------

    if (
        "ERR_CONNECTION_RESET"
        in message
        or
        "ERR_CONNECTION_CLOSED"
        in message
    ):
        return "CONNECTION_RESET"

    # --------------------------------------------------------
    # SSL / certificate
    # --------------------------------------------------------

    ssl_indicators = (
        "ERR_CERT_",
        "ERR_SSL_",
        "CERTIFICATE",
        "SSL_ERROR",
    )

    if any(
        indicator in message
        for indicator
        in ssl_indicators
    ):
        return "SSL_ERROR"

    # --------------------------------------------------------
    # Internet disconnected
    # --------------------------------------------------------

    if (
        "ERR_INTERNET_DISCONNECTED"
        in message
    ):
        return "NETWORK_OFFLINE"

    # --------------------------------------------------------
    # Network unreachable
    # --------------------------------------------------------

    if (
        "ERR_ADDRESS_UNREACHABLE"
        in message
        or
        "ERR_NETWORK_UNREACHABLE"
        in message
    ):
        return "NETWORK_UNREACHABLE"

    # --------------------------------------------------------
    # Redirect failure
    # --------------------------------------------------------

    if (
        "ERR_TOO_MANY_REDIRECTS"
        in message
    ):
        return "REDIRECT_ERROR"

    # --------------------------------------------------------
    # Other network problems
    # --------------------------------------------------------

    network_indicators = (
        "ERR_NETWORK_CHANGED",
        "ERR_NETWORK_ACCESS_DENIED",
        "ERR_PROXY_CONNECTION_FAILED",
        "ERR_TUNNEL_CONNECTION_FAILED",
    )

    if any(
        indicator in message
        for indicator
        in network_indicators
    ):
        return "NETWORK_ERROR"

    return "ERROR"


# ============================================================
# UNSTABLE PAGE DETECTION
# ============================================================

def is_navigation_change_error(
    error_message: str,
) -> bool:
    """
    Determine whether a Playwright exception occurred because
    the page changed navigation while its content was being
    captured.
    """

    message = error_message.upper()

    indicators = (
        "PAGE IS NAVIGATING AND CHANGING THE CONTENT",
        "EXECUTION CONTEXT WAS DESTROYED",
    )

    return any(
        indicator in message
        for indicator in indicators
    )


# ============================================================
# STABLE PAGE SNAPSHOT
# ============================================================

def get_stable_page_snapshot(
    page,
    retries: int = STABLE_PAGE_RETRIES,
) -> dict[str, str]:
    """
    Capture rendered HTML, title and final URL from a page.

    Some websites trigger another navigation immediately after
    the first page load. If that happens while page.content()
    is running, retry after waiting for the new document.

    A persistent navigation problem is re-raised and later
    classified as NAVIGATION_UNSTABLE.
    """

    if retries <= 0:
        raise ValueError(
            "retries must be greater than zero"
        )

    last_error: PlaywrightError | None = None

    for attempt in range(
        1,
        retries + 1,
    ):

        try:

            # ------------------------------------------------
            # Wait for current document when possible
            # ------------------------------------------------

            try:
                page.wait_for_load_state(
                    "domcontentloaded",
                    timeout=STABLE_PAGE_WAIT_MS,
                )

            except PlaywrightTimeoutError:
                # Continue to content capture.
                # The page may still already expose a usable DOM.
                pass

            # ------------------------------------------------
            # Capture the current rendered document
            # ------------------------------------------------

            html = page.content()

            # ------------------------------------------------
            # Capture associated metadata immediately
            # ------------------------------------------------

            title = page.title()

            final_url = page.url

            return {
                "html":
                    html,

                "title":
                    title,

                "final_url":
                    final_url,
            }

        except PlaywrightError as exc:

            last_error = exc

            # ------------------------------------------------
            # Do not retry unrelated Playwright failures
            # ------------------------------------------------

            if not is_navigation_change_error(
                str(exc)
            ):
                raise

            # ------------------------------------------------
            # No retries remain
            # ------------------------------------------------

            if attempt >= retries:
                raise

            # ------------------------------------------------
            # Give the new navigation a short time to settle
            # ------------------------------------------------

            try:
                page.wait_for_load_state(
                    "domcontentloaded",
                    timeout=STABLE_PAGE_WAIT_MS,
                )

            except PlaywrightTimeoutError:
                pass

            page.wait_for_timeout(
                STABLE_PAGE_RETRY_DELAY_MS
            )

    # Defensive fallback.
    # Normally unreachable because the final retry re-raises.
    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "Unable to capture a stable page snapshot."
    )


# ============================================================
# WEBSITE CRAWLER
# ============================================================

def crawl_website(
    url: str,
) -> dict[str, Any]:
    """
    Dynamically render and analyse a website.

    Collects:

        - Navigation metadata
        - Redirect/domain metadata
        - DOM structural features
        - Runtime network behavioral features
        - Resource dependency graph features
        - Raw network request information

    The crawler does not submit forms or enter credentials.
    """

    requested_url = normalize_url(
        url
    )

    # --------------------------------------------------------
    # Metadata used when navigation never completes
    # --------------------------------------------------------

    failure_redirect_metadata = (
        build_redirect_metadata(
            requested_url=
                requested_url,

            final_url="",
        )
    )

    network_requests: list[
        dict[str, str]
    ] = []

    with sync_playwright() as playwright:

        browser = (
            playwright.chromium.launch(
                headless=True,
            )
        )

        context = (
            browser.new_context()
        )

        page = (
            context.new_page()
        )

        # ====================================================
        # NETWORK REQUEST CAPTURE
        # ====================================================

        def capture_request(
            request,
        ) -> None:

            network_requests.append(
                {
                    "url":
                        request.url,

                    "method":
                        request.method,

                    "resource_type":
                        request.resource_type,
                }
            )

        page.on(
            "request",
            capture_request,
        )

        try:

            # =================================================
            # INITIAL NAVIGATION
            # =================================================

            response = page.goto(
                requested_url,
                wait_until="load",
                timeout=DEFAULT_TIMEOUT_MS,
            )

            # =================================================
            # STABLE RENDERED PAGE SNAPSHOT
            # =================================================

            snapshot = (
                get_stable_page_snapshot(
                    page
                )
            )

            html = snapshot[
                "html"
            ]

            title = snapshot[
                "title"
            ]

            final_url = snapshot[
                "final_url"
            ]

            # =================================================
            # HTTP RESPONSE
            # =================================================

            status_code = (
                response.status
                if response
                else None
            )

            crawl_status = (
                determine_crawl_status(
                    status_code
                )
            )

            # =================================================
            # REDIRECT / DOMAIN METADATA
            # =================================================

            redirect_metadata = (
                build_redirect_metadata(
                    requested_url=
                        requested_url,

                    final_url=
                        final_url,
                )
            )

            # =================================================
            # DOM FEATURES
            # =================================================

            dom_features = (
                extract_dom_features(
                    html=html,
                    page_url=final_url,
                )
            )

            # =================================================
            # NETWORK FEATURES
            # =================================================

            network_features = (
                extract_network_features(
                    network_requests=
                        network_requests,

                    page_url=
                        final_url,
                )
            )

            # =================================================
            # RESOURCE DEPENDENCY GRAPH
            # =================================================

            resource_graph = (
                build_resource_graph(
                    html=html,
                    page_url=final_url,
                    network_requests=
                        network_requests,
                )
            )

            graph_features = (
                extract_graph_features(
                    resource_graph
                )
            )

            # =================================================
            # FINAL RESULT
            # =================================================

            return {
                "crawl_status":
                    crawl_status,

                "requested_url":
                    requested_url,

                "requested_domain":
                    redirect_metadata[
                        "requested_domain"
                    ],

                "final_url":
                    final_url,

                "final_domain":
                    redirect_metadata[
                        "final_domain"
                    ],

                "redirected":
                    redirect_metadata[
                        "redirected"
                    ],

                "cross_domain_redirect":
                    redirect_metadata[
                        "cross_domain_redirect"
                    ],

                "label_review_required":
                    redirect_metadata[
                        "label_review_required"
                    ],

                "status_code":
                    status_code,

                "title":
                    title,

                "html_length":
                    len(html),

                "dom_features":
                    dom_features,

                "network_features":
                    network_features,

                "graph_features":
                    graph_features,

                "network_request_count":
                    len(
                        network_requests
                    ),

                "network_requests":
                    network_requests,
            }

        # ====================================================
        # PLAYWRIGHT TIMEOUT
        # ====================================================

        except PlaywrightTimeoutError as exc:

            return {
                "crawl_status":
                    "TIMEOUT",

                "requested_url":
                    requested_url,

                **failure_redirect_metadata,

                "error_type":
                    type(
                        exc
                    ).__name__,

                "error":
                    str(
                        exc
                    ),
            }

        # ====================================================
        # PLAYWRIGHT NAVIGATION / BROWSER ERROR
        # ====================================================

        except PlaywrightError as exc:

            error_message = str(
                exc
            )

            classified_status = (
                classify_navigation_error(
                    error_message
                )
            )

            return {
                "crawl_status":
                    classified_status,

                "requested_url":
                    requested_url,

                **failure_redirect_metadata,

                "error_type":
                    type(
                        exc
                    ).__name__,

                "error":
                    error_message,
            }

        # ====================================================
        # UNEXPECTED PYTHON ERROR
        # ====================================================

        except Exception as exc:

            return {
                "crawl_status":
                    "ERROR",

                "requested_url":
                    requested_url,

                **failure_redirect_metadata,

                "error_type":
                    type(
                        exc
                    ).__name__,

                "error":
                    str(
                        exc
                    ),
            }

        finally:

            context.close()

            browser.close()


# ============================================================
# COMMAND LINE
# ============================================================

def main() -> None:
    """
    Command-line entry point.

    Example:

        python -m \
        src.website_detection.crawler.crawler \
        https://www.virtualbox.org/
    """

    if len(
        sys.argv
    ) != 2:

        print(
            "Usage:\n"
            "python -m "
            "src.website_detection."
            "crawler.crawler <url>"
        )

        sys.exit(1)

    result = crawl_website(
        sys.argv[1]
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()