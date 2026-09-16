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

from src.website_detection.credential_analysis.credential_features import (
    extract_credential_features,
)

from src.website_detection.graph_builder.resource_graph import (
    build_resource_graph,
    extract_graph_features,
)

from src.website_detection.graph_builder.graph_serializer import (
    save_resource_graph,
)

from src.website_detection.url_analysis.url_features import (
    extract_url_features,
)

from src.website_detection.visual_analysis.screenshot_capture import (
    capture_page_screenshot,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TIMEOUT_MS = 30_000

STABLE_PAGE_RETRIES = 3
STABLE_PAGE_WAIT_MS = 5_000
STABLE_PAGE_RETRY_DELAY_MS = 500

# Avoid keeping an unlimited number of failed-request details.
MAX_FAILED_REQUEST_DETAILS = 50


# ============================================================
# URL NORMALIZATION
# ============================================================

def normalize_url(url: str) -> str:
    """
    Normalize a URL supplied to the crawler.

    If there is no HTTP/HTTPS scheme, HTTPS is used.
    """

    url = str(url).strip()

    if not url.lower().startswith(
        (
            "http://",
            "https://",
        )
    ):
        return f"https://{url}"

    return url


# ============================================================
# URL COMPARISON
# ============================================================

def urls_equivalent(
    first_url: str,
    second_url: str,
) -> bool:
    """
    Lightweight comparison used for navigation metadata.

    A trailing slash difference is ignored.
    """

    if not first_url or not second_url:
        return False

    return (
        str(first_url).rstrip("/")
        ==
        str(second_url).rstrip("/")
    )


# ============================================================
# BROWSER INTERNAL ERROR PAGE DETECTION
# ============================================================

def is_browser_internal_error_url(
    url: str,
) -> bool:
    """
    Detect Chromium/browser-generated internal error pages.

    Example:

        chrome-error://chromewebdata/

    These are not real destination websites.
    """

    if not url:
        return False

    normalized = str(
        url
    ).strip().lower()

    browser_error_prefixes = (
        "chrome-error://",
        "chrome://",
        "edge-error://",
        "about:neterror",
    )

    if normalized.startswith(
        browser_error_prefixes
    ):
        return True

    try:

        parsed = urlparse(
            normalized
        )

        hostname = (
            parsed.hostname
            or ""
        ).lower()

        if hostname == "chromewebdata":
            return True

    except Exception:
        pass

    return False


# ============================================================
# DOMAIN EXTRACTION
# ============================================================

def extract_domain(
    url: str,
) -> str:
    """
    Extract normalized hostname.

    Browser-generated internal pages intentionally return
    an empty domain.
    """

    if not url:
        return ""

    if is_browser_internal_error_url(
        url
    ):
        return ""

    try:

        hostname = (
            urlparse(
                url
            ).hostname
            or ""
        ).lower()

        if hostname.startswith(
            "www."
        ):
            hostname = hostname[
                4:
            ]

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
    Compare the originally requested website with the final
    rendered website.

    Cross-domain changes are marked for dataset-label review.

    Browser internal error pages are not redirects.
    """

    requested_domain = extract_domain(
        requested_url
    )

    if (
        not final_url
        or is_browser_internal_error_url(
            final_url
        )
    ):

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

    final_domain = extract_domain(
        final_url
    )

    redirected = not urls_equivalent(
        requested_url,
        final_url,
    )

    cross_domain_redirect = bool(
        requested_domain
        and final_domain
        and requested_domain
        != final_domain
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
            cross_domain_redirect,
    }


# ============================================================
# HTTP STATUS CLASSIFICATION
# ============================================================

def determine_crawl_status(
    status_code: int | None,
) -> str:
    """
    Convert final document HTTP status into an operational
    collection status.

    This is not a phishing classification.
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
    Convert common browser/network failures into cleaner
    research collection statuses.
    """

    message = str(
        error_message
    ).upper()

    # --------------------------------------------------------
    # Browser-generated error page
    # --------------------------------------------------------

    if (
        "CHROME-ERROR://"
        in message
        or
        "CHROMEWEBDATA"
        in message
    ):
        return "BROWSER_ERROR_PAGE"

    # --------------------------------------------------------
    # Continuously changing navigation
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
    # Connection reset
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
    # Redirect loop
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
    Check whether the rendered document changed while
    page.content() was being captured.
    """

    message = str(
        error_message
    ).upper()

    indicators = (
        "PAGE IS NAVIGATING AND CHANGING THE CONTENT",
        "EXECUTION CONTEXT WAS DESTROYED",
    )

    return any(
        indicator in message
        for indicator
        in indicators
    )


# ============================================================
# STABLE PAGE SNAPSHOT
# ============================================================

def get_stable_page_snapshot(
    page,
    retries: int = STABLE_PAGE_RETRIES,
) -> dict[str, str]:
    """
    Retrieve stable HTML, title and final URL.

    Some websites navigate again immediately after the first
    document load. Retry only that specific condition.
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

            try:

                page.wait_for_load_state(
                    "domcontentloaded",
                    timeout=
                        STABLE_PAGE_WAIT_MS,
                )

            except PlaywrightTimeoutError:
                pass

            html = page.content()

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

            if not is_navigation_change_error(
                str(
                    exc
                )
            ):
                raise

            if attempt >= retries:
                raise

            try:

                page.wait_for_load_state(
                    "domcontentloaded",
                    timeout=
                        STABLE_PAGE_WAIT_MS,
                )

            except PlaywrightTimeoutError:
                pass

            page.wait_for_timeout(
                STABLE_PAGE_RETRY_DELAY_MS
            )

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "Unable to capture stable page."
    )


# ============================================================
# SERVER REDIRECT CHAIN
# ============================================================

def extract_server_redirect_chain(
    response,
) -> list[str]:
    """
    Reconstruct server-side HTTP redirect history using
    Playwright's redirected_from chain.

    Example:

        http://example.com
            ↓
        https://example.com
            ↓
        https://www.example.com/
    """

    if response is None:
        return []

    try:
        request = response.request

    except Exception:
        return []

    reversed_chain: list[str] = []

    visited: set[int] = set()

    while request is not None:

        request_identity = id(
            request
        )

        if request_identity in visited:
            break

        visited.add(
            request_identity
        )

        try:

            reversed_chain.append(
                request.url
            )

            request = (
                request.redirected_from
            )

        except Exception:
            break

    reversed_chain.reverse()

    return reversed_chain


# ============================================================
# FAILED REQUEST ERROR TEXT
# ============================================================

def extract_request_failure_text(
    request,
) -> str:
    """
    Safely retrieve Playwright request failure information
    across compatible Playwright representations.
    """

    try:

        failure = request.failure

        if failure is None:
            return ""

        if isinstance(
            failure,
            str,
        ):
            return failure

        error_text = getattr(
            failure,
            "error_text",
            None,
        )

        if error_text:
            return str(
                error_text
            )

        return str(
            failure
        )

    except Exception:
        return ""


# ============================================================
# BEHAVIORAL TELEMETRY BUILDER
# ============================================================

def build_behavioral_telemetry(
    runtime_state: dict[str, Any],
    initial_response,
    final_document_response,
    requested_url: str,
    final_url: str,
) -> dict[str, Any]:
    """
    Produce navigation/network behavioral metadata.

    These are observable runtime signals, not phishing labels.
    """

    redirect_chain = (
        extract_server_redirect_chain(
            final_document_response
        )
    )

    server_redirect_hops = max(
        0,
        len(
            redirect_chain
        ) - 1,
    )

    initial_response_url = ""

    if initial_response is not None:

        try:
            initial_response_url = (
                initial_response.url
            )

        except Exception:
            initial_response_url = ""

    # --------------------------------------------------------
    # Detect navigation occurring after page.goto() returned
    #
    # This may represent JavaScript/meta-refresh navigation.
    # --------------------------------------------------------

    late_navigation_detected = bool(
        initial_response_url
        and final_url
        and not urls_equivalent(
            initial_response_url,
            final_url,
        )
    )

    return {
        "response_count":
            runtime_state.get(
                "response_count",
                0,
            ),

        "main_document_response_count":
            runtime_state.get(
                "main_document_response_count",
                0,
            ),

        "response_3xx_count":
            runtime_state.get(
                "response_3xx_count",
                0,
            ),

        "response_4xx_count":
            runtime_state.get(
                "response_4xx_count",
                0,
            ),

        "response_5xx_count":
            runtime_state.get(
                "response_5xx_count",
                0,
            ),

        "failed_request_count":
            runtime_state.get(
                "failed_request_count",
                0,
            ),

        "failed_document_request_count":
            runtime_state.get(
                "failed_document_request_count",
                0,
            ),

        "server_redirect_hops":
            server_redirect_hops,

        "server_redirect_chain":
            redirect_chain,

        "initial_response_url":
            initial_response_url,

        "late_navigation_detected":
            late_navigation_detected,

        "requested_url":
            requested_url,

        "observed_final_url":
            final_url,
    }


# ============================================================
# BROWSER ERROR RESULT
# ============================================================

def build_browser_error_result(
    requested_url: str,
    final_url: str,
    title: str,
    html: str,
    status_code: int | None,
    network_requests: list[dict[str, str]],
    failed_requests: list[dict[str, str]],
    behavioral_telemetry: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a safe record for Chromium-generated error pages.

    Browser error HTML is not passed into our DOM/graph
    feature extractors because it represents Chromium itself,
    not the requested website.
    """

    redirect_metadata = (
        build_redirect_metadata(
            requested_url=
                requested_url,

            final_url=
                final_url,
        )
    )

    return {
        "crawl_status":
            "BROWSER_ERROR_PAGE",

        "requested_url":
            requested_url,

        "requested_domain":
            redirect_metadata[
                "requested_domain"
            ],

        "final_url":
            final_url,

        "final_domain":
            "",

        "redirected":
            False,

        "cross_domain_redirect":
            False,

        "label_review_required":
            False,

        "status_code":
            status_code,

        "title":
            title,

        "html_length":
            len(
                html
            ),

        # --------------------------------------------
        # URL / DOMAIN FEATURES
        #
        # Chromium's internal error URL must not be
        # treated as the website's real final URL.
        # --------------------------------------------

        "url_features":
            extract_url_features(
                requested_url=
                    requested_url,

                final_url=
                    None,
            ),

        "behavioral_telemetry":
            behavioral_telemetry,

        "network_request_count":
            len(
                network_requests
            ),

        "network_requests":
            network_requests,

        "failed_requests":
            failed_requests,

        "error_type":
            "BrowserErrorPage",

        "error":
            (
                "Chromium rendered an internal "
                f"error page: {final_url}"
            ),
    }


# ============================================================
# WEBSITE CRAWLER
# ============================================================

def crawl_website(
    url: str,
    label: int | None = None,
) -> dict[str, Any]:
    """
    Dynamically render and analyse one website.

    Collects:

        - navigation metadata
        - redirect integrity metadata
        - URL/domain lexical and structural features
        - DOM structural features
        - credential-intent and sensitive-form features
        - rendered visual screenshot evidence
        - runtime network features
        - resource dependency graph features
        - full serialized graph artifact for GNN training
        - failed network requests
        - response status telemetry
        - server redirect chain
        - late/client-side navigation information

    It does not submit forms or enter credentials.
    """

    requested_url = normalize_url(
        url
    )

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

    failed_requests: list[
        dict[str, str]
    ] = []

    runtime_state: dict[
        str,
        Any,
    ] = {
        "response_count":
            0,

        "main_document_response_count":
            0,

        "response_3xx_count":
            0,

        "response_4xx_count":
            0,

        "response_5xx_count":
            0,

        "failed_request_count":
            0,

        "failed_document_request_count":
            0,

        "last_main_document_response":
            None,
    }

    initial_response = None

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
        # REQUEST CAPTURE
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

        # ====================================================
        # RESPONSE CAPTURE
        # ====================================================

        def capture_response(
            response,
        ) -> None:

            try:

                status = int(
                    response.status
                )

            except Exception:
                status = 0

            runtime_state[
                "response_count"
            ] += 1

            if 300 <= status < 400:

                runtime_state[
                    "response_3xx_count"
                ] += 1

            elif 400 <= status < 500:

                runtime_state[
                    "response_4xx_count"
                ] += 1

            elif status >= 500:

                runtime_state[
                    "response_5xx_count"
                ] += 1

            # ------------------------------------------------
            # Track the main-frame document response
            # ------------------------------------------------

            try:

                request = (
                    response.request
                )

                is_main_document = (
                    request.resource_type
                    == "document"
                    and request.frame
                    == page.main_frame
                )

                if is_main_document:

                    runtime_state[
                        "main_document_response_count"
                    ] += 1

                    runtime_state[
                        "last_main_document_response"
                    ] = response

            except Exception:
                pass

        # ====================================================
        # FAILED REQUEST CAPTURE
        # ====================================================

        def capture_failed_request(
            request,
        ) -> None:

            runtime_state[
                "failed_request_count"
            ] += 1

            resource_type = ""

            try:
                resource_type = (
                    request.resource_type
                )

            except Exception:
                pass

            if resource_type == "document":

                runtime_state[
                    "failed_document_request_count"
                ] += 1

            if (
                len(
                    failed_requests
                )
                >= MAX_FAILED_REQUEST_DETAILS
            ):
                return

            failed_requests.append(
                {
                    "url":
                        request.url,

                    "method":
                        request.method,

                    "resource_type":
                        resource_type,

                    "failure":
                        extract_request_failure_text(
                            request
                        ),
                }
            )

        page.on(
            "request",
            capture_request,
        )

        page.on(
            "response",
            capture_response,
        )

        page.on(
            "requestfailed",
            capture_failed_request,
        )

        try:

            # =================================================
            # INITIAL NAVIGATION
            # =================================================

            initial_response = page.goto(
                requested_url,
                wait_until="load",
                timeout=
                    DEFAULT_TIMEOUT_MS,
            )

            # =================================================
            # STABLE SNAPSHOT
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
            # MOST RECENT MAIN DOCUMENT RESPONSE
            # =================================================

            final_document_response = (
                runtime_state.get(
                    "last_main_document_response"
                )
                or initial_response
            )

            status_code = None

            if final_document_response is not None:

                try:

                    status_code = int(
                        final_document_response.status
                    )

                except Exception:
                    status_code = None

            # =================================================
            # NEW BEHAVIORAL TELEMETRY
            # =================================================

            behavioral_telemetry = (
                build_behavioral_telemetry(
                    runtime_state=
                        runtime_state,

                    initial_response=
                        initial_response,

                    final_document_response=
                        final_document_response,

                    requested_url=
                        requested_url,

                    final_url=
                        final_url,
                )
            )

            # =================================================
            # CHROMIUM INTERNAL ERROR PAGE
            # =================================================

            if is_browser_internal_error_url(
                final_url
            ):

                return (
                    build_browser_error_result(
                        requested_url=
                            requested_url,

                        final_url=
                            final_url,

                        title=
                            title,

                        html=
                            html,

                        status_code=
                            status_code,

                        network_requests=
                            network_requests,

                        failed_requests=
                            failed_requests,

                        behavioral_telemetry=
                            behavioral_telemetry,
                    )
                )

            # =================================================
            # CRAWL STATUS
            # =================================================

            crawl_status = (
                determine_crawl_status(
                    status_code
                )
            )

            # =================================================
            # VISUAL SCREENSHOT
            # =================================================
            #
            # Capture only successfully rendered pages.
            #
            # The dataset collector passes the known class label
            # (0 = phishing, 1 = legitimate). Direct CLI crawls
            # may omit the label, which stores screenshots under
            # the utility's "unknown" class directory.
            # =================================================

            visual_features: dict[str, Any] = {}

            if crawl_status == "SUCCESS":

                visual_features = (
                    capture_page_screenshot(
                        page=
                            page,

                        requested_url=
                            requested_url,

                        label=
                            label,
                    )
                )

            # =================================================
            # REDIRECT METADATA
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
            # URL / DOMAIN FEATURES
            # =================================================

            url_features = (
                extract_url_features(
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
                    html=
                        html,

                    page_url=
                        final_url,
                )
            )

            # =================================================
            # CREDENTIAL INTENT FEATURES
            # =================================================
            #
            # Observational only:
            # - identifies sensitive input intent
            # - examines form destinations
            # - never fills or submits a form
            # =================================================

            credential_features = (
                extract_credential_features(
                    html=
                        html,

                    page_url=
                        final_url,
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
                    html=
                        html,

                    page_url=
                        final_url,

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
            # FULL GRAPH SERIALIZATION
            # =================================================
            #
            # Graph summaries above remain available for ML.
            # The full node/edge structure below is saved for
            # future GCN / GraphSAGE / GAT training.
            #
            # As with screenshots, only successful rendered
            # pages are saved as graph artifacts.
            # =================================================

            graph_artifact: dict[str, Any] = {}

            if crawl_status == "SUCCESS":

                graph_artifact = (
                    save_resource_graph(
                        graph=
                            resource_graph,

                        requested_url=
                            requested_url,

                        final_url=
                            final_url,

                        label=
                            label,

                        graph_features=
                            graph_features,
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
                    len(
                        html
                    ),

                "url_features":
                    url_features,

                "dom_features":
                    dom_features,

                "credential_features":
                    credential_features,

                "visual_features":
                    visual_features,

                "network_features":
                    network_features,

                "graph_features":
                    graph_features,

                "graph_artifact":
                    graph_artifact,

                # --------------------------------------------
                # NEW BEHAVIORAL INFORMATION
                # --------------------------------------------

                "behavioral_telemetry":
                    behavioral_telemetry,

                "network_request_count":
                    len(
                        network_requests
                    ),

                "network_requests":
                    network_requests,

                "failed_requests":
                    failed_requests,
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

                "url_features":
                    extract_url_features(
                        requested_url=
                            requested_url,

                        final_url=
                            None,
                    ),

                "error_type":
                    type(
                        exc
                    ).__name__,

                "error":
                    str(
                        exc
                    ),

                "failed_requests":
                    failed_requests,
            }

        # ====================================================
        # PLAYWRIGHT / BROWSER ERROR
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

                "url_features":
                    extract_url_features(
                        requested_url=
                            requested_url,

                        final_url=
                            None,
                    ),

                "error_type":
                    type(
                        exc
                    ).__name__,

                "error":
                    error_message,

                "failed_requests":
                    failed_requests,
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

                "url_features":
                    extract_url_features(
                        requested_url=
                            requested_url,

                        final_url=
                            None,
                    ),

                "error_type":
                    type(
                        exc
                    ).__name__,

                "error":
                    str(
                        exc
                    ),

                "failed_requests":
                    failed_requests,
            }

        finally:

            context.close()

            browser.close()


# ============================================================
# COMMAND LINE
# ============================================================

def main() -> None:

    if len(
        sys.argv
    ) != 2:

        print(
            "Usage:\n"
            "python -m "
            "src.website_detection."
            "crawler.crawler <url>"
        )

        sys.exit(
            1
        )

    result = crawl_website(
        sys.argv[
            1
        ]
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