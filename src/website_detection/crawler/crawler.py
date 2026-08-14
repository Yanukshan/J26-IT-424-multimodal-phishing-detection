import json
import sys
from typing import Any

from playwright.sync_api import (
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


DEFAULT_TIMEOUT_MS = 30_000


def normalize_url(
    url: str,
) -> str:
    """
    Normalize command-line URL input.
    """

    url = url.strip()

    if not url.startswith(
        (
            "http://",
            "https://",
        )
    ):
        return (
            f"https://{url}"
        )

    return url


def determine_crawl_status(
    status_code: int | None,
) -> str:
    """
    Crawl status is operational metadata,
    not a phishing classification.
    """

    if status_code is None:
        return "NO_RESPONSE"

    if status_code == 403:
        return "BLOCKED"

    if 200 <= status_code < 400:
        return "SUCCESS"

    return "HTTP_ERROR"


def crawl_website(
    url: str,
) -> dict[str, Any]:
    """
    Render one website and collect:

    1. Navigation metadata
    2. DOM structural features
    3. Network behavioral features
    4. DOM/resource dependency graph features
    5. Raw network requests

    The crawler does not submit forms or
    provide credentials.
    """

    requested_url = normalize_url(
        url
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

        # -----------------------------------------------------
        # Network listener
        # -----------------------------------------------------

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

            # -------------------------------------------------
            # Navigation
            # -------------------------------------------------

            response = page.goto(
                requested_url,
                wait_until="load",
                timeout=DEFAULT_TIMEOUT_MS,
            )

            # -------------------------------------------------
            # Rendered information
            # -------------------------------------------------

            html = (
                page.content()
            )

            title = (
                page.title()
            )

            final_url = (
                page.url
            )

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

            # -------------------------------------------------
            # DOM features
            # -------------------------------------------------

            dom_features = (
                extract_dom_features(
                    html=html,
                    page_url=final_url,
                )
            )

            # -------------------------------------------------
            # Network features
            # -------------------------------------------------

            network_features = (
                extract_network_features(
                    network_requests=
                        network_requests,

                    page_url=
                        final_url,
                )
            )

            # -------------------------------------------------
            # Resource dependency graph
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Redirect detection
            # -------------------------------------------------

            redirected = (
                final_url.rstrip("/")
                != requested_url.rstrip("/")
            )

            # -------------------------------------------------
            # Final response
            # -------------------------------------------------

            return {
                "crawl_status":
                    crawl_status,

                "requested_url":
                    requested_url,

                "final_url":
                    final_url,

                "redirected":
                    redirected,

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

        except PlaywrightTimeoutError:

            return {
                "crawl_status":
                    "TIMEOUT",

                "requested_url":
                    requested_url,

                "error":
                    (
                        "Page did not finish "
                        "loading within "
                        f"{DEFAULT_TIMEOUT_MS} ms."
                    ),
            }

        except Exception as exc:

            return {
                "crawl_status":
                    "ERROR",

                "requested_url":
                    requested_url,

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


def main() -> None:
    """
    Command-line entry point.
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