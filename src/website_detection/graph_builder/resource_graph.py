from urllib.parse import urljoin, urlparse

import networkx as nx
from bs4 import BeautifulSoup


def _hostname(url: str) -> str:
    """
    Extract and normalize hostname.
    """
    try:
        host = urlparse(url).hostname or ""
        host = host.lower()

        if host.startswith("www."):
            host = host[4:]

        return host

    except Exception:
        return ""


def _is_external(
    url: str,
    page_url: str,
) -> bool:
    """
    Determine whether a resource belongs
    to another hostname.
    """

    target_host = _hostname(
        url
    )

    page_host = _hostname(
        page_url
    )

    return bool(
        target_host
        and page_host
        and target_host != page_host
    )


def _valid_resource_url(
    url: str,
) -> bool:
    """
    Ignore non-network resources that should not
    become normal URL resource nodes.
    """

    if not url:
        return False

    lowered = url.lower().strip()

    blocked_prefixes = (
        "javascript:",
        "mailto:",
        "tel:",
        "data:",
        "#",
    )

    return not lowered.startswith(
        blocked_prefixes
    )


def build_resource_graph(
    html: str,
    page_url: str,
    network_requests: list[dict],
) -> nx.DiGraph:
    """
    Build the website DOM/resource dependency graph.

    Node types:
        page
        script
        inline_script
        stylesheet
        image
        iframe
        form
        form_target
        network_resource

    Edge types:
        contains
        loads
        submits_to
        requests
    """

    graph = nx.DiGraph()

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # ---------------------------------------------------------
    # Root page
    # ---------------------------------------------------------

    graph.add_node(
        page_url,
        node_type="page",
        url=page_url,
        hostname=_hostname(
            page_url
        ),
        external=False,
        runtime_requested=True,
    )

    # ---------------------------------------------------------
    # Scripts
    # ---------------------------------------------------------

    for index, script in enumerate(
        soup.find_all("script")
    ):

        src = script.get("src")

        if src:

            resource_url = urljoin(
                page_url,
                str(src),
            )

            if not _valid_resource_url(
                resource_url
            ):
                continue

            graph.add_node(
                resource_url,
                node_type="script",
                url=resource_url,
                hostname=_hostname(
                    resource_url
                ),
                external=_is_external(
                    resource_url,
                    page_url,
                ),
            )

            graph.add_edge(
                page_url,
                resource_url,
                edge_type="loads",
            )

        else:

            node_id = (
                f"inline_script_{index}"
            )

            graph.add_node(
                node_id,
                node_type="inline_script",
                external=False,
            )

            graph.add_edge(
                page_url,
                node_id,
                edge_type="contains",
            )

    # ---------------------------------------------------------
    # Stylesheets
    # ---------------------------------------------------------

    for link in soup.find_all(
        "link",
        href=True,
    ):

        rel = [
            str(item).lower()
            for item in link.get(
                "rel",
                [],
            )
        ]

        if "stylesheet" not in rel:
            continue

        href = str(
            link.get(
                "href",
                "",
            )
        )

        resource_url = urljoin(
            page_url,
            href,
        )

        if not _valid_resource_url(
            resource_url
        ):
            continue

        graph.add_node(
            resource_url,
            node_type="stylesheet",
            url=resource_url,
            hostname=_hostname(
                resource_url
            ),
            external=_is_external(
                resource_url,
                page_url,
            ),
        )

        graph.add_edge(
            page_url,
            resource_url,
            edge_type="loads",
        )

    # ---------------------------------------------------------
    # Images
    # ---------------------------------------------------------

    for image in soup.find_all(
        "img",
        src=True,
    ):

        resource_url = urljoin(
            page_url,
            str(
                image.get(
                    "src",
                    "",
                )
            ),
        )

        if not _valid_resource_url(
            resource_url
        ):
            continue

        graph.add_node(
            resource_url,
            node_type="image",
            url=resource_url,
            hostname=_hostname(
                resource_url
            ),
            external=_is_external(
                resource_url,
                page_url,
            ),
        )

        graph.add_edge(
            page_url,
            resource_url,
            edge_type="loads",
        )

    # ---------------------------------------------------------
    # Iframes
    # ---------------------------------------------------------

    for iframe in soup.find_all(
        "iframe",
        src=True,
    ):

        resource_url = urljoin(
            page_url,
            str(
                iframe.get(
                    "src",
                    "",
                )
            ),
        )

        if not _valid_resource_url(
            resource_url
        ):
            continue

        graph.add_node(
            resource_url,
            node_type="iframe",
            url=resource_url,
            hostname=_hostname(
                resource_url
            ),
            external=_is_external(
                resource_url,
                page_url,
            ),
        )

        graph.add_edge(
            page_url,
            resource_url,
            edge_type="loads",
        )

    # ---------------------------------------------------------
    # Forms
    # ---------------------------------------------------------

    for index, form in enumerate(
        soup.find_all("form")
    ):

        form_id = (
            f"form_{index}"
        )

        method = str(
            form.get(
                "method",
                "GET",
            )
        ).upper()

        graph.add_node(
            form_id,
            node_type="form",
            method=method,
            external=False,
        )

        graph.add_edge(
            page_url,
            form_id,
            edge_type="contains",
        )

        action = form.get(
            "action"
        )

        if not action:
            continue

        action_url = urljoin(
            page_url,
            str(action),
        )

        if not _valid_resource_url(
            action_url
        ):
            continue

        target_node_id = (
            f"form_target::{action_url}"
        )

        graph.add_node(
            target_node_id,
            node_type="form_target",
            url=action_url,
            hostname=_hostname(
                action_url
            ),
            external=_is_external(
                action_url,
                page_url,
            ),
        )

        graph.add_edge(
            form_id,
            target_node_id,
            edge_type="submits_to",
        )

    # ---------------------------------------------------------
    # Runtime network requests
    # ---------------------------------------------------------

    for request in network_requests:

        request_url = str(
            request.get(
                "url",
                "",
            )
        )

        if not _valid_resource_url(
            request_url
        ):
            continue

        method = str(
            request.get(
                "method",
                "",
            )
        ).upper()

        resource_type = str(
            request.get(
                "resource_type",
                "unknown",
            )
        ).lower()

        if graph.has_node(
            request_url
        ):

            graph.nodes[
                request_url
            ]["runtime_requested"] = True

            graph.nodes[
                request_url
            ]["method"] = method

            graph.nodes[
                request_url
            ]["resource_type"] = (
                resource_type
            )

        else:

            graph.add_node(
                request_url,
                node_type="network_resource",
                resource_type=resource_type,
                url=request_url,
                hostname=_hostname(
                    request_url
                ),
                external=_is_external(
                    request_url,
                    page_url,
                ),
                runtime_requested=True,
                method=method,
            )

        if (
            request_url != page_url
            and not graph.has_edge(
                page_url,
                request_url,
            )
        ):

            graph.add_edge(
                page_url,
                request_url,
                edge_type="requests",
            )

    return graph


def extract_graph_features(
    graph: nx.DiGraph,
) -> dict:
    """
    Extract graph-level numerical features.

    These features can later be used for:
    - baseline ML experiments
    - GNN dataset inspection
    - graph-level comparison
    """

    node_count = (
        graph.number_of_nodes()
    )

    edge_count = (
        graph.number_of_edges()
    )

    external_nodes = sum(
        1
        for _, data
        in graph.nodes(
            data=True
        )
        if data.get(
            "external",
            False,
        )
    )

    form_nodes = sum(
        1
        for _, data
        in graph.nodes(
            data=True
        )
        if data.get(
            "node_type"
        ) == "form"
    )

    script_nodes = sum(
        1
        for _, data
        in graph.nodes(
            data=True
        )
        if data.get(
            "node_type"
        )
        in {
            "script",
            "inline_script",
        }
    )

    iframe_nodes = sum(
        1
        for _, data
        in graph.nodes(
            data=True
        )
        if data.get(
            "node_type"
        ) == "iframe"
    )

    form_target_nodes = sum(
        1
        for _, data
        in graph.nodes(
            data=True
        )
        if data.get(
            "node_type"
        ) == "form_target"
    )

    network_nodes = sum(
        1
        for _, data
        in graph.nodes(
            data=True
        )
        if data.get(
            "node_type"
        ) == "network_resource"
    )

    runtime_requested_nodes = sum(
        1
        for _, data
        in graph.nodes(
            data=True
        )
        if data.get(
            "runtime_requested",
            False,
        )
    )

    density = (
        nx.density(
            graph
        )
        if node_count > 1
        else 0.0
    )

    average_degree = (
        sum(
            dict(
                graph.degree()
            ).values()
        )
        / node_count
        if node_count > 0
        else 0.0
    )

    external_node_ratio = (
        external_nodes
        / node_count
        if node_count > 0
        else 0.0
    )

    return {
        "node_count":
            node_count,

        "edge_count":
            edge_count,

        "external_node_count":
            external_nodes,

        "external_node_ratio":
            round(
                external_node_ratio,
                4,
            ),

        "form_node_count":
            form_nodes,

        "form_target_node_count":
            form_target_nodes,

        "script_node_count":
            script_nodes,

        "iframe_node_count":
            iframe_nodes,

        "network_resource_nodes":
            network_nodes,

        "runtime_requested_nodes":
            runtime_requested_nodes,

        "graph_density":
            round(
                density,
                6,
            ),

        "average_degree":
            round(
                average_degree,
                4,
            ),
    }