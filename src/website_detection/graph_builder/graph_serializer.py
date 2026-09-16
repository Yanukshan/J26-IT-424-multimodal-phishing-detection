from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx


# ============================================================
# CONFIGURATION
# ============================================================

GRAPH_SCHEMA_VERSION = "1.0"

DEFAULT_GRAPH_ROOT = Path(
    "data/graphs"
)


# ============================================================
# CLASS DIRECTORY
# ============================================================

def get_class_directory(
    label: int | None,
) -> str:
    """
    Dataset labels:

        0 = phishing
        1 = legitimate

    Direct crawler tests may not provide a label, in which
    case graphs are stored under "unknown".
    """

    if label == 0:
        return "phishing"

    if label == 1:
        return "legitimate"

    return "unknown"


# ============================================================
# STABLE GRAPH IDENTIFIER
# ============================================================

def create_graph_id(
    url: str,
    length: int = 20,
) -> str:
    """
    Create a stable SHA-256 identifier for the website URL.

    The same 20-character SHA-256 convention is used by the
    screenshot pipeline, which makes it easy to pair:

        screenshot <-> graph <-> tabular row
    """

    normalized = str(
        url or ""
    ).strip()

    digest = hashlib.sha256(
        normalized.encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()

    return digest[
        :length
    ]


# ============================================================
# OUTPUT PATH
# ============================================================

def build_graph_path(
    requested_url: str,
    label: int | None,
    root_directory: Path | str = DEFAULT_GRAPH_ROOT,
) -> Path:
    """
    Build the JSON output path for one website graph.
    """

    root_directory = Path(
        root_directory
    )

    class_directory = (
        get_class_directory(
            label
        )
    )

    output_directory = (
        root_directory
        / class_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph_id = create_graph_id(
        requested_url
    )

    return (
        output_directory
        / f"{graph_id}.json"
    )


# ============================================================
# JSON-SAFE CONVERSION
# ============================================================

def _json_safe(
    value: Any,
) -> Any:
    """
    Convert NetworkX attribute values to JSON-safe values.

    The current resource graph already uses simple strings,
    booleans and numbers, but this helper makes serialization
    robust if later graph features use tuples, sets, Path
    objects or other lightweight values.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        float,
    ):

        if math.isfinite(
            value
        ):
            return value

        return None

    if isinstance(
        value,
        Path,
    ):
        return value.as_posix()

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key):
                _json_safe(
                    item
                )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            _json_safe(
                item
            )
            for item
            in value
        ]

    # Safe fallback for unexpected lightweight attributes.
    return str(
        value
    )


# ============================================================
# GRAPH -> SERIALIZABLE DICTIONARY
# ============================================================

def graph_to_serializable_dict(
    graph: nx.DiGraph,
    requested_url: str,
    final_url: str,
    label: int | None,
    graph_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Convert a NetworkX website graph into a stable JSON-ready
    representation for later GCN / GraphSAGE / GAT processing.

    Node IDs are preserved for research traceability, and every
    node also receives an integer `index`, which is convenient
    when we later create PyTorch Geometric edge_index tensors.
    """

    graph_id = create_graph_id(
        requested_url
    )

    node_ids = list(
        graph.nodes()
    )

    node_to_index = {
        node_id:
            index
        for index, node_id
        in enumerate(
            node_ids
        )
    }

    nodes: list[
        dict[str, Any]
    ] = []

    for node_id in node_ids:

        attributes = dict(
            graph.nodes[
                node_id
            ]
        )

        node_record = {
            "index":
                node_to_index[
                    node_id
                ],

            "id":
                str(
                    node_id
                ),
        }

        for key, value in attributes.items():

            node_record[
                str(
                    key
                )
            ] = _json_safe(
                value
            )

        nodes.append(
            node_record
        )

    edges: list[
        dict[str, Any]
    ] = []

    for source, target, attributes in graph.edges(
        data=True
    ):

        edge_record = {
            "source":
                node_to_index[
                    source
                ],

            "target":
                node_to_index[
                    target
                ],

            "source_id":
                str(
                    source
                ),

            "target_id":
                str(
                    target
                ),
        }

        for key, value in dict(
            attributes
        ).items():

            edge_record[
                str(
                    key
                )
            ] = _json_safe(
                value
            )

        edges.append(
            edge_record
        )

    if graph_features is None:
        graph_features = {}

    return {
        "schema_version":
            GRAPH_SCHEMA_VERSION,

        "graph_id":
            graph_id,

        "directed":
            graph.is_directed(),

        "multigraph":
            graph.is_multigraph(),

        "requested_url":
            str(
                requested_url or ""
            ),

        "final_url":
            str(
                final_url or ""
            ),

        "label":
            label,

        "class_name":
            get_class_directory(
                label
            ),

        "node_count":
            graph.number_of_nodes(),

        "edge_count":
            graph.number_of_edges(),

        "graph_features":
            _json_safe(
                graph_features
            ),

        "nodes":
            nodes,

        "edges":
            edges,
    }


# ============================================================
# SAVE RESOURCE GRAPH
# ============================================================

def save_resource_graph(
    graph: nx.DiGraph,
    requested_url: str,
    final_url: str,
    label: int | None,
    graph_features: dict[str, Any] | None = None,
    root_directory: Path | str = DEFAULT_GRAPH_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Save one full website graph as JSON.

    Returns lightweight artifact metadata for the collector CSV.

    The graph is observational only. This function does not
    interact with the website or browser.
    """

    output_path = build_graph_path(
        requested_url=
            requested_url,

        label=
            label,

        root_directory=
            root_directory,
    )

    if (
        output_path.exists()
        and not overwrite
    ):

        try:

            file_size = (
                output_path.stat().st_size
            )

        except OSError:
            file_size = 0

        return {
            "artifact_saved":
                True,

            "artifact_path":
                output_path.as_posix(),

            "artifact_file_size":
                file_size,

            "artifact_reused":
                True,

            "artifact_schema_version":
                GRAPH_SCHEMA_VERSION,

            "artifact_error_type":
                "",

            "artifact_error":
                "",
        }

    try:

        payload = (
            graph_to_serializable_dict(
                graph=
                    graph,

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

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )

        file_size = (
            output_path.stat().st_size
            if output_path.exists()
            else 0
        )

        return {
            "artifact_saved":
                output_path.exists(),

            "artifact_path":
                output_path.as_posix(),

            "artifact_file_size":
                file_size,

            "artifact_reused":
                False,

            "artifact_schema_version":
                GRAPH_SCHEMA_VERSION,

            "artifact_error_type":
                "",

            "artifact_error":
                "",
        }

    except Exception as exc:

        return {
            "artifact_saved":
                False,

            "artifact_path":
                "",

            "artifact_file_size":
                0,

            "artifact_reused":
                False,

            "artifact_schema_version":
                GRAPH_SCHEMA_VERSION,

            "artifact_error_type":
                type(
                    exc
                ).__name__,

            "artifact_error":
                str(
                    exc
                ),
        }


# ============================================================
# GRAPH ARTIFACT VALIDATION
# ============================================================

def validate_graph_artifact(
    result: dict[str, Any],
) -> bool:
    """
    Lightweight validation of one serialized graph artifact.
    """

    if not result.get(
        "artifact_saved",
        False,
    ):
        return False

    artifact_path = result.get(
        "artifact_path",
        "",
    )

    if not artifact_path:
        return False

    path = Path(
        artifact_path
    )

    if not path.exists():
        return False

    try:

        if (
            path.stat().st_size
            <= 0
        ):
            return False

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            payload = json.load(
                file
            )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return False

    required_keys = {
        "schema_version",
        "graph_id",
        "node_count",
        "edge_count",
        "nodes",
        "edges",
    }

    return required_keys.issubset(
        payload.keys()
    )
