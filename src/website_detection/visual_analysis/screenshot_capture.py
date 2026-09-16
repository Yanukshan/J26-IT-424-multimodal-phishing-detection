from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

SCREENSHOT_WIDTH = 1280
SCREENSHOT_HEIGHT = 720

DEFAULT_SCREENSHOT_ROOT = Path(
    "data/screenshots"
)


# ============================================================
# URL HASH
# ============================================================

def create_url_hash(
    url: str,
    length: int = 20,
) -> str:
    """
    Create a stable short SHA-256 identifier for a URL.

    We do not use the URL directly as a filename because URLs
    can contain characters that are invalid in Windows paths.
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
# CLASS DIRECTORY
# ============================================================

def get_class_directory(
    label: int,
) -> str:
    """
    Dataset labels:

        0 = phishing
        1 = legitimate
    """

    if label == 0:
        return "phishing"

    if label == 1:
        return "legitimate"

    return "unknown"


# ============================================================
# SCREENSHOT PATH
# ============================================================

def build_screenshot_path(
    url: str,
    label: int,
    root_directory: Path | str = DEFAULT_SCREENSHOT_ROOT,
) -> Path:

    root_directory = Path(
        root_directory
    )

    class_directory = (
        get_class_directory(
            label
        )
    )

    url_hash = create_url_hash(
        url
    )

    output_directory = (
        root_directory
        / class_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        output_directory
        / f"{url_hash}.png"
    )


# ============================================================
# SCREENSHOT CAPTURE
# ============================================================

def capture_page_screenshot(
    page,
    requested_url: str,
    label: int,
    root_directory: Path | str = DEFAULT_SCREENSHOT_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Capture a standardized viewport screenshot.

    This function is observational only.

    It does NOT:
        - click anything
        - submit forms
        - enter credentials
        - download files
        - interact with page controls

    A fixed viewport is preferred for the visual ML pipeline
    because CNN/ViT training benefits from consistent input
    framing.
    """

    screenshot_path = (
        build_screenshot_path(
            url=requested_url,
            label=label,
            root_directory=root_directory,
        )
    )

    # --------------------------------------------------------
    # Reuse an existing screenshot when appropriate.
    # --------------------------------------------------------

    if (
        screenshot_path.exists()
        and not overwrite
    ):

        file_size = (
            screenshot_path.stat().st_size
        )

        return {
            "screenshot_saved":
                True,

            "screenshot_path":
                screenshot_path.as_posix(),

            "screenshot_width":
                SCREENSHOT_WIDTH,

            "screenshot_height":
                SCREENSHOT_HEIGHT,

            "screenshot_file_size":
                file_size,

            "screenshot_reused":
                True,
        }

    try:

        # ----------------------------------------------------
        # Standardize viewport.
        # ----------------------------------------------------

        page.set_viewport_size(
            {
                "width":
                    SCREENSHOT_WIDTH,

                "height":
                    SCREENSHOT_HEIGHT,
            }
        )

        # ----------------------------------------------------
        # Small rendering delay.
        #
        # This allows fonts/layout already requested by the
        # page to settle. It does not trigger interaction.
        # ----------------------------------------------------

        page.wait_for_timeout(
            300
        )

        # ----------------------------------------------------
        # Capture only the viewport.
        #
        # We deliberately avoid full_page=True because some
        # websites are extremely long or dynamically expanding.
        # ----------------------------------------------------

        page.screenshot(
            path=str(
                screenshot_path
            ),
            full_page=False,
            type="png",
        )

        file_size = (
            screenshot_path.stat().st_size
            if screenshot_path.exists()
            else 0
        )

        return {
            "screenshot_saved":
                screenshot_path.exists(),

            "screenshot_path":
                screenshot_path.as_posix(),

            "screenshot_width":
                SCREENSHOT_WIDTH,

            "screenshot_height":
                SCREENSHOT_HEIGHT,

            "screenshot_file_size":
                file_size,

            "screenshot_reused":
                False,
        }

    except Exception as exc:

        return {
            "screenshot_saved":
                False,

            "screenshot_path":
                "",

            "screenshot_width":
                SCREENSHOT_WIDTH,

            "screenshot_height":
                SCREENSHOT_HEIGHT,

            "screenshot_file_size":
                0,

            "screenshot_reused":
                False,

            "screenshot_error_type":
                type(
                    exc
                ).__name__,

            "screenshot_error":
                str(
                    exc
                ),
        }


# ============================================================
# RESULT VALIDATION
# ============================================================

def validate_screenshot_result(
    result: dict[str, Any],
) -> bool:
    """
    Lightweight validation used before a screenshot is accepted
    into the visual dataset.
    """

    if not result.get(
        "screenshot_saved",
        False,
    ):
        return False

    screenshot_path = result.get(
        "screenshot_path",
        "",
    )

    if not screenshot_path:
        return False

    path = Path(
        screenshot_path
    )

    if not path.exists():
        return False

    try:

        file_size = (
            path.stat().st_size
        )

    except OSError:
        return False

    # Reject empty/corrupt-looking output.
    if file_size <= 0:
        return False

    return True