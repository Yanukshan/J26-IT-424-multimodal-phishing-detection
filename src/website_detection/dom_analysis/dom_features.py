from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


def _hostname(url: str) -> str:
    """
    Extract and normalize the hostname from a URL.
    """
    try:
        host = urlparse(url).hostname or ""
        host = host.lower()

        if host.startswith("www."):
            host = host[4:]

        return host

    except Exception:
        return ""


def _is_external(url: str, base_url: str) -> bool:
    """
    Return True when the target URL belongs to another host.
    """
    if not url:
        return False

    try:
        absolute_url = urljoin(base_url, url)

        target_host = _hostname(absolute_url)
        base_host = _hostname(base_url)

        if not target_host or not base_host:
            return False

        return target_host != base_host

    except Exception:
        return False


def extract_dom_features(
    html: str,
    page_url: str,
) -> dict:
    """
    Extract structural DOM features.

    These values are research features only. No individual
    feature should be treated as proof of phishing.
    """

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    links = soup.find_all("a")
    forms = soup.find_all("form")
    scripts = soup.find_all("script")
    images = soup.find_all("img")
    iframes = soup.find_all("iframe")
    inputs = soup.find_all("input")

    # ---------------------------------------------------------
    # Input features
    # ---------------------------------------------------------

    password_fields = [
        field
        for field in inputs
        if str(
            field.get("type", "")
        ).lower() == "password"
    ]

    hidden_inputs = [
        field
        for field in inputs
        if str(
            field.get("type", "")
        ).lower() == "hidden"
    ]

    # ---------------------------------------------------------
    # Link features
    # ---------------------------------------------------------

    external_links = 0
    javascript_links = 0
    mailto_links = 0

    for link in links:
        href = link.get("href")

        if not href:
            continue

        href = str(href).strip()
        href_lower = href.lower()

        if href_lower.startswith("javascript:"):
            javascript_links += 1

        elif href_lower.startswith("mailto:"):
            mailto_links += 1

        elif _is_external(
            href,
            page_url,
        ):
            external_links += 1

    # ---------------------------------------------------------
    # Script features
    # ---------------------------------------------------------

    external_scripts = 0
    inline_scripts = 0

    for script in scripts:
        src = script.get("src")

        if not src:
            inline_scripts += 1
            continue

        if _is_external(
            str(src),
            page_url,
        ):
            external_scripts += 1

    # ---------------------------------------------------------
    # Image features
    # ---------------------------------------------------------

    external_images = 0

    for image in images:
        src = image.get("src")

        if (
            src
            and _is_external(
                str(src),
                page_url,
            )
        ):
            external_images += 1

    # ---------------------------------------------------------
    # Iframe features
    # ---------------------------------------------------------

    external_iframes = 0

    for iframe in iframes:
        src = iframe.get("src")

        if (
            src
            and _is_external(
                str(src),
                page_url,
            )
        ):
            external_iframes += 1

    # ---------------------------------------------------------
    # Form features
    # ---------------------------------------------------------

    forms_without_action = 0
    external_form_actions = 0
    insecure_form_actions = 0
    post_forms = 0

    for form in forms:
        action = form.get("action")

        method = str(
            form.get(
                "method",
                "GET",
            )
        ).upper()

        if method == "POST":
            post_forms += 1

        if not action:
            forms_without_action += 1
            continue

        absolute_action = urljoin(
            page_url,
            str(action),
        )

        if _is_external(
            absolute_action,
            page_url,
        ):
            external_form_actions += 1

        if absolute_action.lower().startswith(
            "http://"
        ):
            insecure_form_actions += 1

    # ---------------------------------------------------------
    # Derived features
    # ---------------------------------------------------------

    total_links = len(links)

    external_link_ratio = (
        external_links / total_links
        if total_links > 0
        else 0.0
    )

    login_form_present = (
        len(password_fields) > 0
    )

    # ---------------------------------------------------------
    # Final features
    # ---------------------------------------------------------

    return {
        "links": len(links),
        "forms": len(forms),
        "scripts": len(scripts),
        "images": len(images),
        "iframes": len(iframes),
        "inputs": len(inputs),

        "password_fields": len(
            password_fields
        ),

        "hidden_inputs": len(
            hidden_inputs
        ),

        "external_links": external_links,

        "external_link_ratio": round(
            external_link_ratio,
            4,
        ),

        "javascript_links": javascript_links,
        "mailto_links": mailto_links,

        "external_scripts": external_scripts,
        "inline_scripts": inline_scripts,

        "external_images": external_images,
        "external_iframes": external_iframes,

        "forms_without_action":
            forms_without_action,

        "external_form_actions":
            external_form_actions,

        "insecure_form_actions":
            insecure_form_actions,

        "post_forms": post_forms,

        "login_form_present":
            login_form_present,
    }