from __future__ import annotations

import math
import re
from collections import Counter
from ipaddress import ip_address
from typing import Any
from urllib.parse import parse_qsl, urlparse, urlunparse


# ============================================================
# GENERAL SUSPICIOUS URL TOKENS
# ============================================================
#
# These are generic action/security/payment words rather than
# specific company names.
#
# We deliberately avoid brand names such as PayPal, Microsoft,
# Amazon, etc. because brand-specific features could make the
# model memorize particular campaigns instead of learning
# general phishing characteristics.
# ============================================================

SUSPICIOUS_TOKENS = {
    "account",
    "auth",
    "authenticate",
    "authentication",
    "billing",
    "confirm",
    "confirmation",
    "credential",
    "credentials",
    "invoice",
    "login",
    "log-in",
    "payment",
    "password",
    "recover",
    "recovery",
    "secure",
    "security",
    "session",
    "signin",
    "sign-in",
    "support",
    "token",
    "unlock",
    "update",
    "validation",
    "verification",
    "verify",
    "wallet",
}


LOGIN_TOKENS = {
    "login",
    "log-in",
    "signin",
    "sign-in",
    "password",
    "credential",
    "credentials",
}


VERIFY_TOKENS = {
    "verify",
    "verification",
    "validation",
    "confirm",
    "confirmation",
}


ACCOUNT_TOKENS = {
    "account",
    "profile",
    "session",
    "recover",
    "recovery",
    "unlock",
}


SECURITY_TOKENS = {
    "secure",
    "security",
    "auth",
    "authenticate",
    "authentication",
    "token",
}


PAYMENT_TOKENS = {
    "payment",
    "billing",
    "invoice",
    "wallet",
    "card",
}


# ============================================================
# SAFE VALUE HELPERS
# ============================================================

def safe_string(
    value: Any,
) -> str:
    """
    Convert a nullable value to a clean string.
    """

    if value is None:
        return ""

    return str(
        value
    ).strip()


# ============================================================
# URL PREPARATION
# ============================================================

def prepare_url_for_parsing(
    url: str,
) -> str:
    """
    Prepare a URL for urllib parsing.

    If a URL has no explicit scheme, HTTPS is added only for
    parsing purposes.

    Example:

        example.com/login

    becomes:

        https://example.com/login

    This does NOT modify the URL used by the crawler.
    """

    url = safe_string(
        url
    )

    if not url:
        return ""

    parsed = urlparse(
        url
    )

    if parsed.scheme:

        return url

    return (
        "https://"
        + url
    )


# ============================================================
# HOSTNAME NORMALIZATION
# ============================================================

def normalize_hostname(
    hostname: str,
) -> str:
    """
    Normalize hostname for comparison.

    Example:

        WWW.Example.com
            ->
        example.com
    """

    hostname = safe_string(
        hostname
    ).lower()

    hostname = hostname.rstrip(
        "."
    )

    if hostname.startswith(
        "www."
    ):

        hostname = hostname[
            4:
        ]

    return hostname


# ============================================================
# SAFE URL PARSER
# ============================================================

def safe_parse(
    url: str,
):
    """
    Safely parse a URL.

    Returns None if parsing fails.
    """

    prepared = (
        prepare_url_for_parsing(
            url
        )
    )

    if not prepared:
        return None

    try:

        return urlparse(
            prepared
        )

    except Exception:

        return None


# ============================================================
# IP HOST DETECTION
# ============================================================

def is_ip_hostname(
    hostname: str,
) -> bool:
    """
    Return True if the hostname is an IPv4 or IPv6 address.
    """

    hostname = safe_string(
        hostname
    )

    if not hostname:
        return False

    # IPv6 URLs may include square brackets.
    hostname = hostname.strip(
        "[]"
    )

    try:

        ip_address(
            hostname
        )

        return True

    except ValueError:

        return False


# ============================================================
# SHANNON ENTROPY
# ============================================================

def shannon_entropy(
    value: str,
) -> float:
    """
    Calculate Shannon entropy.

    Higher entropy can indicate a more random-looking string.

    Example:

        abc.com
        has relatively low entropy

        a8z91xk2q7.example
        may have higher entropy

    Entropy alone is NOT considered phishing evidence.
    """

    value = safe_string(
        value
    )

    if not value:
        return 0.0

    counts = Counter(
        value
    )

    length = len(
        value
    )

    entropy = 0.0

    for count in counts.values():

        probability = (
            count
            / length
        )

        entropy -= (
            probability
            * math.log2(
                probability
            )
        )

    return round(
        entropy,
        6,
    )


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_url(
    url: str,
) -> list[str]:
    """
    Split a URL into lowercase lexical tokens.

    Example:

        https://secure-login.example.com/account/verify

    may produce tokens including:

        secure
        login
        example
        com
        account
        verify
    """

    value = safe_string(
        url
    ).lower()

    if not value:
        return []

    return [
        token
        for token
        in re.split(
            r"[^a-z0-9]+",
            value,
        )
        if token
    ]


# ============================================================
# TOKEN GROUP MATCHING
# ============================================================

def token_group_present(
    tokens: list[str],
    group: set[str],
) -> bool:
    """
    Check whether a URL token belongs to a token group.
    """

    token_set = set(
        tokens
    )

    return bool(
        token_set
        & group
    )


# ============================================================
# QUERY PARAMETER COUNT
# ============================================================

def count_query_parameters(
    query: str,
) -> int:
    """
    Count URL query parameters.
    """

    if not query:
        return 0

    try:

        return len(
            parse_qsl(
                query,
                keep_blank_values=True,
            )
        )

    except Exception:

        # Fallback for malformed query strings.
        return (
            query.count(
                "&"
            )
            + 1
        )


# ============================================================
# PATH DEPTH
# ============================================================

def calculate_path_depth(
    path: str,
) -> int:
    """
    Count non-empty path segments.

    Example:

        /account/security/login

    depth = 3
    """

    if not path:
        return 0

    segments = [
        segment
        for segment
        in path.split(
            "/"
        )
        if segment
    ]

    return len(
        segments
    )


# ============================================================
# SUBDOMAIN DEPTH
# ============================================================

def calculate_subdomain_depth(
    hostname: str,
) -> int:
    """
    Estimate the number of subdomain levels.

    Example:

        login.secure.example.com

    labels:
        login
        secure
        example
        com

    heuristic subdomain depth = 2

    IMPORTANT:
    This is a hostname-level heuristic.

    It does not use the Public Suffix List, therefore domains
    such as example.co.uk may not be perfectly represented.

    Later, the final research pipeline can use tldextract or a
    Public Suffix List library for registrable-domain analysis.
    """

    hostname = normalize_hostname(
        hostname
    )

    if not hostname:
        return 0

    if is_ip_hostname(
        hostname
    ):
        return 0

    labels = [
        label
        for label
        in hostname.split(
            "."
        )
        if label
    ]

    if len(
        labels
    ) <= 2:

        return 0

    return (
        len(
            labels
        )
        - 2
    )


# ============================================================
# URL CANONICALIZATION
# ============================================================

def canonical_url_key(
    url: str,
) -> str:
    """
    Create a normalized URL key for redirect comparison.

    Fragment is ignored.

    Query string is preserved.
    """

    parsed = safe_parse(
        url
    )

    if parsed is None:
        return ""

    hostname = normalize_hostname(
        parsed.hostname
        or ""
    )

    if not hostname:
        return ""

    scheme = (
        parsed.scheme
        or ""
    ).lower()

    try:

        port = parsed.port

    except ValueError:

        port = None

    if port:

        netloc = (
            f"{hostname}:{port}"
        )

    else:

        netloc = hostname

    path = (
        parsed.path
        or "/"
    )

    if path != "/":

        path = path.rstrip(
            "/"
        )

    return urlunparse(
        (
            scheme,
            netloc,
            path,
            "",
            parsed.query,
            "",
        )
    )


# ============================================================
# SINGLE URL LEXICAL FEATURES
# ============================================================

def extract_single_url_features(
    url: str,
) -> dict[str, Any]:
    """
    Extract lexical and structural URL features.

    No web request is made here.
    """

    raw_url = safe_string(
        url
    )

    parsed = safe_parse(
        raw_url
    )

    # --------------------------------------------------------
    # Default safe result for invalid/empty URLs.
    # --------------------------------------------------------

    if parsed is None:

        return {
            "length": 0,
            "hostname_length": 0,
            "path_length": 0,
            "query_length": 0,
            "fragment_length": 0,

            "https": False,
            "has_query": False,
            "has_fragment": False,

            "hostname_label_count": 0,
            "subdomain_depth": 0,
            "path_depth": 0,
            "query_parameter_count": 0,

            "digit_count": 0,
            "digit_ratio": 0.0,

            "letter_count": 0,
            "letter_ratio": 0.0,

            "dot_count": 0,
            "hyphen_count": 0,
            "underscore_count": 0,
            "at_count": 0,
            "percent_count": 0,
            "ampersand_count": 0,
            "equals_count": 0,
            "special_character_count": 0,

            "has_ip_hostname": False,
            "has_punycode": False,

            "has_explicit_port": False,
            "has_nonstandard_port": False,

            "has_username": False,
            "has_password_in_url": False,

            "double_slash_in_path": False,

            "entropy": 0.0,
            "hostname_entropy": 0.0,
            "path_entropy": 0.0,

            "token_count": 0,
            "suspicious_token_count": 0,

            "has_login_token": False,
            "has_verify_token": False,
            "has_account_token": False,
            "has_security_token": False,
            "has_payment_token": False,
        }

    hostname = normalize_hostname(
        parsed.hostname
        or ""
    )

    path = (
        parsed.path
        or ""
    )

    query = (
        parsed.query
        or ""
    )

    fragment = (
        parsed.fragment
        or ""
    )

    tokens = tokenize_url(
        raw_url
    )

    # --------------------------------------------------------
    # Character counts
    # --------------------------------------------------------

    digit_count = sum(
        1
        for character
        in raw_url
        if character.isdigit()
    )

    letter_count = sum(
        1
        for character
        in raw_url
        if character.isalpha()
    )

    special_character_count = sum(
        1
        for character
        in raw_url
        if not character.isalnum()
    )

    url_length = len(
        raw_url
    )

    if url_length > 0:

        digit_ratio = (
            digit_count
            / url_length
        )

        letter_ratio = (
            letter_count
            / url_length
        )

    else:

        digit_ratio = 0.0
        letter_ratio = 0.0

    # --------------------------------------------------------
    # Hostname labels
    # --------------------------------------------------------

    hostname_labels = [
        label
        for label
        in hostname.split(
            "."
        )
        if label
    ]

    # --------------------------------------------------------
    # Port
    # --------------------------------------------------------

    try:

        port = parsed.port

    except ValueError:

        port = None

    has_explicit_port = (
        port is not None
    )

    standard_ports = {
        80,
        443,
    }

    has_nonstandard_port = (
        has_explicit_port
        and port
        not in standard_ports
    )

    # --------------------------------------------------------
    # Suspicious tokens
    # --------------------------------------------------------

    suspicious_token_count = sum(
        1
        for token
        in tokens
        if token
        in SUSPICIOUS_TOKENS
    )

    # --------------------------------------------------------
    # Build feature dictionary
    # --------------------------------------------------------

    return {
        # ----------------------------------------------------
        # Length features
        # ----------------------------------------------------

        "length":
            url_length,

        "hostname_length":
            len(
                hostname
            ),

        "path_length":
            len(
                path
            ),

        "query_length":
            len(
                query
            ),

        "fragment_length":
            len(
                fragment
            ),

        # ----------------------------------------------------
        # Scheme / structure
        # ----------------------------------------------------

        "https":
            (
                parsed.scheme.lower()
                == "https"
            ),

        "has_query":
            bool(
                query
            ),

        "has_fragment":
            bool(
                fragment
            ),

        "hostname_label_count":
            len(
                hostname_labels
            ),

        "subdomain_depth":
            calculate_subdomain_depth(
                hostname
            ),

        "path_depth":
            calculate_path_depth(
                path
            ),

        "query_parameter_count":
            count_query_parameters(
                query
            ),

        # ----------------------------------------------------
        # Character distribution
        # ----------------------------------------------------

        "digit_count":
            digit_count,

        "digit_ratio":
            round(
                digit_ratio,
                6,
            ),

        "letter_count":
            letter_count,

        "letter_ratio":
            round(
                letter_ratio,
                6,
            ),

        "dot_count":
            raw_url.count(
                "."
            ),

        "hyphen_count":
            raw_url.count(
                "-"
            ),

        "underscore_count":
            raw_url.count(
                "_"
            ),

        "at_count":
            raw_url.count(
                "@"
            ),

        "percent_count":
            raw_url.count(
                "%"
            ),

        "ampersand_count":
            raw_url.count(
                "&"
            ),

        "equals_count":
            raw_url.count(
                "="
            ),

        "special_character_count":
            special_character_count,

        # ----------------------------------------------------
        # Host features
        # ----------------------------------------------------

        "has_ip_hostname":
            is_ip_hostname(
                hostname
            ),

        "has_punycode":
            any(
                label.startswith(
                    "xn--"
                )
                for label
                in hostname_labels
            ),

        "has_explicit_port":
            has_explicit_port,

        "has_nonstandard_port":
            has_nonstandard_port,

        # ----------------------------------------------------
        # Credentials inside URL
        # ----------------------------------------------------

        "has_username":
            bool(
                parsed.username
            ),

        "has_password_in_url":
            bool(
                parsed.password
            ),

        # ----------------------------------------------------
        # Path anomaly indicator
        # ----------------------------------------------------

        "double_slash_in_path":
            (
                "//"
                in path
            ),

        # ----------------------------------------------------
        # Entropy
        # ----------------------------------------------------

        "entropy":
            shannon_entropy(
                raw_url.lower()
            ),

        "hostname_entropy":
            shannon_entropy(
                hostname
            ),

        "path_entropy":
            shannon_entropy(
                path.lower()
            ),

        # ----------------------------------------------------
        # Lexical tokens
        # ----------------------------------------------------

        "token_count":
            len(
                tokens
            ),

        "suspicious_token_count":
            suspicious_token_count,

        "has_login_token":
            token_group_present(
                tokens,
                LOGIN_TOKENS,
            ),

        "has_verify_token":
            token_group_present(
                tokens,
                VERIFY_TOKENS,
            ),

        "has_account_token":
            token_group_present(
                tokens,
                ACCOUNT_TOKENS,
            ),

        "has_security_token":
            token_group_present(
                tokens,
                SECURITY_TOKENS,
            ),

        "has_payment_token":
            token_group_present(
                tokens,
                PAYMENT_TOKENS,
            ),
    }


# ============================================================
# FINAL URL / REDIRECT FEATURES
# ============================================================

def extract_redirect_features(
    requested_url: str,
    final_url: str | None,
) -> dict[str, Any]:
    """
    Compare requested and final URLs.

    This does not decide whether a redirect is malicious.

    A redirect is simply recorded as behavior/evidence.
    """

    requested_url = safe_string(
        requested_url
    )

    final_url = safe_string(
        final_url
    )

    if not final_url:

        return {
            "redirected": False,
            "hostname_changed": False,
            "scheme_changed": False,

            "final_length": 0,
            "final_hostname_length": 0,
            "final_https": False,
            "final_subdomain_depth": 0,
            "final_entropy": 0.0,
        }

    requested = safe_parse(
        requested_url
    )

    final = safe_parse(
        final_url
    )

    if requested is None or final is None:

        return {
            "redirected": False,
            "hostname_changed": False,
            "scheme_changed": False,

            "final_length":
                len(
                    final_url
                ),

            "final_hostname_length": 0,
            "final_https": False,
            "final_subdomain_depth": 0,

            "final_entropy":
                shannon_entropy(
                    final_url.lower()
                ),
        }

    requested_hostname = (
        normalize_hostname(
            requested.hostname
            or ""
        )
    )

    final_hostname = (
        normalize_hostname(
            final.hostname
            or ""
        )
    )

    requested_key = (
        canonical_url_key(
            requested_url
        )
    )

    final_key = (
        canonical_url_key(
            final_url
        )
    )

    redirected = (
        bool(
            requested_key
        )
        and bool(
            final_key
        )
        and requested_key
        != final_key
    )

    hostname_changed = (
        bool(
            requested_hostname
        )
        and bool(
            final_hostname
        )
        and requested_hostname
        != final_hostname
    )

    scheme_changed = (
        requested.scheme.lower()
        != final.scheme.lower()
    )

    return {
        "redirected":
            redirected,

        "hostname_changed":
            hostname_changed,

        "scheme_changed":
            scheme_changed,

        "final_length":
            len(
                final_url
            ),

        "final_hostname_length":
            len(
                final_hostname
            ),

        "final_https":
            (
                final.scheme.lower()
                == "https"
            ),

        "final_subdomain_depth":
            calculate_subdomain_depth(
                final_hostname
            ),

        "final_entropy":
            shannon_entropy(
                final_url.lower()
            ),
    }


# ============================================================
# PUBLIC FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_url_features(
    requested_url: str,
    final_url: str | None = None,
) -> dict[str, Any]:
    """
    Main function used by the website phishing pipeline.

    Inputs
    ------
    requested_url:
        Original URL supplied to the crawler.

    final_url:
        Final browser URL after navigation/redirects.
        This can be None when it is not yet known.

    Returns
    -------
    dict
        Numerical / boolean URL-domain features.

    Example
    -------

    features = extract_url_features(
        requested_url=
            "https://secure-login.example.com/account/verify",

        final_url=
            "https://example.com/login",
    )

    IMPORTANT
    ---------
    These values are evidence only.

    None of these individual values should automatically
    classify a website as phishing.
    """

    features = (
        extract_single_url_features(
            requested_url
        )
    )

    redirect_features = (
        extract_redirect_features(
            requested_url=
                requested_url,

            final_url=
                final_url,
        )
    )

    features.update(
        redirect_features
    )

    return features


# ============================================================
# LOCAL MANUAL TEST
# ============================================================

if __name__ == "__main__":

    test_requested_url = (
        "https://secure-login.example.com/"
        "account/verify?id=12345&session=abc"
    )

    test_final_url = (
        "https://example.com/login"
    )

    result = extract_url_features(
        requested_url=
            test_requested_url,

        final_url=
            test_final_url,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "URL FEATURE EXTRACTION TEST"
    )

    print(
        "=" * 70
    )

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

    print(
        "=" * 70
    )