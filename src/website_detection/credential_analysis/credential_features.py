from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


# ============================================================
# KEYWORD GROUPS
# ============================================================

EMAIL_TERMS = {
    "email",
    "e-mail",
    "mail",
    "username",
    "user",
    "userid",
    "user-id",
}

PASSWORD_TERMS = {
    "password",
    "pass",
    "passwd",
    "pwd",
    "passcode",
}

OTP_TERMS = {
    "otp",
    "one-time-password",
    "one time password",
    "verification-code",
    "verification code",
    "security-code",
    "security code",
    "auth-code",
    "authentication-code",
}

CARD_TERMS = {
    "card",
    "card-number",
    "card number",
    "credit-card",
    "credit card",
    "debit-card",
    "debit card",
    "pan",
}

CVV_TERMS = {
    "cvv",
    "cvc",
    "security-code",
    "security code",
    "card-code",
}

EXPIRY_TERMS = {
    "expiry",
    "expiration",
    "exp-date",
    "expiry-date",
    "expiration-date",
}

PIN_TERMS = {
    "pin",
    "atm-pin",
    "card-pin",
}

BANK_TERMS = {
    "bank",
    "account-number",
    "account number",
    "iban",
    "routing-number",
    "routing number",
}

IDENTITY_TERMS = {
    "ssn",
    "social-security",
    "social security",
    "passport",
    "national-id",
    "national id",
    "nic",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip().lower()


def normalize_hostname(hostname: str) -> str:
    hostname = safe_text(hostname).rstrip(".")

    if hostname.startswith("www."):
        hostname = hostname[4:]

    return hostname


def extract_hostname(url: str) -> str:
    if not url:
        return ""

    try:
        parsed = urlparse(url)

        return normalize_hostname(
            parsed.hostname or ""
        )

    except Exception:
        return ""


# ============================================================
# INPUT DESCRIPTION
# ============================================================

def build_input_text(element) -> str:
    """
    Build a searchable description from an HTML input field.

    We use:
        type
        name
        id
        placeholder
        autocomplete
        aria-label
    """

    parts = [
        element.get("type", ""),
        element.get("name", ""),
        element.get("id", ""),
        element.get("placeholder", ""),
        element.get("autocomplete", ""),
        element.get("aria-label", ""),
    ]

    return " ".join(
        safe_text(part)
        for part in parts
        if part is not None
    )


# ============================================================
# KEYWORD MATCHING
# ============================================================

def contains_any_term(
    text: str,
    terms: set[str],
) -> bool:

    normalized = safe_text(text)

    if not normalized:
        return False

    # Token-friendly normalized form.
    normalized_spaces = re.sub(
        r"[_\-\.:/]+",
        " ",
        normalized,
    )

    for term in terms:

        term_normalized = term.replace(
            "-",
            " ",
        )

        if (
            term in normalized
            or term_normalized in normalized_spaces
        ):
            return True

    return False


# ============================================================
# FORM DESTINATION
# ============================================================

def analyse_form_destination(
    form,
    page_url: str,
) -> dict[str, Any]:

    action = safe_text(
        form.get(
            "action",
            "",
        )
    )

    if not action:
        return {
            "action_url": "",
            "action_domain": "",
            "external": False,
            "insecure": False,
            "empty_action": True,
        }

    try:
        action_url = urljoin(
            page_url,
            action,
        )

    except Exception:
        action_url = action

    page_domain = extract_hostname(
        page_url
    )

    action_domain = extract_hostname(
        action_url
    )

    external = bool(
        page_domain
        and action_domain
        and page_domain != action_domain
    )

    insecure = (
        action_url.lower().startswith(
            "http://"
        )
    )

    return {
        "action_url": action_url,
        "action_domain": action_domain,
        "external": external,
        "insecure": insecure,
        "empty_action": False,
    }


# ============================================================
# CREDENTIAL FEATURE EXTRACTION
# ============================================================

def extract_credential_features(
    html: str,
    page_url: str,
) -> dict[str, Any]:
    """
    Analyse forms and sensitive-input intent.

    These features indicate what kind of information the page
    appears to request and where forms submit data.

    IMPORTANT:
    These features are evidence only.
    They do not individually classify a site as phishing.
    """

    soup = BeautifulSoup(
        html or "",
        "html.parser",
    )

    forms = soup.find_all(
        "form"
    )

    inputs = soup.find_all(
        [
            "input",
            "textarea",
            "select",
        ]
    )

    # --------------------------------------------------------
    # Sensitive input counts
    # --------------------------------------------------------

    email_fields = 0
    password_fields = 0
    otp_fields = 0
    card_fields = 0
    cvv_fields = 0
    expiry_fields = 0
    pin_fields = 0
    bank_fields = 0
    identity_fields = 0

    for element in inputs:

        description = build_input_text(
            element
        )

        input_type = safe_text(
            element.get(
                "type",
                "",
            )
        )

        autocomplete = safe_text(
            element.get(
                "autocomplete",
                "",
            )
        )

        # ----------------------------------------------------
        # Email / username
        # ----------------------------------------------------

        if (
            input_type == "email"
            or autocomplete in {
                "email",
                "username",
            }
            or contains_any_term(
                description,
                EMAIL_TERMS,
            )
        ):
            email_fields += 1

        # ----------------------------------------------------
        # Password
        # ----------------------------------------------------

        if (
            input_type == "password"
            or autocomplete in {
                "current-password",
                "new-password",
            }
            or contains_any_term(
                description,
                PASSWORD_TERMS,
            )
        ):
            password_fields += 1

        # ----------------------------------------------------
        # OTP
        # ----------------------------------------------------

        if (
            autocomplete == "one-time-code"
            or contains_any_term(
                description,
                OTP_TERMS,
            )
        ):
            otp_fields += 1

        # ----------------------------------------------------
        # Card number
        # ----------------------------------------------------

        if (
            autocomplete == "cc-number"
            or contains_any_term(
                description,
                CARD_TERMS,
            )
        ):
            card_fields += 1

        # ----------------------------------------------------
        # CVV / CVC
        # ----------------------------------------------------

        if (
            autocomplete == "cc-csc"
            or contains_any_term(
                description,
                CVV_TERMS,
            )
        ):
            cvv_fields += 1

        # ----------------------------------------------------
        # Card expiry
        # ----------------------------------------------------

        if (
            autocomplete
            in {
                "cc-exp",
                "cc-exp-month",
                "cc-exp-year",
            }
            or contains_any_term(
                description,
                EXPIRY_TERMS,
            )
        ):
            expiry_fields += 1

        # ----------------------------------------------------
        # PIN
        # ----------------------------------------------------

        if contains_any_term(
            description,
            PIN_TERMS,
        ):
            pin_fields += 1

        # ----------------------------------------------------
        # Bank account
        # ----------------------------------------------------

        if contains_any_term(
            description,
            BANK_TERMS,
        ):
            bank_fields += 1

        # ----------------------------------------------------
        # Identity data
        # ----------------------------------------------------

        if contains_any_term(
            description,
            IDENTITY_TERMS,
        ):
            identity_fields += 1

    # ========================================================
    # FORM-LEVEL DESTINATION ANALYSIS
    # ========================================================

    external_form_actions = 0
    insecure_form_actions = 0
    empty_form_actions = 0

    credential_forms = 0
    external_credential_forms = 0
    insecure_credential_forms = 0

    for form in forms:

        destination = analyse_form_destination(
            form=
                form,
            page_url=
                page_url,
        )

        if destination[
            "external"
        ]:
            external_form_actions += 1

        if destination[
            "insecure"
        ]:
            insecure_form_actions += 1

        if destination[
            "empty_action"
        ]:
            empty_form_actions += 1

        # ----------------------------------------------------
        # Check whether this specific form requests
        # sensitive information.
        # ----------------------------------------------------

        form_inputs = form.find_all(
            [
                "input",
                "textarea",
                "select",
            ]
        )

        form_sensitive = False

        for element in form_inputs:

            description = build_input_text(
                element
            )

            input_type = safe_text(
                element.get(
                    "type",
                    "",
                )
            )

            autocomplete = safe_text(
                element.get(
                    "autocomplete",
                    "",
                )
            )

            if (
                input_type == "password"
                or autocomplete
                in {
                    "current-password",
                    "new-password",
                    "one-time-code",
                    "cc-number",
                    "cc-csc",
                }
                or contains_any_term(
                    description,
                    PASSWORD_TERMS
                    | OTP_TERMS
                    | CARD_TERMS
                    | CVV_TERMS
                    | PIN_TERMS
                    | BANK_TERMS
                    | IDENTITY_TERMS,
                )
            ):
                form_sensitive = True
                break

        if form_sensitive:

            credential_forms += 1

            if destination[
                "external"
            ]:
                external_credential_forms += 1

            if destination[
                "insecure"
            ]:
                insecure_credential_forms += 1

    # ========================================================
    # AGGREGATE SENSITIVE-FIELD COUNTS
    # ========================================================

    sensitive_field_count = (
        password_fields
        + otp_fields
        + card_fields
        + cvv_fields
        + expiry_fields
        + pin_fields
        + bank_fields
        + identity_fields
    )

    financial_field_count = (
        card_fields
        + cvv_fields
        + expiry_fields
        + pin_fields
        + bank_fields
    )

    # ========================================================
    # INTENT FLAGS
    # ========================================================

    login_intent_present = bool(
        password_fields > 0
        or (
            email_fields > 0
            and password_fields > 0
        )
    )

    payment_intent_present = bool(
        card_fields > 0
        or cvv_fields > 0
        or expiry_fields > 0
    )

    otp_intent_present = (
        otp_fields > 0
    )

    identity_intent_present = (
        identity_fields > 0
    )

    external_sensitive_submission = (
        external_credential_forms > 0
    )

    insecure_sensitive_submission = (
        insecure_credential_forms > 0
    )

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "form_count":
            len(
                forms
            ),

        "input_count":
            len(
                inputs
            ),

        "email_fields":
            email_fields,

        "password_fields":
            password_fields,

        "otp_fields":
            otp_fields,

        "card_fields":
            card_fields,

        "cvv_fields":
            cvv_fields,

        "expiry_fields":
            expiry_fields,

        "pin_fields":
            pin_fields,

        "bank_fields":
            bank_fields,

        "identity_fields":
            identity_fields,

        "sensitive_field_count":
            sensitive_field_count,

        "financial_field_count":
            financial_field_count,

        "credential_forms":
            credential_forms,

        "external_form_actions":
            external_form_actions,

        "insecure_form_actions":
            insecure_form_actions,

        "empty_form_actions":
            empty_form_actions,

        "external_credential_forms":
            external_credential_forms,

        "insecure_credential_forms":
            insecure_credential_forms,

        "login_intent_present":
            login_intent_present,

        "payment_intent_present":
            payment_intent_present,

        "otp_intent_present":
            otp_intent_present,

        "identity_intent_present":
            identity_intent_present,

        "external_sensitive_submission":
            external_sensitive_submission,

        "insecure_sensitive_submission":
            insecure_sensitive_submission,
    }


# ============================================================
# MANUAL LOCAL TEST
# ============================================================

if __name__ == "__main__":

    test_html = """
    <html>
        <body>

            <form action="https://collector.example.net/login"
                  method="post">

                <input type="email"
                       name="email">

                <input type="password"
                       name="password">

                <input type="text"
                       name="otp"
                       autocomplete="one-time-code">

                <button type="submit">
                    Login
                </button>

            </form>

        </body>
    </html>
    """

    features = extract_credential_features(
        html=
            test_html,

        page_url=
            "https://example.com/login",
    )

    print()
    print(
        "=" * 70
    )

    print(
        "CREDENTIAL INTENT FEATURE TEST"
    )

    print(
        "=" * 70
    )

    for key, value in features.items():

        print(
            f"{key}: {value}"
        )

    print(
        "=" * 70
    )