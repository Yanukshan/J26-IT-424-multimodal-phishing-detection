import pytest

from quishguard.decode.decoder import payload_type


@pytest.mark.parametrize("text, kind", [
    ("https://www.sliit.lk/", "url"),
    ("www.example.com", "url"),
    ("WIFI:S:Library;T:WPA;P:secret;;", "wifi"),
    ("tel:+94112345678", "phone"),
    ("BEGIN:VCARD\nFN:Test\nEND:VCARD", "contact"),
    ("mailto:someone@example.com", "email"),
    ("Table 12 - order at the counter", "text"),
])
def test_payload_type(text, kind):
    assert payload_type(text) == kind
