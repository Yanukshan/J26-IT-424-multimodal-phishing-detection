from quishguard.data.urls import get_host, is_ip, normalize_url, registered_domain


def test_normalize_removes_scheme_www_and_trailing_slash():
    assert normalize_url("HTTPS://www.Example.com/Path/") == "example.com/Path"


def test_get_host_drops_port_and_www():
    assert get_host("https://www.sliit.lk:8080/x") == "sliit.lk"


def test_registered_domain_handles_two_part_suffix():
    assert registered_domain("login.paypal.com.evil.co.uk") == "evil.co.uk"


def test_is_ip():
    assert is_ip("192.0.2.1")
    assert not is_ip("sliit.lk")
