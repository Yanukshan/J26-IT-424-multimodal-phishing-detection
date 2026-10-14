from quishguard.url_model.features import extract_one


def test_ip_download_link_features():
    f = extract_one("http://192.0.2.45:40606/bin.sh")
    assert f["is_ip"] == 1 and f["has_port"] == 1 and f["risky_ext"] == 1


def test_shortener_detected():
    assert extract_one("https://bit.ly/abc")["is_shortener"] == 1
