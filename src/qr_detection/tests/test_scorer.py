import pytest

from conftest import ROOT
from quishguard.url_model.predict import UrlScorer


@pytest.fixture(scope="module")
def scorer():
    return UrlScorer(ROOT / "models" / "url_model_v2_portable.joblib")


def test_phishing_links_are_malicious(scorer):
    for url in ("http://192.0.2.45:40606/bin.sh", "http://secure-login.paypal-verify.example/account/update"):
        r = scorer.score(url)
        assert r["malicious"] and r["url_score"] > 50


def test_normal_links_are_not_malicious(scorer):
    for url in ("https://www.sliit.lk/", "https://www.wikipedia.org/"):
        assert not scorer.score(url)["malicious"]


def test_score_gives_reasons(scorer):
    assert len(scorer.score("http://192.0.2.45:40606/bin.sh", top_k=3)["reasons"]) == 3
