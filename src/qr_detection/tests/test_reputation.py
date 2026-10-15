from conftest import ROOT
from quishguard.url_model.reputation import Reputation

rep = Reputation(ROOT / "models" / "tranco_top10k.csv")


def test_popular_platform_is_capped():
    score, reason, rank = rep.adjust("https://forms.gle/abc123", 90.0)
    assert score == 50.0 and reason and rank


def test_low_scores_and_unknown_sites_unchanged():
    assert rep.adjust("https://forms.gle/abc123", 20.0)[0] == 20.0
    assert rep.adjust("http://192.0.2.45/x", 90.0)[0] == 90.0
