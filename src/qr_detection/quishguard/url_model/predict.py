"""Score one URL with the trained URL model: 0-100 score plus reasons."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from quishguard import config
from quishguard.data.urls import mask_tld, normalize_url
from quishguard.url_model.features import UrlFeaturizer

READABLE = {
    "len_url": "URL length", "len_host": "host length", "len_path": "path length",
    "len_query": "query length", "digit_ratio": "share of digits", "letter_ratio": "share of letters",
    "symbol_ratio": "share of symbols", "upper_ratio": "share of capitals",
    "entropy_url": "randomness of URL", "entropy_host": "randomness of host name",
    "host_digit_ratio": "digits in host name", "host_hyphens": "hyphens in host name",
    "n_dots": "number of dots", "n_host_labels": "host name parts",
    "n_subdomain_labels": "number of subdomains", "len_subdomain": "subdomain length",
    "len_domain_label": "domain name length", "len_tld": "TLD length", "is_ip": "raw IP address as host",
    "has_port": "explicit port number", "path_depth": "path depth", "n_query_params": "query parameters",
    "n_at": "'@' characters", "n_percent": "'%' encoding", "n_equals": "'=' characters",
    "n_tilde": "'~' characters", "n_underscore": "'_' characters", "has_punycode": "punycode (look-alike letters)",
    "has_double_slash_in_path": "'//' inside path", "n_suspicious_words": "phishing words (login, verify…)",
    "is_shortener": "link shortener", "is_free_host": "free hosting / dynamic DNS",
    "risky_ext": "download file type (.exe, .apk, .sh…)", "web_ext": "script page (.php, .html…)",
    "max_token_len": "longest word in URL", "n_tokens": "number of URL parts",
    "max_char_run": "repeated characters", "host_vowel_ratio": "vowel share in host (gibberish check)",
    "brand_in_subdomain": "brand name in subdomain", "tld_risk": "TLD often used for phishing",
    "char_ngram_prob": "character patterns look like known phishing URLs",
}


class UrlScorer:
    def __init__(self, path: Path | str = config.MODELS_DIR / "url_model.joblib"):
        b = joblib.load(path)
        self.feat: UrlFeaturizer = b["featurizer"]
        self.model = b["model"]
        if self.model is None and b.get("model_json") is not None:
            # portable bundle: the XGBoost model is stored in XGBoost's own JSON format
            # (loads on any OS / build), not as a pickled Python object
            import xgboost as xgb
            self.model = xgb.XGBClassifier()
            self.model.load_model(bytearray(b["model_json"]))
        self.cal = b["calibrator"]
        self.thresholds = b["thresholds"]
        self.names = b["feature_names"]
        self.char = b.get("char_model")
        self._booster = None
        try:
            import xgboost as xgb
            if isinstance(self.model, xgb.XGBClassifier):
                self._booster = self.model.get_booster()
                self._xgb = xgb
        except ImportError:
            pass
