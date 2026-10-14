"""Module C2: lexical URL features.

All features are computed on the NORMALISED URL (scheme and "www." removed),
so the model cannot use "starts with http://" as a shortcut. Nothing here
touches the network.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from quishguard.data.urls import get_host, is_ip, normalize_url, registered_domain

SUSPICIOUS_WORDS = (
    "login", "log-in", "signin", "sign-in", "verify", "verification", "account",
    "update", "secure", "security", "confirm", "banking", "bank", "password",
    "passwd", "wallet", "invoice", "payment", "billing", "support", "unlock",
    "suspend", "webscr", "cmd=", "auth", "token", "reset", "recover", "gift",
    "bonus", "free", "promo", "claim", "paypal", "apple", "microsoft", "office365",
    "outlook", "netflix", "amazon", "dhl", "fedex", "usps",
)
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rebrand.ly", "shorturl.at", "rb.gy", "tiny.cc", "t.ly", "s.id",
    "v.gd", "qrco.de", "shorturl.asia", "bl.ink",
}
# free hosting / dynamic DNS / page builders often abused for phishing
FREE_HOSTS = {
    "duckdns.org", "000webhostapp.com", "workers.dev", "pages.dev", "web.app",
    "firebaseapp.com", "weebly.com", "wixsite.com", "blogspot.com", "github.io",
    "glitch.me", "herokuapp.com", "netlify.app", "vercel.app", "ngrok.io",
    "ngrok-free.app", "no-ip.com", "ddns.net", "hopto.org", "zapto.org",
    "sites.google.com", "googleusercontent.com", "square.site", "godaddysites.com",
    "webflow.io", "r2.dev", "ipfs.io", "azurewebsites.net", "appspot.com",
}
RISKY_EXT = (".exe", ".apk", ".scr", ".bat", ".sh", ".bin", ".jar", ".msi", ".dll",
             ".zip", ".rar", ".7z", ".js", ".vbs", ".ps1", ".m", ".mips", ".arm7")
WEB_EXT = (".php", ".html", ".htm", ".asp", ".aspx", ".jsp", ".cgi")

_SYMBOLS = set("-_.~!*'();:@&=+$,/?#[]%")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())
