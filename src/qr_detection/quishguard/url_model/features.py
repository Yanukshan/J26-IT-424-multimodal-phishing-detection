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
