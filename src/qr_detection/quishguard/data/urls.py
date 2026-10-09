"""URL normalisation and domain helpers shared by data prep and the URL model."""
from __future__ import annotations

import ipaddress
import re
from functools import lru_cache
from urllib.parse import urlsplit

# Also catches typos seen in the dataset such as "hhttps://".
_SCHEME_RE = re.compile(r"^\s*[a-z][a-z0-9+.\-]{0,15}://", re.IGNORECASE)

# Used only when tldextract is not installed.
_TWO_PART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "co.nz",
    "co.jp", "ne.jp", "or.jp", "co.in", "net.in", "org.in", "co.za", "com.br",
    "com.cn", "net.cn", "org.cn", "com.tw", "com.hk", "com.sg", "com.my",
    "com.mx", "com.ar", "com.tr", "com.pk", "com.ng", "co.id", "co.kr",
    "ac.lk", "com.lk", "gov.lk", "edu.lk", "org.lk", "co.il", "com.ua", "com.ru",
}

try:  # tldextract gives correct registered domains; use its bundled list, no network
    import tldextract

    _EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
except Exception:  # pragma: no cover - fallback path
    _EXTRACT = None


def has_scheme(url: str) -> bool:
    return bool(_SCHEME_RE.match(url or ""))
