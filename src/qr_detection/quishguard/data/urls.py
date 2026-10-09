"""URL normalisation and domain helpers shared by data prep and the URL model."""
from __future__ import annotations

import ipaddress
import re
from functools import lru_cache
from urllib.parse import urlsplit

# Also catches typos seen in the dataset such as "hhttps://".
_SCHEME_RE = re.compile(r"^\s*[a-z][a-z0-9+.\-]{0,15}://", re.IGNORECASE)

# Used only when tldextract is not installed.
