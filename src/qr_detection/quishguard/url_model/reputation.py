"""Domain reputation (offline): is the link on a very popular platform?

Lexical URL models cannot judge links on big shared platforms (drive.google.com/file/<random id>,
forms.gle/<id>, chat.whatsapp.com/<id>): the random IDs look exactly like phishing text, and
phishers DO abuse these platforms. So for a registered domain in the Tranco top-N list the link
score is capped at Suspicious: the user is warned to check who placed the code, but a
legitimate Google Form is not reported as Phishing.

Uses the Tranco list already downloaded for training (raw/tranco_top1m.csv: rank,domain).
No network request is made.
"""
from __future__ import annotations

from pathlib import Path

from quishguard.data.urls import get_host, is_ip, registered_domain

SUSPICIOUS_CAP = 50.0


class Reputation:
    def __init__(self, tranco_csv: Path | str, top_n: int = 10_000):
        self.top_n = top_n
        self.rank: dict[str, int] = {}
        with open(tranco_csv, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                r, _, d = line.strip().partition(",")
                if not d:
                    continue
                try:
                    r = int(r)
                except ValueError:
                    continue
                if r > top_n:
                    break
                self.rank[d.lower()] = r
