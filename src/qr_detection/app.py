"""QuishGuard (50% version): upload a QR code image -> Normal or Malicious.

This version uses two parts of the full system:
  1. Safe QR decoding (the link is read as text and never opened)
  2. The URL model (lexical features + XGBoost) with the popular-platform check

The visual tamper model (ViT / PatchCore), score fusion, alert database and
phone page are part of the full (100%) version.

Run:  python app.py      then open http://localhost:8050
"""
from __future__ import annotations

import argparse
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from quishguard.decode.decoder import decode, decode_photo
from quishguard.url_model.predict import UrlScorer
from quishguard.url_model.reputation import Reputation

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
MAX_BYTES = 15 * 1024 * 1024
MAX_HISTORY = 200

PAYLOAD_NAMES = {"wifi": "Wi-Fi details", "payment": "payment request", "email": "email address",
                 "phone": "phone number", "sms": "SMS", "contact": "contact card", "geo": "map location",
                 "text": "plain text"}

_SCHEME = re.compile(r"^([a-z][a-z0-9+.\-]*)://", re.IGNORECASE)


def defang(text: str | None) -> str:
    """hxxps://login[.]example[.]com/x : shows a link without making it clickable."""
    if not text:
        return ""
    t = text.strip()
    scheme = ""
    m = _SCHEME.match(t)
    if m:
        scheme = re.sub(r"^http", "hxxp", m.group(1), flags=re.IGNORECASE) + "://"
        t = t[m.end():]
    host, sep, rest = t.partition("/")
    return scheme + host.replace(".", "[.]") + sep + rest
