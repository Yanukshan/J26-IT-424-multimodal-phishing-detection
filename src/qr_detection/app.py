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


print("Loading URL model ...")
scorer = UrlScorer(MODELS / "url_model_v2_portable.joblib")
reputation = Reputation(MODELS / "tranco_top10k.csv")
history: list[dict] = []

app = FastAPI(title="QuishGuard 50%", docs_url=None, redoc_url=None)


def check(data: bytes, filename: str) -> dict:
    t0 = time.perf_counter()
    gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise HTTPException(400, "This file is not an image. Upload a PNG or JPG of a QR code.")
    d = decode(gray)
    if not d.ok:
        d = decode_photo(gray)  # camera photos: try cleaned-up versions
    out = {"id": uuid.uuid4().hex[:10], "time": datetime.now().strftime("%H:%M:%S"), "file": filename,
           "content": defang(d.text) if d.ok else "", "payload_type": d.payload_type if d.ok else "none",
           "score": None, "reasons": []}
    if not d.ok:
        out.update(verdict="Unreadable", message="No readable QR code was found in this image.")
    elif d.payload_type != "url":
        kind = PAYLOAD_NAMES.get(d.payload_type, "data")
        out.update(verdict="Normal", message=f"This QR code holds {kind}, not a link. No harmful link inside.")
    else:
        r = scorer.score(d.text, explain=True, top_k=3)
        score, popular, rank = reputation.adjust(d.text, r["url_score"])
        malicious = bool(r["malicious"]) and popular is None
        out["score"] = round(score, 1)
        out["reasons"] = [f"{x['text']} ({x['effect']})" for x in r.get("reasons", [])]
        if malicious:
            out.update(verdict="Malicious", message="This link looks like phishing or malware. Do not open it.")
        elif popular:
            out.update(verdict="Normal", message=f"The link is on a well-known website (rank {rank:,}). "
                                                 "Open it only if you trust who placed the code.")
        else:
            out.update(verdict="Normal", message="This link looks normal.")
    out["ms"] = round((time.perf_counter() - t0) * 1000)
    return out


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")
