import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "sample_qr"
sys.path.insert(0, str(ROOT))


@pytest.fixture
def make_qr():
    """Build a QR code PNG (bytes) holding the given text."""
    def _make(text: str) -> bytes:
        img = cv2.QRCodeEncoder.create().encode(text)
        img = cv2.resize(img, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        img = cv2.copyMakeBorder(img, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=255)
        ok, buf = cv2.imencode(".png", img)
        assert ok
        return buf.tobytes()
    return _make


@pytest.fixture
def sample():
    return lambda name: (SAMPLES / name).read_bytes()


@pytest.fixture
def blank_png():
    ok, buf = cv2.imencode(".png", np.full((200, 200), 255, np.uint8))
    return buf.tobytes()
