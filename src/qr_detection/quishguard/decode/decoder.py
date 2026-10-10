"""Module B: safe QR decoding.

Finds, straightens and decodes a QR image and labels what kind of payload it
holds. This module never makes a network request: the decoded link is only
text from here on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

try:  # optional second decoder
    from pyzbar import pyzbar as _pyzbar
except Exception:  # pragma: no cover - pyzbar missing or zbar DLL missing
    _pyzbar = None

try:  # optional third decoder (strongest on real photos and designer codes): pip install zxing-cpp
    import zxingcpp as _zx
except Exception:  # pragma: no cover
    _zx = None

ImageLike = Union[str, Path, np.ndarray, "PIL.Image.Image"]  # noqa: F821

PAD = 24  # CIC images have no white quiet zone; decoders need one


@dataclass
class DecodeResult:
    ok: bool
    text: Optional[str] = None
    payload_type: str = "none"
    method: Optional[str] = None
    corners: Optional[np.ndarray] = field(default=None, repr=False)  # 4x2, in padded image
    image: Optional[np.ndarray] = field(default=None, repr=False)    # padded grayscale


def to_gray(img: ImageLike) -> np.ndarray:
    """Load any supported input as a uint8 grayscale array."""
    if isinstance(img, (str, Path)):
        arr = cv2.imdecode(np.fromfile(str(img), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise ValueError(f"Cannot read image: {img}")
        return arr
    if isinstance(img, np.ndarray):
        arr = img
    else:  # PIL image (mode "1" for CIC images)
        arr = np.array(img.convert("L"))
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    return arr.astype(np.uint8)
