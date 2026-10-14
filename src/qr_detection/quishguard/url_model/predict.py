"""Score one URL with the trained URL model: 0-100 score plus reasons."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from quishguard import config
from quishguard.data.urls import mask_tld, normalize_url
from quishguard.url_model.features import UrlFeaturizer

