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

