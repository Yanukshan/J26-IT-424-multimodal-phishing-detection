import cv2
import numpy as np

from conftest import SAMPLES
from quishguard.decode.decoder import decode


def read(name):
    return cv2.imdecode(np.fromfile(str(SAMPLES / name), np.uint8), cv2.IMREAD_GRAYSCALE)


def test_decodes_sample_codes():
    assert decode(read("1_safe_sliit.png")).text == "https://www.sliit.lk/"
    r = decode(read("2_malicious_ip_download.png"))
    assert r.ok and r.payload_type == "url" and "192.0.2.45" in r.text


def test_blank_image_is_not_decoded():
    assert not decode(np.full((200, 200), 255, np.uint8)).ok
