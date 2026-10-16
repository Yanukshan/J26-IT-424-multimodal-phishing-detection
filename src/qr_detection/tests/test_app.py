import pytest
from fastapi import HTTPException

import app


def test_defang_hides_link():
    assert app.defang("https://login.example.com/a.b") == "hxxps://login[.]example[.]com/a.b"
    assert app.defang(None) == ""
