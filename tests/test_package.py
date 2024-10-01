from __future__ import annotations

import importlib.metadata

import postprod as m


def test_version():
    assert importlib.metadata.version("postprod") == m.__version__
