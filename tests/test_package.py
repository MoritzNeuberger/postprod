from __future__ import annotations

import importlib.metadata

import postproc as m


def test_version():
    assert importlib.metadata.version("postproc") == m.__version__
