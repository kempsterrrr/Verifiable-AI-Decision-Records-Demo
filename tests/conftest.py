"""Pytest configuration for the smoke suite.

The demo app (``app/``) isn't pip-installable — only ``ario_mlflow`` is.
A couple of chain-integrity tests need to import ``app.main``, so we put
the repo root on ``sys.path`` once here instead of mutating it inside
individual test bodies.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
