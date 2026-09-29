"""Guards the cold-start constraint: importing any docuchat module must never
pull in heavy ML dependencies at module scope. Runs each check in a fresh
subprocess so no other test's imports can pollute sys.modules first.
"""

import subprocess
import sys

import pytest

HEAVY = {"torch", "docling", "sentence_transformers"}


@pytest.mark.parametrize(
    "module",
    [
        "docuchat",
        "docuchat.config",
        "docuchat.chunking",
        "docuchat.classify",
        "docuchat.models",
        "docuchat.pipeline",
        "docuchat.ingest",
    ],
)
def test_import_does_not_load_heavy_deps(module):
    pytest.importorskip(module)
    code = (
        f"import {module}, sys; "
        f"loaded = {HEAVY!r} & set(sys.modules); "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
