"""Packaged JSON Schema documents for Auto MLX's G0 contracts.

These Draft 2020-12 schemas describe the wire shape of the same documents
``auto_mlx.contracts`` and ``auto_mlx.receipts`` validate in Python. They are
shipped as ordinary package resources -- readable with :mod:`importlib.resources`
whether Auto MLX is installed from a wheel, an sdist, or run directly from a
source checkout -- so external tooling can validate Auto MLX documents without
importing the Python contract classes.

This is the single source of truth for the schema documents; nothing outside
this package duplicates them.
"""

from __future__ import annotations

from importlib import resources
from typing import Final

__all__: Final = ["schema_names", "schema_path", "schema_text"]


def schema_names() -> tuple[str, ...]:
    """Return the sorted file names of every packaged schema document."""

    return tuple(sorted(entry.name for entry in resources.files(__name__).iterdir() if entry.name.endswith(".json")))


def schema_path(name: str) -> resources.abc.Traversable:
    """Return a :mod:`importlib.resources` handle for a packaged schema by file name."""

    return resources.files(__name__).joinpath(name)


def schema_text(name: str) -> str:
    """Return a packaged schema document's raw JSON text by file name."""

    return schema_path(name).read_text(encoding="utf-8")
