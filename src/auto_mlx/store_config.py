"""Store-root resolution for the content-addressed receipt/decision store.

Deliberately a separate module from :mod:`auto_mlx.keys`: the receipt/
decision :class:`auto_mlx.receipts.ContentAddressedStore` root and the
attestation key directory must never nest inside one another. Receipts and
decisions are content-addressed evidence meant to be copied, backed up, or
inspected freely; the attestation key is a secret that must never end up
inside anything treated as bulk, shareable evidence -- and vice versa, the
key directory must never swallow the store.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from .errors import FailureCode, StoreConfigError
from .keys import resolve_key_dir
from .receipts import ContentAddressedStore


STORE_ROOT_ENV: Final = "AUTO_MLX_STORE"
DEFAULT_STORE_DIR_NAME: Final = "auto-mlx-store"


def default_store_root() -> Path:
    """``./auto-mlx-store`` -- relative to the current working directory."""

    return Path.cwd() / DEFAULT_STORE_DIR_NAME


def resolve_store_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the store root: explicit arg > ``AUTO_MLX_STORE`` > default.

    Like :func:`auto_mlx.keys.resolve_key_dir`, this only makes the result
    an absolute, lexically-normalized path -- it never calls
    ``Path.resolve()`` and so never silently dereferences a symlinked
    ancestor.  Callers whose input travels through one (for example
    ``$TMPDIR`` on macOS) must resolve it themselves first.
    """

    if explicit is not None:
        candidate = Path(os.fspath(explicit))
    else:
        from_env = os.environ.get(STORE_ROOT_ENV)
        candidate = Path(from_env) if from_env else default_store_root()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.normpath(str(candidate)))


def _lexical(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.normpath(os.fspath(path)))


def validate_disjoint_roots(
    store_root: str | os.PathLike[str], key_dir: str | os.PathLike[str]
) -> None:
    """Fail closed if the store root and key directory would nest.

    Comparison is lexical (matching the non-resolving convention of both
    resolvers above) -- callers that pass paths through different symlinks
    to the same real directory are responsible for resolving them first,
    exactly as with every other path this module and :mod:`auto_mlx.keys`
    handle.
    """

    real_store = _lexical(store_root)
    real_key = _lexical(key_dir)
    if real_store == real_key or real_store in real_key.parents or real_key in real_store.parents:
        raise StoreConfigError(
            "the receipt store root and the attestation key directory must not nest inside one another "
            f"(store={real_store}, key_dir={real_key})",
            code=FailureCode.STORE_CONFIG_INVALID,
        )


def open_store(
    explicit: str | os.PathLike[str] | None = None,
    *,
    key_dir: str | os.PathLike[str] | None = None,
) -> ContentAddressedStore:
    """Resolve the store root through the convention above and open it.

    Always cross-checks the resolved store root against the resolved key
    directory (see :func:`validate_disjoint_roots`) -- ``key_dir`` follows
    the same explicit-arg/env/default precedence as
    :func:`auto_mlx.keys.resolve_key_dir` when not given directly.
    """

    store_root = resolve_store_root(explicit)
    resolved_key_dir = resolve_key_dir(key_dir)
    validate_disjoint_roots(store_root, resolved_key_dir)
    return ContentAddressedStore(str(store_root))


__all__: Final = [
    "DEFAULT_STORE_DIR_NAME",
    "STORE_ROOT_ENV",
    "default_store_root",
    "open_store",
    "resolve_store_root",
    "validate_disjoint_roots",
]
