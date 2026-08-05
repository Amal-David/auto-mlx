"""Local attestation-key management.

The attestation key is the out-of-band HMAC secret :mod:`auto_mlx.supervisor`
uses to mint a receipt attestation (see
:func:`auto_mlx.receipts.receipt_attestation`).  It is deliberately kept in
its own directory, entirely separate from the
:class:`auto_mlx.receipts.ContentAddressedStore` root (see
:mod:`auto_mlx.store_config`) -- receipts and decisions are content-addressed
evidence meant to be copied, backed up, or inspected freely; the key must
never end up inside anything treated as bulk, shareable evidence.

Nothing in this module is imported by :mod:`auto_mlx.evaluator`,
:mod:`auto_mlx.executor`, :mod:`auto_mlx.sandbox`, or
:mod:`auto_mlx.measurement` -- only :mod:`auto_mlx.supervisor` is meant to
call :func:`load_attestation_key`.  See
``tests/test_supervisor.py::EvaluatorKeyIsolationTests`` for the module-
boundary proof.

Every check below fails closed with a typed :class:`KeyMaterialError`:
a missing directory or file, any symlinked path component (never silently
followed -- see the docstrings on the two directory/file walks below),
a key directory mode other than ``0700``, a key file mode other than
``0600``, a non-regular key file, or a key shorter than
``MIN_ATTESTATION_KEY_BYTES``.  The key itself is never logged and never
included in any dict/JSON value this module returns.
"""

from __future__ import annotations

import errno
import os
import secrets
import stat
from pathlib import Path
from typing import Final

from .errors import ArtifactIntegrityError, FailureCode, KeyMaterialError
from .paths import _open_root_directory, _open_verified_file


ATTESTATION_KEY_BYTES: Final = 32
MIN_ATTESTATION_KEY_BYTES: Final = 16
MAX_ATTESTATION_KEY_BYTES: Final = 4096
KEY_FILE_NAME: Final = "attestation.key"
KEY_DIR_ENV: Final = "AUTO_MLX_KEY_DIR"
_REQUIRED_DIR_MODE: Final = 0o700
_ANCESTOR_DIR_MODE: Final = 0o755
_REQUIRED_FILE_MODE: Final = 0o600


def default_key_dir() -> Path:
    """``~/.auto-mlx/keys`` -- a lexical default, never itself resolved."""

    return Path.home() / ".auto-mlx" / "keys"


def resolve_key_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the key directory: explicit arg > ``AUTO_MLX_KEY_DIR`` > default.

    This only makes the result an absolute, lexically-normalized path -- it
    deliberately never calls ``Path.resolve()`` (which would silently
    dereference symlinks).  Callers whose input path travels through a
    symlinked ancestor (for example ``$TMPDIR`` on macOS) must resolve it
    themselves before handing it to this module; :func:`load_attestation_key`
    and :func:`store_attestation_key` then correctly -- and strictly --
    reject any symlink still present in the path they are actually given.
    """

    if explicit is not None:
        candidate = Path(os.fspath(explicit))
    else:
        from_env = os.environ.get(KEY_DIR_ENV)
        candidate = Path(from_env) if from_env else default_key_dir()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.normpath(str(candidate)))


def generate_attestation_key() -> bytes:
    """A fresh 256-bit random key. Never logged; never placed in a dict/JSON value."""

    return secrets.token_bytes(ATTESTATION_KEY_BYTES)


def _walk_open_flags() -> int:
    if (
        not hasattr(os, "O_NOFOLLOW")
        or os.open not in getattr(os, "supports_dir_fd", ())
        or getattr(os, "O_DIRECTORY", None) is None
    ):
        raise KeyMaterialError(
            "descriptor-relative no-follow key-directory access is unavailable on this host",
            code=FailureCode.KEY_MATERIAL_INVALID,
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_or_create_key_directory(path: Path) -> int:
    """Create every missing component of ``path`` via a strict O_NOFOLLOW walk.

    Unlike :func:`auto_mlx.receipts._open_or_create_absolute_directory`
    (which calls ``Path.resolve()`` to find the content-addressed store's
    real location, transparently following any symlinked ancestor), key
    material is sensitive enough that this walk never resolves anything: it
    opens each path component, one at a time, directly from ``path.anchor``
    with ``O_NOFOLLOW``, so any symlink anywhere in the chain surfaces as a
    typed, fail-closed :class:`KeyMaterialError` instead of being silently
    followed.  Ancestor directories are created at ``0755``; the final
    (leaf) key directory is always ``fchmod``'d to exactly ``0700`` before
    this returns, regardless of whether it was just created or already
    existed with looser permissions.  The caller owns and must close the
    returned descriptor.
    """

    if not path.is_absolute():
        raise KeyMaterialError("key directory must be an absolute path", code=FailureCode.KEY_MATERIAL_INVALID)
    flags = _walk_open_flags()
    parts = path.parts[1:]
    try:
        descriptor = os.open(path.anchor, flags)
    except OSError as exc:
        raise KeyMaterialError(f"key directory anchor is unavailable: {exc}", code=FailureCode.KEY_MATERIAL_INVALID) from exc
    try:
        for index, component in enumerate(parts):
            is_leaf = index == len(parts) - 1
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(component, _REQUIRED_DIR_MODE if is_leaf else _ANCESTOR_DIR_MODE, dir_fd=descriptor)
                except FileExistsError:
                    pass
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except OSError as exc:
                    raise KeyMaterialError(
                        f"could not open key directory component after creating it: {exc}",
                        code=FailureCode.KEY_MATERIAL_INVALID,
                    ) from exc
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise KeyMaterialError("key directory path contains a symlink", code=FailureCode.KEY_MATERIAL_INVALID) from exc
                raise KeyMaterialError(f"could not open key directory component: {exc}", code=FailureCode.KEY_MATERIAL_INVALID) from exc
            os.close(descriptor)
            descriptor = child
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise KeyMaterialError("key directory path contains a non-directory component", code=FailureCode.KEY_MATERIAL_INVALID)
            if is_leaf:
                os.fchmod(descriptor, _REQUIRED_DIR_MODE)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def store_attestation_key(key: bytes | bytearray, *, key_dir: str | os.PathLike[str] | None = None) -> Path:
    """Atomically create-only write ``key`` as the local attestation secret.

    Raises ``FileExistsError`` if a key is already stored at this location
    (matching :meth:`auto_mlx.receipts.ContentAddressedStore.put_receipt`'s
    immutable-once-written convention) -- this never silently overwrites an
    existing secret.  Callers that want "reuse the existing key if present"
    semantics should call :func:`ensure_attestation_key` instead.
    """

    if type(key) not in {bytes, bytearray} or not key:
        raise KeyMaterialError("attestation key must be non-empty bytes", code=FailureCode.KEY_MATERIAL_INVALID)
    if len(key) < MIN_ATTESTATION_KEY_BYTES:
        raise KeyMaterialError(
            f"attestation key must be at least {MIN_ATTESTATION_KEY_BYTES} bytes, got {len(key)}",
            code=FailureCode.KEY_MATERIAL_INVALID,
        )
    resolved_dir = resolve_key_dir(key_dir)
    dir_descriptor = _open_or_create_key_directory(resolved_dir)
    created = False
    try:
        write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            write_flags |= os.O_NOFOLLOW
        file_descriptor = os.open(KEY_FILE_NAME, write_flags, _REQUIRED_FILE_MODE, dir_fd=dir_descriptor)
        created = True
        try:
            view = memoryview(bytes(key))
            while view:
                written = os.write(file_descriptor, view)
                view = view[written:]
            # Belt-and-braces: os.open's mode argument is subject to umask,
            # so the exact 0600 requirement is re-asserted directly on the
            # descriptor regardless of the process umask.
            os.fchmod(file_descriptor, _REQUIRED_FILE_MODE)
            try:
                os.fsync(file_descriptor)
            except OSError:
                pass
        finally:
            os.close(file_descriptor)
        try:
            os.fsync(dir_descriptor)
        except OSError:
            pass
        return resolved_dir / KEY_FILE_NAME
    except BaseException:
        if created:
            try:
                os.unlink(KEY_FILE_NAME, dir_fd=dir_descriptor)
            except OSError:
                pass
        raise
    finally:
        os.close(dir_descriptor)


def load_attestation_key(*, key_dir: str | os.PathLike[str] | None = None) -> bytes:
    """Load and strictly validate the local attestation key.

    Reuses the repository's descriptor-relative, no-follow artifact-access
    idioms in :mod:`auto_mlx.paths` (the same ones the content-addressed
    receipt/decision store and artifact verification use) to open the key
    directory and key file -- both fail closed on any symlinked path
    component.  Returns raw key bytes; never logs them and never places
    them in a dict/JSON value.
    """

    resolved_dir = resolve_key_dir(key_dir)
    try:
        directory_descriptor = _open_root_directory(resolved_dir)
    except ArtifactIntegrityError as exc:
        code = FailureCode.KEY_MATERIAL_MISSING if exc.code.value == "artifact_missing" else FailureCode.KEY_MATERIAL_INVALID
        raise KeyMaterialError(f"key directory is unavailable: {exc}", code=code) from exc
    try:
        directory_mode = stat.S_IMODE(os.fstat(directory_descriptor).st_mode)
        if directory_mode != _REQUIRED_DIR_MODE:
            raise KeyMaterialError(
                f"key directory has insecure permissions {oct(directory_mode)}; expected {oct(_REQUIRED_DIR_MODE)}",
                code=FailureCode.KEY_MATERIAL_INVALID,
            )
    finally:
        os.close(directory_descriptor)

    try:
        file_descriptor = _open_verified_file(resolved_dir, KEY_FILE_NAME)
    except ArtifactIntegrityError as exc:
        code = FailureCode.KEY_MATERIAL_MISSING if exc.code.value == "artifact_missing" else FailureCode.KEY_MATERIAL_INVALID
        raise KeyMaterialError(f"attestation key file is unavailable: {exc}", code=code) from exc
    try:
        file_mode = stat.S_IMODE(os.fstat(file_descriptor).st_mode)
        if file_mode != _REQUIRED_FILE_MODE:
            raise KeyMaterialError(
                f"attestation key file has insecure permissions {oct(file_mode)}; expected {oct(_REQUIRED_FILE_MODE)}",
                code=FailureCode.KEY_MATERIAL_INVALID,
            )
        payload = os.read(file_descriptor, MAX_ATTESTATION_KEY_BYTES + 1)
    finally:
        os.close(file_descriptor)

    if len(payload) > MAX_ATTESTATION_KEY_BYTES:
        raise KeyMaterialError("attestation key file is implausibly large", code=FailureCode.KEY_MATERIAL_INVALID)
    if len(payload) < MIN_ATTESTATION_KEY_BYTES:
        raise KeyMaterialError(
            f"attestation key must be at least {MIN_ATTESTATION_KEY_BYTES} bytes, got {len(payload)}",
            code=FailureCode.KEY_MATERIAL_INVALID,
        )
    return payload


def ensure_attestation_key(*, key_dir: str | os.PathLike[str] | None = None) -> bytes:
    """Load the existing local key, or generate, store, and return a fresh one."""

    try:
        return load_attestation_key(key_dir=key_dir)
    except KeyMaterialError as exc:
        if exc.code != FailureCode.KEY_MATERIAL_MISSING:
            raise
    store_attestation_key(generate_attestation_key(), key_dir=key_dir)
    # Re-read through the same strict validation path used by every other
    # caller, rather than trusting the bytes just written, so a subtle
    # store-side bug can never hand back unverified key material.
    return load_attestation_key(key_dir=key_dir)


__all__: Final = [
    "ATTESTATION_KEY_BYTES",
    "KEY_DIR_ENV",
    "KEY_FILE_NAME",
    "MAX_ATTESTATION_KEY_BYTES",
    "MIN_ATTESTATION_KEY_BYTES",
    "default_key_dir",
    "ensure_attestation_key",
    "generate_attestation_key",
    "load_attestation_key",
    "resolve_key_dir",
    "store_attestation_key",
]
