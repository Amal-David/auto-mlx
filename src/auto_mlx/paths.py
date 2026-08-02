"""Safe relative artifact paths and content-addressed file verification."""

from __future__ import annotations

import hashlib
import errno
import os
import re
import stat
from pathlib import Path, PureWindowsPath
from typing import Any

from .errors import ArtifactIntegrityError, FailureCode, UnsafePathError


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NOFOLLOW_AVAILABLE = hasattr(os, "O_NOFOLLOW")
_NONBLOCK_AVAILABLE = hasattr(os, "O_NONBLOCK")
_OPEN_SUPPORTS_DIR_FD = os.open in getattr(os, "supports_dir_fd", ())
_MAX_HASH_BYTES = 1 << 30


def validate_sha256(value: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ArtifactIntegrityError(
            "sha256 must be exactly 64 lowercase hexadecimal characters",
            code=FailureCode.INVALID_DIGEST,
        )
    return value


def validate_non_negative_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ArtifactIntegrityError(
            f"{label} must be a non-negative integer",
            code=FailureCode.WRONG_TYPE,
        )
    return value


def validate_relative_posix_path(value: str) -> str:
    """Validate a path before it is joined to an artifact root."""

    if type(value) is not str or not value:
        raise UnsafePathError("artifact path must be a non-empty string")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise UnsafePathError("artifact path must not contain control characters")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise UnsafePathError("artifact path must not contain unpaired surrogates")
    if "\x00" in value:
        raise UnsafePathError("artifact path must not contain NUL")
    if "\\" in value:
        raise UnsafePathError("artifact path must use POSIX separators")
    if value.startswith("/") or PureWindowsPath(value).drive:
        raise UnsafePathError("artifact path must be relative")
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise UnsafePathError("artifact path contains an empty, dot, or dotdot component")
    return value


def _open_flags(*, directory: bool, nonblocking: bool = False) -> int:
    """Return flags required for descriptor-relative no-follow traversal."""

    if not _NOFOLLOW_AVAILABLE or not _OPEN_SUPPORTS_DIR_FD:
        raise ArtifactIntegrityError(
            "descriptor-relative no-follow artifact access is unavailable",
            code=FailureCode.ARTIFACT_SECURITY_UNAVAILABLE,
        )
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    elif nonblocking:
        if not _NONBLOCK_AVAILABLE:
            raise ArtifactIntegrityError(
                "nonblocking artifact classification is unavailable",
                code=FailureCode.ARTIFACT_SECURITY_UNAVAILABLE,
            )
        flags |= os.O_NONBLOCK
    return flags


def _open_error(exc: OSError, *, path: str) -> ArtifactIntegrityError:
    if exc.errno == errno.ENOENT:
        return ArtifactIntegrityError(f"artifact does not exist: {path}", code=FailureCode.ARTIFACT_MISSING)
    if exc.errno == errno.ELOOP:
        return ArtifactIntegrityError(f"artifact path contains a symlink: {path}", code=FailureCode.ARTIFACT_SYMLINK)
    if exc.errno in {errno.ENOTDIR, errno.EISDIR}:
        return ArtifactIntegrityError(f"artifact path is not a regular file: {path}", code=FailureCode.ARTIFACT_NOT_REGULAR)
    if exc.errno in {errno.EACCES, errno.EPERM}:
        return ArtifactIntegrityError(f"cannot access artifact safely: {exc}", code=FailureCode.ARTIFACT_ACCESS)
    return ArtifactIntegrityError(f"cannot open artifact safely: {exc}", code=FailureCode.ARTIFACT_IO_ERROR)


def _read_error(exc: OSError, *, path: str) -> ArtifactIntegrityError:
    if exc.errno in {errno.EACCES, errno.EPERM}:
        return ArtifactIntegrityError(f"cannot access artifact: {path}: {exc}", code=FailureCode.ARTIFACT_ACCESS)
    return ArtifactIntegrityError(f"cannot read artifact: {path}: {exc}", code=FailureCode.ARTIFACT_IO_ERROR)


def _close_descriptor(descriptor: int) -> None:
    """Close a descriptor during cleanup without masking the active failure."""

    try:
        os.close(descriptor)
    except OSError:
        pass


def _open_root_directory(root: str | os.PathLike[str]) -> int:
    """Open every root component with a stable descriptor and O_NOFOLLOW."""

    root_input = Path(root)
    try:
        input_stat = os.lstat(root_input)
        if stat.S_ISLNK(input_stat.st_mode):
            raise ArtifactIntegrityError("artifact root must not be a symlink", code=FailureCode.ARTIFACT_SYMLINK)
        root_path = root_input.resolve(strict=True)
    except ArtifactIntegrityError:
        raise
    except FileNotFoundError as exc:
        raise ArtifactIntegrityError("artifact root does not exist", code=FailureCode.ARTIFACT_MISSING) from exc
    except OSError as exc:
        raise _open_error(exc, path=str(root_input)) from exc
    parts = root_path.parts
    anchor = root_path.anchor
    if not anchor:
        raise ArtifactIntegrityError("artifact root must have an absolute anchor", code=FailureCode.ARTIFACT_MISSING)
    flags = _open_flags(directory=True)
    descriptor: int | None = None
    try:
        descriptor = os.open(anchor, flags)
        for component in parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            try:
                os.close(descriptor)
            except OSError:
                _close_descriptor(child)
                raise
            descriptor = child
        root_stat = os.fstat(descriptor)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ArtifactIntegrityError("artifact root must be a directory", code=FailureCode.ARTIFACT_NOT_REGULAR)
        if (root_stat.st_dev, root_stat.st_ino) != (input_stat.st_dev, input_stat.st_ino):
            raise ArtifactIntegrityError(
                "artifact root changed while it was being opened", code=FailureCode.IDENTITY_MISMATCH
            )
        return descriptor
    except ArtifactIntegrityError:
        if descriptor is not None:
            _close_descriptor(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            _close_descriptor(descriptor)
        raise _open_error(exc, path=str(root_path)) from exc


def _open_verified_file(root: str | os.PathLike[str], relative_path: str) -> int:
    """Open an artifact by stable parent descriptors, never by a rebuilt path."""

    safe = validate_relative_posix_path(relative_path)
    components = safe.split("/")
    root_descriptor = _open_root_directory(root)
    descriptors = [root_descriptor]
    final_descriptor: int | None = None
    success = False
    try:
        directory_flags = _open_flags(directory=True)
        for component in components[:-1]:
            try:
                child = os.open(component, directory_flags, dir_fd=descriptors[-1])
            except OSError as exc:
                # macOS reports O_NOFOLLOW|O_DIRECTORY on a symlink as
                # ENOTDIR. Classify it from the same stable parent descriptor;
                # the subsequent security decision is still made by openat
                # with O_NOFOLLOW, never by this metadata probe.
                if exc.errno == errno.ENOTDIR:
                    try:
                        entry = os.stat(component, dir_fd=descriptors[-1], follow_symlinks=False)
                    except OSError as stat_exc:
                        raise _open_error(stat_exc, path=safe) from exc
                    if stat.S_ISLNK(entry.st_mode):
                        raise ArtifactIntegrityError(
                            f"artifact path contains a symlink: {safe}", code=FailureCode.ARTIFACT_SYMLINK
                        ) from exc
                raise
            descriptors.append(child)
            child_stat = os.fstat(child)
            if not stat.S_ISDIR(child_stat.st_mode):
                raise ArtifactIntegrityError(
                    f"artifact parent is not a directory: {safe}", code=FailureCode.ARTIFACT_NOT_REGULAR
                )
        final_descriptor = os.open(
            components[-1], _open_flags(directory=False, nonblocking=True), dir_fd=descriptors[-1]
        )
        opened_stat = os.fstat(final_descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            os.close(final_descriptor)
            final_descriptor = None
            raise ArtifactIntegrityError(
                f"artifact is not a regular file: {safe}", code=FailureCode.ARTIFACT_NOT_REGULAR
            )
        success = True
        return final_descriptor
    except ArtifactIntegrityError:
        raise
    except OSError as exc:
        raise _open_error(exc, path=safe) from exc
    finally:
        if final_descriptor is not None and not success:
            _close_descriptor(final_descriptor)
        for descriptor in reversed(descriptors):
            _close_descriptor(descriptor)


def file_identity(
    root: str | os.PathLike[str],
    relative_path: str,
    *,
    expected_size: int | None = None,
    max_bytes: int = _MAX_HASH_BYTES,
) -> tuple[int, str]:
    """Return a regular file's identity without reading beyond a configured bound."""

    descriptor = _open_verified_file(root, relative_path)
    digest = hashlib.sha256()
    close_error: OSError | None = None
    try:
        if expected_size is not None:
            validate_non_negative_int(expected_size, label="expected_size")
        validate_non_negative_int(max_bytes, label="max_bytes")
        if expected_size is not None and expected_size > max_bytes:
            raise ArtifactIntegrityError(
                "artifact declared size exceeds the configured hash bound",
                code=FailureCode.ARTIFACT_SIZE_MISMATCH,
            )
        limit = max_bytes if expected_size is None else expected_size
        try:
            initial_stat = os.fstat(descriptor)
        except OSError as exc:
            raise _read_error(exc, path=relative_path) from exc
        if initial_stat.st_size > limit:
            raise ArtifactIntegrityError(
                "artifact exceeds its declared/configured size bound",
                code=FailureCode.ARTIFACT_SIZE_MISMATCH,
            )
        size = 0
        remaining = limit
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        try:
            final_stat = os.fstat(descriptor)
        except OSError as exc:
            raise _read_error(exc, path=relative_path) from exc
        if final_stat.st_size > limit:
            raise ArtifactIntegrityError(
                "artifact exceeds its declared/configured size bound",
                code=FailureCode.ARTIFACT_SIZE_MISMATCH,
            )
    except OSError as exc:
        raise _read_error(exc, path=relative_path) from exc
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            close_error = exc
    if close_error is not None:
        raise _read_error(close_error, path=relative_path) from close_error
    return size, digest.hexdigest()


def verify_artifact(root: str | os.PathLike[str], artifact: Any) -> None:
    """Fail closed unless an artifact's path, size, and digest match the file."""

    try:
        path = artifact.path
        expected_size = artifact.size_bytes
        expected_digest = artifact.sha256
    except AttributeError as exc:
        raise ArtifactIntegrityError(
            "artifact must expose path, sha256, and size_bytes",
            code=FailureCode.WRONG_TYPE,
        ) from exc
    validate_relative_posix_path(path)
    validate_sha256(expected_digest)
    validate_non_negative_int(expected_size, label="size_bytes")
    actual_size, actual_digest = file_identity(root, path, expected_size=expected_size)
    if actual_size != expected_size:
        raise ArtifactIntegrityError(
            f"artifact size mismatch: expected {expected_size}, got {actual_size}",
            code=FailureCode.ARTIFACT_SIZE_MISMATCH,
        )
    if actual_digest != expected_digest:
        raise ArtifactIntegrityError(
            "artifact digest mismatch",
            code=FailureCode.ARTIFACT_DIGEST_MISMATCH,
        )


safe_relative_path = validate_relative_posix_path
check_artifact = verify_artifact
