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
_STAT_SUPPORTS_DIR_FD = os.stat in getattr(os, "supports_dir_fd", ())
_STAT_SUPPORTS_NOFOLLOW = os.stat in getattr(os, "supports_follow_symlinks", ())
_MAX_HASH_BYTES = 1 << 30
_NON_REGULAR_OPEN_ERRNOS = {
    errno.EISDIR,
    errno.ENOTDIR,
    errno.ENXIO,
    errno.EOPNOTSUPP,
}
for _errno_name in ("ENOTSUP", "ENODEV", "ENOTTY"):
    _errno_value = getattr(errno, _errno_name, None)
    if _errno_value is not None:
        _NON_REGULAR_OPEN_ERRNOS.add(_errno_value)


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

    if not _NOFOLLOW_AVAILABLE or not _OPEN_SUPPORTS_DIR_FD or getattr(os, "O_DIRECTORY", None) is None:
        raise ArtifactIntegrityError(
            "descriptor-relative no-follow artifact access is unavailable",
            code=FailureCode.ARTIFACT_SECURITY_UNAVAILABLE,
        )
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if directory:
        flags |= os.O_DIRECTORY
    elif nonblocking:
        if not _NONBLOCK_AVAILABLE:
            raise ArtifactIntegrityError(
                "nonblocking artifact classification is unavailable",
                code=FailureCode.ARTIFACT_SECURITY_UNAVAILABLE,
            )
        flags |= os.O_NONBLOCK
    return flags


def _open_error(exc: OSError | ValueError | NotImplementedError, *, path: str) -> ArtifactIntegrityError:
    if not isinstance(exc, OSError):
        return ArtifactIntegrityError("cannot open artifact safely", code=FailureCode.ARTIFACT_IO_ERROR)
    if exc.errno == errno.ENOENT:
        return ArtifactIntegrityError(f"artifact does not exist: {path}", code=FailureCode.ARTIFACT_MISSING)
    if exc.errno == errno.ELOOP:
        return ArtifactIntegrityError(f"artifact path contains a symlink: {path}", code=FailureCode.ARTIFACT_SYMLINK)
    if exc.errno in _NON_REGULAR_OPEN_ERRNOS:
        return ArtifactIntegrityError(f"artifact path is not a regular file: {path}", code=FailureCode.ARTIFACT_NOT_REGULAR)
    if exc.errno in {errno.EACCES, errno.EPERM}:
        return ArtifactIntegrityError(f"cannot access artifact safely: {exc}", code=FailureCode.ARTIFACT_ACCESS)
    return ArtifactIntegrityError(f"cannot open artifact safely: {exc}", code=FailureCode.ARTIFACT_IO_ERROR)


def _read_error(exc: OSError | ValueError | NotImplementedError, *, path: str) -> ArtifactIntegrityError:
    if not isinstance(exc, OSError):
        return ArtifactIntegrityError("cannot read artifact", code=FailureCode.ARTIFACT_IO_ERROR)
    if exc.errno in {errno.EACCES, errno.EPERM}:
        return ArtifactIntegrityError(f"cannot access artifact: {path}: {exc}", code=FailureCode.ARTIFACT_ACCESS)
    return ArtifactIntegrityError(f"cannot read artifact: {path}: {exc}", code=FailureCode.ARTIFACT_IO_ERROR)


def _path_parts(
    path_value: str | os.PathLike[str], *, label: str, allow_empty: bool = False
) -> tuple[Path, str, tuple[str, ...]]:
    try:
        raw = os.fspath(path_value)
    except TypeError as exc:
        raise ArtifactIntegrityError(f"{label} path is not path-like", code=FailureCode.UNSAFE_PATH) from exc
    if not isinstance(raw, str) or not raw:
        raise ArtifactIntegrityError(f"{label} path must be a non-empty string", code=FailureCode.UNSAFE_PATH)
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in raw):
        raise ArtifactIntegrityError(f"{label} path must not contain control characters", code=FailureCode.UNSAFE_PATH)
    if any(0xD800 <= ord(character) <= 0xDFFF for character in raw):
        raise ArtifactIntegrityError(
            f"{label} path must not contain unpaired surrogates", code=FailureCode.UNSAFE_PATH
        )
    try:
        path = Path(raw)
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(f"cannot parse {label} path", code=FailureCode.UNSAFE_PATH) from exc
    parts = path.parts
    anchor = path.anchor or "."
    components = parts[1:] if path.anchor else parts
    if not components and not allow_empty:
        raise ArtifactIntegrityError(f"{label} path has no final component", code=FailureCode.ARTIFACT_MISSING)
    return path, anchor, tuple(components)


def _component_open_error(
    path: Path, component: str, descriptor: int, exc: OSError | ValueError | NotImplementedError
) -> ArtifactIntegrityError:
    if not isinstance(exc, OSError):
        return _open_error(exc, path=str(path))
    if exc.errno == errno.ENOTDIR and _STAT_SUPPORTS_DIR_FD and _STAT_SUPPORTS_NOFOLLOW:
        try:
            entry = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
        except (OSError, ValueError, NotImplementedError):
            entry = None
        if entry is not None and stat.S_ISLNK(entry.st_mode):
            return ArtifactIntegrityError(
                f"artifact path contains a symlink: {path}", code=FailureCode.ARTIFACT_SYMLINK
            )
    return _open_error(exc, path=str(path))


def _close_owned(descriptors: list[int]) -> list[tuple[int, OSError | ValueError]]:
    """Attempt every owned close and discard ownership even when close fails."""

    failures: list[tuple[int, OSError | ValueError]] = []
    while descriptors:
        descriptor = descriptors.pop()
        try:
            os.close(descriptor)
        except (OSError, ValueError) as exc:
            failures.append((descriptor, exc))
    return failures


def _cleanup_detail(failures: list[tuple[int, OSError | ValueError]]) -> str:
    return "; ".join(f"fd {descriptor}: {error}" for descriptor, error in failures)


def _artifact_cleanup_error(
    primary: ArtifactIntegrityError | None,
    failures: list[tuple[int, OSError | ValueError]],
    *,
    context: str,
) -> ArtifactIntegrityError:
    code = primary.code if primary is not None else FailureCode.ARTIFACT_MISSING
    prefix = primary.message if primary is not None else f"{context} cleanup failed"
    return ArtifactIntegrityError(f"{prefix}; {context} cleanup failed: {_cleanup_detail(failures)}", code=code)


def _open_directory_chain(path_value: str | os.PathLike[str], *, label: str) -> tuple[int, tuple[str, ...], Path]:
    """Walk from one stable anchor, opening each directory with O_NOFOLLOW."""

    path, anchor, components = _path_parts(path_value, label=label, allow_empty=True)
    flags = _open_flags(directory=True)
    descriptor: int | None = None
    owned: list[int] = []
    try:
        descriptor = os.open(anchor, flags)
        owned.append(descriptor)
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ArtifactIntegrityError(f"{label} anchor is not a directory", code=FailureCode.ARTIFACT_NOT_REGULAR)
        for component in components:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except (OSError, ValueError, NotImplementedError) as exc:
                raise _component_open_error(path, component, descriptor, exc) from exc
            owned.append(child)
            if not stat.S_ISDIR(os.fstat(child).st_mode):
                raise ArtifactIntegrityError(
                    f"{label} contains a non-directory component: {path}",
                    code=FailureCode.ARTIFACT_NOT_REGULAR,
                )
            try:
                os.close(descriptor)
            except (OSError, ValueError) as exc:
                owned.remove(descriptor)
                raise ArtifactIntegrityError(
                    f"cannot close artifact directory descriptor: {exc}",
                    code=FailureCode.ARTIFACT_MISSING,
                ) from exc
            owned.remove(descriptor)
            descriptor = child
        if descriptor is None:  # pragma: no cover - _path_parts guarantees a component
            raise ArtifactIntegrityError("artifact directory could not be opened", code=FailureCode.ARTIFACT_MISSING)
        return descriptor, components, path
    except ArtifactIntegrityError as primary:
        failures = _close_owned(owned)
        if failures:
            raise _artifact_cleanup_error(primary, failures, context=f"{label} descriptor") from primary
        raise
    except (OSError, ValueError, NotImplementedError) as exc:
        primary = _open_error(exc, path=str(path))
        failures = _close_owned(owned)
        if failures:
            raise _artifact_cleanup_error(primary, failures, context=f"{label} descriptor") from primary
        raise primary from exc


def _open_root_directory(root: str | os.PathLike[str]) -> int:
    """Open the artifact root through a stable anchor without resolving links."""

    root_input, _, _ = _path_parts(root, label="artifact root", allow_empty=True)
    try:
        input_stat = os.lstat(root_input)
    except FileNotFoundError as exc:
        raise ArtifactIntegrityError("artifact root does not exist", code=FailureCode.ARTIFACT_MISSING) from exc
    except OSError as exc:
        raise _open_error(exc, path=str(root_input)) from exc
    if stat.S_ISLNK(input_stat.st_mode):
        raise ArtifactIntegrityError("artifact root must not be a symlink", code=FailureCode.ARTIFACT_SYMLINK)
    descriptor, _, _ = _open_directory_chain(root_input, label="artifact root")
    try:
        opened_stat = os.fstat(descriptor)
    except (OSError, ValueError, NotImplementedError) as exc:
        primary = _open_error(exc, path=str(root_input))
        failures = _close_owned([descriptor])
        if failures:
            raise _artifact_cleanup_error(primary, failures, context="artifact root descriptor") from primary
        raise primary from exc
    if (opened_stat.st_dev, opened_stat.st_ino) != (input_stat.st_dev, input_stat.st_ino):
        primary = ArtifactIntegrityError(
            "artifact root changed while it was being opened", code=FailureCode.IDENTITY_MISMATCH
        )
        failures = _close_owned([descriptor])
        if failures:
            raise _artifact_cleanup_error(primary, failures, context="artifact root descriptor") from primary
        raise primary
    return descriptor


def _open_verified_file(root: str | os.PathLike[str], relative_path: str) -> int:
    """Open an artifact by stable parent descriptors, never by a rebuilt path."""

    safe = validate_relative_posix_path(relative_path)
    components = safe.split("/")
    root_descriptor = _open_root_directory(root)
    owned = [root_descriptor]
    try:
        directory_flags = _open_flags(directory=True)
        for component in components[:-1]:
            try:
                child = os.open(component, directory_flags, dir_fd=owned[-1])
            except (OSError, ValueError, NotImplementedError) as exc:
                raise _component_open_error(Path(safe), component, owned[-1], exc) from exc
            owned.append(child)
            try:
                child_stat = os.fstat(child)
            except (OSError, ValueError, NotImplementedError):
                raise
            if not stat.S_ISDIR(child_stat.st_mode):
                raise ArtifactIntegrityError(
                    f"artifact parent is not a directory: {safe}", code=FailureCode.ARTIFACT_NOT_REGULAR
                )
            parent_descriptor = owned[-2]
            try:
                os.close(parent_descriptor)
            except (OSError, ValueError) as exc:
                owned.remove(parent_descriptor)
                raise ArtifactIntegrityError(
                    f"cannot close artifact parent descriptor: {exc}", code=FailureCode.ARTIFACT_MISSING
                ) from exc
            owned.remove(parent_descriptor)

        parent_descriptor = owned[-1]
        if _STAT_SUPPORTS_DIR_FD and _STAT_SUPPORTS_NOFOLLOW:
            try:
                entry = os.stat(components[-1], dir_fd=parent_descriptor, follow_symlinks=False)
            except (OSError, ValueError, NotImplementedError) as exc:
                if isinstance(exc, OSError) and exc.errno in _NON_REGULAR_OPEN_ERRNOS:
                    raise ArtifactIntegrityError(
                        f"artifact is not a regular file: {safe}", code=FailureCode.ARTIFACT_NOT_REGULAR
                    ) from exc
            else:
                if stat.S_ISLNK(entry.st_mode):
                    raise ArtifactIntegrityError(
                        f"artifact path contains a symlink: {safe}", code=FailureCode.ARTIFACT_SYMLINK
                    )
                if not stat.S_ISREG(entry.st_mode):
                    raise ArtifactIntegrityError(
                        f"artifact is not a regular file: {safe}", code=FailureCode.ARTIFACT_NOT_REGULAR
                    )

        final_descriptor = os.open(
            components[-1], _open_flags(directory=False, nonblocking=True), dir_fd=parent_descriptor
        )
        owned.append(final_descriptor)
        opened_stat = os.fstat(final_descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ArtifactIntegrityError(
                f"artifact is not a regular file: {safe}", code=FailureCode.ARTIFACT_NOT_REGULAR
            )

        parent_descriptors = owned[:-1]
        del owned[:-1]
        failures = _close_owned(parent_descriptors)
        if failures:
            failures.extend(_close_owned(owned))
            raise _artifact_cleanup_error(None, failures, context="artifact descriptor")
        return final_descriptor
    except ArtifactIntegrityError as primary:
        failures = _close_owned(owned)
        if failures:
            raise _artifact_cleanup_error(primary, failures, context="artifact descriptor") from primary
        raise
    except (OSError, ValueError, NotImplementedError) as exc:
        primary = _open_error(exc, path=safe)
        failures = _close_owned(owned)
        if failures:
            raise _artifact_cleanup_error(primary, failures, context="artifact descriptor") from primary
        raise primary from exc


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
    primary: ArtifactIntegrityError | None = None
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
        initial_stat = os.fstat(descriptor)
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
        final_stat = os.fstat(descriptor)
        if final_stat.st_size > limit:
            raise ArtifactIntegrityError(
                "artifact exceeds its declared/configured size bound",
                code=FailureCode.ARTIFACT_SIZE_MISMATCH,
            )
    except ArtifactIntegrityError as exc:
        primary = exc
    except (OSError, ValueError, NotImplementedError) as exc:
        primary = _read_error(exc, path=relative_path)
    failures = _close_owned([descriptor])
    if failures:
        if primary is None:
            raise _artifact_cleanup_error(None, failures, context="artifact descriptor") from failures[0][1]
        raise _artifact_cleanup_error(primary, failures, context="artifact descriptor") from primary
    if primary is not None:
        raise primary
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
