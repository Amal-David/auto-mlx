from __future__ import annotations

import hashlib
import errno
import json
import os
import re
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx import Artifact, file_identity, validate_relative_posix_path, verify_artifact
from auto_mlx.errors import ArtifactIntegrityError, FailureCode, UnsafePathError
import auto_mlx.paths as paths_module


class SafePathTests(unittest.TestCase):
    def test_relative_posix_paths_reject_traversal_and_cross_platform_escapes(self) -> None:
        for raw in (
            "",
            "/absolute",
            "C:relative",
            "C:/absolute",
            "a//b",
            "a/",
            "a/./b",
            "a/../b",
            "../a",
            "a\\b",
            "a\x00b",
            "a\nb",
            ".",
            "..",
        ):
            with self.subTest(raw=raw), self.assertRaises(UnsafePathError):
                validate_relative_posix_path(raw)
        self.assertEqual(validate_relative_posix_path("nested/model.bin"), "nested/model.bin")

    def test_artifact_schema_and_python_reject_the_same_path_edge_cases(self) -> None:
        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "artifact.json"
        path_schema = json.loads(schema_path.read_text(encoding="utf-8"))["properties"]["path"]
        pattern = path_schema["pattern"]
        not_pattern = path_schema["not"]["pattern"]
        for raw in ("a/", "foo\n/../../bar", "foo\nbar", "a//b", "a/../b", "a\\b", "a\ud800b"):
            with self.subTest(raw=raw):
                if "\ud800" in raw:
                    self.assertIsNotNone(re.search(pattern, raw))
                    self.assertIsNotNone(re.search(not_pattern, raw))
                else:
                    self.assertIsNone(re.search(pattern, raw))
                with self.assertRaises(UnsafePathError):
                    validate_relative_posix_path(raw)

    def test_artifact_reads_remain_on_open_parent_descriptor_after_namespace_swap(self) -> None:
        original_payload = b"trusted bytes"
        attacker_payload = b"attacker bytes"
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            parent = root / "parent"
            parent.mkdir()
            (parent / "model.bin").write_bytes(original_payload)
            attacker = root / "attacker"
            attacker.mkdir()
            (attacker / "model.bin").write_bytes(attacker_payload)
            real_open = os.open
            swapped = False

            def swapping_open(path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
                nonlocal swapped
                if not swapped and str(path).endswith("model.bin"):
                    os.rename(parent, root / "original-parent")
                    os.symlink(attacker, parent, target_is_directory=True)
                    swapped = True
                if dir_fd is None:
                    return real_open(path, flags, mode)
                return real_open(path, flags, mode, dir_fd=dir_fd)

            with patch.object(paths_module.os, "open", swapping_open):
                size, digest = file_identity(root, "parent/model.bin")
            self.assertTrue(swapped)
            self.assertEqual((size, digest), (len(original_payload), hashlib.sha256(original_payload).hexdigest()))

    def test_path_resolution_is_not_publicly_exposed_as_a_lexical_path_helper(self) -> None:
        import auto_mlx

        self.assertFalse(hasattr(auto_mlx, "resolve_safe_path"))

    def test_digest_and_size_bind_the_artifact_bytes(self) -> None:
        payload = b"frozen model bytes\n"
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            path = root / "nested" / "model.bin"
            path.parent.mkdir()
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            artifact = Artifact("nested/model.bin", digest, len(payload))
            self.assertEqual(file_identity(root, artifact.path), (len(payload), digest))
            verify_artifact(root, artifact)
            with self.assertRaises(ArtifactIntegrityError) as context:
                verify_artifact(root, Artifact(artifact.path, digest, len(payload) + 1))
            self.assertEqual(context.exception.code, FailureCode.ARTIFACT_SIZE_MISMATCH)
            with self.assertRaises(ArtifactIntegrityError) as context:
                verify_artifact(root, Artifact(artifact.path, "0" * 64, len(payload)))
            self.assertEqual(context.exception.code, FailureCode.ARTIFACT_DIGEST_MISMATCH)

    def test_file_identity_rejects_bytes_beyond_declared_size_during_read(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "model.bin").write_bytes(b"a" + b"x" * (2 * 1024 * 1024))
            with self.assertRaises(ArtifactIntegrityError) as context:
                file_identity(root, "model.bin", expected_size=1)
            self.assertEqual(context.exception.code, FailureCode.ARTIFACT_SIZE_MISMATCH)

    def test_symlink_and_non_regular_artifacts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "real").write_bytes(b"x")
            (root / "alias").symlink_to(root / "real")
            with self.assertRaises(ArtifactIntegrityError) as context:
                file_identity(root, "alias")
            self.assertEqual(context.exception.code, FailureCode.ARTIFACT_SYMLINK)
            (root / "nested-alias").symlink_to(root)
            with self.assertRaises(ArtifactIntegrityError) as context:
                file_identity(root, "nested-alias/real")
            self.assertEqual(context.exception.code, FailureCode.ARTIFACT_SYMLINK)
            (root / "directory").mkdir()
            with self.assertRaises(ArtifactIntegrityError) as context:
                file_identity(root, "directory")
            self.assertEqual(context.exception.code, FailureCode.ARTIFACT_NOT_REGULAR)

    def test_open_and_read_failures_do_not_look_like_digest_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "model.bin").write_bytes(b"x")
            with patch.object(paths_module.os, "open", side_effect=OSError(errno.EACCES, "denied")):
                with self.assertRaises(ArtifactIntegrityError) as context:
                    file_identity(root, "model.bin")
            self.assertEqual(context.exception.code, FailureCode.ARTIFACT_ACCESS)

            real_read = paths_module.os.read
            real_close = paths_module.os.close
            closed: list[int] = []

            def tracking_close(descriptor: int) -> None:
                closed.append(descriptor)
                real_close(descriptor)

            with patch.object(paths_module.os, "read", side_effect=OSError(errno.EIO, "read failed")):
                with patch.object(paths_module.os, "close", tracking_close):
                    with self.assertRaises(ArtifactIntegrityError) as context:
                        file_identity(root, "model.bin")
            self.assertEqual(context.exception.code, FailureCode.ARTIFACT_IO_ERROR)
            self.assertTrue(closed)
            self.assertIs(real_read, paths_module.os.read)

    def test_parent_descriptor_is_closed_when_child_metadata_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "nested").mkdir()
            (root / "nested" / "model.bin").write_bytes(b"x")
            real_open = paths_module.os.open
            real_fstat = paths_module.os.fstat
            real_close = paths_module.os.close
            opened: list[int] = []
            closed: list[int] = []
            fstat_calls = 0

            def tracking_open(path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
                if dir_fd is None:
                    descriptor = real_open(path, flags, mode)
                else:
                    descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
                opened.append(descriptor)
                return descriptor

            def failing_fstat(descriptor: int) -> os.stat_result:
                nonlocal fstat_calls
                fstat_calls += 1
                if fstat_calls == 2:
                    raise OSError(errno.EIO, "metadata failed")
                return real_fstat(descriptor)

            def tracking_close(descriptor: int) -> None:
                closed.append(descriptor)
                real_close(descriptor)

            with patch.object(paths_module.os, "open", tracking_open):
                with patch.object(paths_module.os, "fstat", failing_fstat):
                    with patch.object(paths_module.os, "close", tracking_close):
                        with self.assertRaises(ArtifactIntegrityError) as context:
                            file_identity(root, "nested/model.bin")
            self.assertEqual(context.exception.code, FailureCode.ARTIFACT_IO_ERROR)
            self.assertGreaterEqual(fstat_calls, 2)
            self.assertIn(opened[-1], closed)
