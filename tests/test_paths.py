from __future__ import annotations

import hashlib
import os
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
        for raw in ("", "/absolute", "C:relative", "C:/absolute", "a//b", "a/./b", "a/../b", "../a", "a\\b", "a\x00b", ".", ".."):
            with self.subTest(raw=raw), self.assertRaises(UnsafePathError):
                validate_relative_posix_path(raw)
        self.assertEqual(validate_relative_posix_path("nested/model.bin"), "nested/model.bin")

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
