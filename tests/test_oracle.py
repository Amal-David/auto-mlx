from __future__ import annotations

import hashlib
import sys
import tempfile
from unittest.mock import patch
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx import Artifact
from auto_mlx.errors import ArtifactIntegrityError, ContractError
from auto_mlx.oracle import DEFAULT_MAX_ORACLE_ARTIFACT_BYTES, ExactOutputOracle


class OracleTests(unittest.TestCase):
    def test_exact_oracle_freezes_bytes_and_rejects_any_difference(self) -> None:
        mutable = bytearray(b"golden\n")
        oracle = ExactOutputOracle(mutable)
        mutable[:] = b"attacker"
        self.assertEqual(oracle.expected, b"golden\n")
        match = oracle.evaluate(b"golden\n")
        mismatch = oracle.evaluate(b"golden")
        self.assertTrue(match.matched)
        self.assertFalse(mismatch.matched)
        self.assertEqual(mismatch.failure.code.value, "oracle_mismatch")  # type: ignore[union-attr]
        self.assertEqual(mismatch.expected_digest, hashlib.sha256(b"golden\n").hexdigest())

    def test_expected_digest_must_bind_the_expected_bytes(self) -> None:
        with self.assertRaises(ContractError):
            ExactOutputOracle(b"golden", expected_digest="0" * 64)

    def test_oracle_size_limit_is_strict_and_applies_to_expected_and_actual_bytes(self) -> None:
        self.assertLess(DEFAULT_MAX_ORACLE_ARTIFACT_BYTES, 1 << 30)
        for invalid in (True, 0, -1, 1.0, "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ContractError):
                    ExactOutputOracle(b"x", max_bytes=invalid)  # type: ignore[arg-type]

        with self.assertRaises(ContractError):
            ExactOutputOracle(b"xx", max_bytes=1)
        oracle = ExactOutputOracle(b"x", max_bytes=1)
        with self.assertRaises(ContractError):
            oracle.evaluate(b"xx")

    def test_expected_artifact_is_read_from_a_verified_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "expected.bin"
            expected.write_bytes(b"artifact-output")
            artifact = Artifact.from_file(str(root), "expected.bin")
            oracle = ExactOutputOracle.from_artifact(str(root), artifact)
            self.assertTrue(oracle.evaluate(b"artifact-output").matched)
            expected.write_bytes(b"mutated")
            with self.assertRaises(ArtifactIntegrityError):
                ExactOutputOracle.from_artifact(str(root), artifact)

    def test_oracle_artifact_symlink_is_rejected_before_lexical_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.bin"
            target.write_bytes(b"secret")
            link = root / "expected.bin"
            link.symlink_to(target)
            with self.assertRaises(ArtifactIntegrityError):
                ExactOutputOracle.from_artifact(
                    str(root),
                    Artifact("expected.bin", hashlib.sha256(b"secret").hexdigest(), len(b"secret")),
                )

    def test_oracle_descriptor_read_stops_at_declared_size_before_accumulating_extra_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "expected.bin"
            expected.write_bytes(b"a" + b"x" * (2 * 1024 * 1024))
            artifact = Artifact("expected.bin", hashlib.sha256(b"a").hexdigest(), 1)
            with self.assertRaises(ContractError):
                ExactOutputOracle.from_artifact(str(root), artifact)

    def test_one_gib_oracle_artifact_is_rejected_before_open_or_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Artifact("expected.bin", "0" * 64, 1 << 30)
            with patch("auto_mlx.oracle._open_verified_file") as open_verified, patch("auto_mlx.oracle.os.read") as read:
                with self.assertRaises(ContractError):
                    ExactOutputOracle.from_artifact(temporary, artifact)
            open_verified.assert_not_called()
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
