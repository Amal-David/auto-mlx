from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx import canonical_bytes, canonical_json, sha256_hex, strict_json_loads
from auto_mlx.errors import CanonicalJSONError, DuplicateKeyError, FailureCode


class CanonicalJSONTests(unittest.TestCase):
    def test_canonical_json_is_sorted_compact_and_utf8(self) -> None:
        value = {"z": "café", "a": [True, None, 4]}
        self.assertEqual(canonical_json(value), '{"a":[true,null,4],"z":"café"}')
        self.assertEqual(canonical_bytes(value), canonical_json(value).encode("utf-8"))
        self.assertEqual(sha256_hex(value), hashlib.sha256(canonical_bytes(value)).hexdigest())

    def test_duplicate_keys_are_rejected_before_identity_can_be_created(self) -> None:
        with self.assertRaises(DuplicateKeyError) as context:
            strict_json_loads('{"name":"first","name":"second"}')
        self.assertEqual(context.exception.code, FailureCode.DUPLICATE_KEY)

    def test_float_nan_and_inf_are_not_accepted_as_contract_data(self) -> None:
        for raw in ("1.0", "NaN", "Infinity", "-Infinity"):
            with self.subTest(raw=raw), self.assertRaises(CanonicalJSONError):
                strict_json_loads(raw)
        with self.assertRaises(CanonicalJSONError):
            canonical_json({"measurement": 1.25})

    def test_canonicalizer_rejects_non_json_mutable_or_executable_values(self) -> None:
        with self.assertRaises(CanonicalJSONError):
            canonical_json({"values": (1, 2)})
        with self.assertRaises(CanonicalJSONError):
            canonical_json({"callable": lambda: None})
