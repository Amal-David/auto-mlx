from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import unittest

try:
    import jsonschema
except ImportError:  # pragma: no cover - the project itself has no test dependencies
    jsonschema = None  # type: ignore[assignment]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx import validate_relative_posix_path
from auto_mlx.errors import UnsafePathError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SchemaParityTests(unittest.TestCase):
    def _schema(self, name: str) -> dict[str, object]:
        with (PROJECT_ROOT / "schemas" / name).open(encoding="utf-8") as handle:
            return json.load(handle)

    def test_checked_in_schemas_parse_and_are_valid_draft_2020_12_documents(self) -> None:
        schema_paths = sorted((PROJECT_ROOT / "schemas").glob("*.json"))
        self.assertTrue(schema_paths)
        for path in schema_paths:
            with self.subTest(schema=path.name):
                schema = self._schema(path.name)
                self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
                if jsonschema is not None:
                    jsonschema.Draft202012Validator.check_schema(schema)

    def test_artifact_schema_rejects_surrogate_paths_and_python_matches(self) -> None:
        schema = self._schema("artifact.json")
        path_schema = schema["properties"]["path"]  # type: ignore[index]
        invalid_path = "nested/model\ud800.bin"
        instance = {"path": invalid_path, "sha256": "0" * 64, "size_bytes": 0}
        if jsonschema is not None:
            errors = list(jsonschema.Draft202012Validator(schema).iter_errors(instance))
            self.assertTrue(any(error.validator == "not" for error in errors))
        else:
            self.assertIsNotNone(re.search(path_schema["not"]["pattern"], invalid_path))  # type: ignore[index]
        with self.assertRaises(UnsafePathError):
            validate_relative_posix_path(invalid_path)

    def test_declarative_provider_schema_rejects_duplicate_normalized_configs(self) -> None:
        schema = self._schema("declarative_provider.json")
        instance = {
            "provider_id": "grid",
            "configs": [{"mode": "a", "threads": 1}, {"threads": 1, "mode": "a"}],
        }
        self.assertTrue(schema["properties"]["configs"]["uniqueItems"])  # type: ignore[index]
        if jsonschema is not None:
            errors = list(jsonschema.Draft202012Validator(schema).iter_errors(instance))
            self.assertTrue(any(error.validator == "uniqueItems" for error in errors))

    def test_knob_schema_rejects_surrogate_enum_values(self) -> None:
        schema = self._schema("knob.json")
        instance = {"name": "mode", "type": "enum", "values": ["bad\ud800"], "minimum": None, "maximum": None}
        if jsonschema is not None:
            errors = list(jsonschema.Draft202012Validator(schema).iter_errors(instance))
            self.assertTrue(any(error.validator == "not" for error in errors))
        else:
            pattern = schema["allOf"][0]["then"]["properties"]["values"]["items"]["not"]["pattern"]  # type: ignore[index]
            self.assertIsNotNone(re.search(pattern, instance["values"][0]))  # type: ignore[arg-type]

    def test_recursive_workload_schema_calls_out_to_loader_depth_gate(self) -> None:
        schema = self._schema("frozen_workload.json")
        self.assertIn("MAX_JSON_DEPTH=64", schema["$comment"])
        self.assertIn("recursive", schema["$comment"])


if __name__ == "__main__":
    unittest.main()
