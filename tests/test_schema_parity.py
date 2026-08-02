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

    def test_all_schema_objects_have_reusable_no_surrogate_property_name_constraints(self) -> None:
        for path in sorted((PROJECT_ROOT / "schemas").glob("*.json")):
            with self.subTest(schema=path.name):
                schema = self._schema(path.name)
                self.assertEqual(schema["propertyNames"], {"$ref": "#/$defs/no_surrogate_string"})
                self.assertIn("no_surrogate_string", schema["$defs"])

        workload = self._schema("frozen_workload.json")
        self.assertEqual(
            workload["$defs"]["json_value"]["oneOf"][5]["propertyNames"],
            {"$ref": "#/$defs/no_surrogate_string"},
        )
        self.assertEqual(workload["properties"]["knobs"]["maxItems"], 64)

    def test_live_draft_2020_12_validators_reject_surrogates_on_every_schema_surface(self) -> None:
        cases = (
            ("artifact.json", {"path": "model\ud800.bin", "sha256": "0" * 64, "size_bytes": 0}),
            ("candidate_proposal.json", {"provider_id": "grid\ud800", "workload_hash": "0" * 64, "config": {}, "candidate_id": "0" * 64}),
            ("candidate_proposal.json", {"provider_id": "grid", "workload_hash": "0" * 64, "config": {"mode\ud800": "a"}, "candidate_id": "0" * 64}),
            ("candidate_proposal.json", {"provider_id": "grid", "workload_hash": "0" * 64, "config": {"mode": "a\ud800"}, "candidate_id": "0" * 64}),
            ("declarative_provider.json", {"provider_id": "grid\ud800", "configs": []}),
            ("declarative_provider.json", {"provider_id": "grid", "configs": [{"mode\ud800": "a"}]}),
            ("declarative_provider.json", {"provider_id": "grid", "configs": [{"mode": "a\ud800"}]}),
            ("evaluation_policy.json", {"warmup_runs": 0, "measurement_runs": 1, "timeout_seconds": 1, "max_output_bytes": 1, "bad\ud800": 1}),
            ("frozen_workload.json", {"name": "toy\ud800", "artifacts": [], "knobs": [], "parameters": {}}),
            ("frozen_workload.json", {"name": "toy", "artifacts": [], "knobs": [], "parameters": {"nested": {"bad\ud800": "ok"}}}),
            ("frozen_workload.json", {"name": "toy", "artifacts": [], "knobs": [], "parameters": {"nested": "bad\ud800"}}),
            ("knob.json", {"name": "mode\ud800", "type": "enum", "values": ["a"], "minimum": None, "maximum": None}),
            ("knob.json", {"name": "mode", "type": "enum", "values": ["bad\ud800"], "minimum": None, "maximum": None}),
            ("runtime_identity.json", {"runtime": "python\ud800", "version": "3.11", "platform": "Darwin", "machine": "arm64"}),
        )
        for schema_name, instance in cases:
            with self.subTest(schema=schema_name, instance=repr(instance)):
                schema = self._schema(schema_name)
                if jsonschema is not None:
                    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(instance))
                    self.assertTrue(errors)
                else:
                    self.assertEqual(schema["propertyNames"], {"$ref": "#/$defs/no_surrogate_string"})

    def test_recursive_workload_schema_calls_out_to_loader_depth_gate(self) -> None:
        schema = self._schema("frozen_workload.json")
        self.assertIn("MAX_JSON_DEPTH=64", schema["$comment"])
        self.assertIn("recursive", schema["$comment"])


if __name__ == "__main__":
    unittest.main()
