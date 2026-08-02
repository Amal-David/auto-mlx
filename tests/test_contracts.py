from __future__ import annotations

import copy
import json
import math
import sys
import tomllib
from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx import Artifact, CandidateProposal, EvaluationPolicy, FrozenWorkload, Knob, RuntimeIdentity, canonical_json, sha256_hex, validate_config
from auto_mlx.errors import AutoMLXError, CanonicalJSONError, ContractError, Failure, FailureCode, UnknownFieldError, UnsafePathError
from auto_mlx.contracts import MAX_CONFIG_ENTRIES, MAX_JSON_DEPTH, MAX_MEASUREMENT_RUNS, MAX_POLICY_OUTPUT_BYTES, MAX_WARMUP_RUNS


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enum = Knob("mode", "enum", values=("eager", "compiled"))
        self.integer = Knob("batch", "integer", minimum=1, maximum=4)
        self.boolean = Knob("cache", "bool")
        self.parameters = {"inputs": {"tokens": [1, 2, 3]}, "seed": 7}
        self.workload = FrozenWorkload(
            "toy",
            knobs=(self.enum, self.integer, self.boolean),
            parameters=self.parameters,
        )

    def test_workload_deep_freezes_and_hashes_every_declared_input(self) -> None:
        original_hash = self.workload.workload_hash
        self.parameters["inputs"]["tokens"].append(99)
        self.assertEqual(self.workload.parameters["inputs"]["tokens"], (1, 2, 3))
        self.assertEqual(self.workload.workload_hash, original_hash)
        altered = FrozenWorkload("toy", knobs=self.workload.knobs, parameters={"seed": 8})
        self.assertNotEqual(altered.workload_hash, original_hash)

    def test_unknown_fields_and_duplicate_keys_are_not_tolerated(self) -> None:
        data = self.workload.to_dict()
        data["surprise"] = True
        with self.assertRaises(UnknownFieldError):
            FrozenWorkload.from_dict(data)
        with self.assertRaises(UnknownFieldError):
            EvaluationPolicy.from_dict({**EvaluationPolicy().to_dict(), "threshold": 1})
        with self.assertRaises(ContractError):
            FrozenWorkload.from_json('{"name":"toy","name":"other","artifacts":[],"knobs":[],"parameters":{}}')

    def test_integer_fields_reject_booleans_and_floats(self) -> None:
        with self.assertRaises(ContractError):
            Artifact("model.bin", "0" * 64, True)
        with self.assertRaises(ContractError):
            Artifact("model.bin", "not-a-digest", 0)
        with self.assertRaises(ContractError):
            EvaluationPolicy(warmup_runs=0.0)  # type: ignore[arg-type]
        with self.assertRaises(ContractError):
            FrozenWorkload("toy", knobs=(Knob("x", "integer", minimum=1, maximum=2),), parameters={"ratio": 0.5})
        with self.assertRaises(ContractError):
            Knob("candidate_id", "enum", values=("attacker-choice",))

    def test_enum_values_validate_unicode_before_acceptance_and_serialization(self) -> None:
        knob = Knob("mode", "enum", values=("café", "compiled"))
        self.assertTrue(knob.accepts("café"))
        self.assertFalse(knob.accepts("missing"))
        self.assertFalse(knob.accepts("bad\ud800"))
        self.assertEqual(Knob.from_json(knob.to_json()), knob)
        self.assertIn('"values":["café","compiled"]', knob.to_json())

        for values in (("bad\ud800",), ["bad\udfff"]):
            with self.subTest(values=repr(values)), self.assertRaises(ContractError) as context:
                Knob("bad", "enum", values=values)
            self.assertEqual(context.exception.code, FailureCode.INVALID_UNICODE)

        with self.assertRaises(ContractError) as context:
            Knob.from_dict(
                {"name": "bad", "type": "enum", "values": ["bad\ud800"], "minimum": None, "maximum": None}
            )
        self.assertEqual(context.exception.code, FailureCode.INVALID_UNICODE)

    def test_malformed_knob_and_sequence_shapes_never_escape_as_raw_type_errors(self) -> None:
        for malformed_type in (None, 1, [], {}):
            with self.subTest(malformed_type=malformed_type), self.assertRaises(ContractError):
                Knob("bad", malformed_type)  # type: ignore[arg-type]
        with self.assertRaises(ContractError):
            FrozenWorkload("bad", artifacts=None)  # type: ignore[arg-type]
        with self.assertRaises(ContractError):
            FrozenWorkload("bad", knobs=(object(),))  # type: ignore[arg-type]

    def test_failure_details_are_recursively_immutable_and_serializable(self) -> None:
        source = {"nested": {"items": [{"value": 1}]}}
        failure = Failure(FailureCode.INVALID_VALUE, "bad input", source)
        source["nested"]["items"][0]["value"] = 99
        self.assertEqual(failure.details["nested"]["items"][0]["value"], 1)
        with self.assertRaises(TypeError):
            failure.details["nested"]["items"][0]["value"] = 2  # type: ignore[index]
        self.assertEqual(failure.to_dict()["details"], {"nested": {"items": [{"value": 1}]}})

    def test_error_codes_and_failure_details_are_validated_at_construction(self) -> None:
        with self.assertRaises(TypeError):
            AutoMLXError("bad code", code="invalid")  # type: ignore[arg-type]
        for details in ({"bad": {1, 2}}, {"bad": object()}, {"bad": math.nan}, {"bad": "\ud800"}, {"bad\ud800": 1}):
            with self.subTest(details=repr(details)), self.assertRaises(TypeError):
                Failure(FailureCode.INVALID_VALUE, "bad", details)

    def test_error_messages_are_canonical_strings_before_serialization(self) -> None:
        message = "bad café"
        failure = Failure(FailureCode.INVALID_VALUE, message)
        error = AutoMLXError(message, code=FailureCode.INVALID_VALUE)
        expected = '{"code":"invalid_value","details":{},"message":"bad café"}'
        self.assertEqual(canonical_json(failure.to_dict()), expected)
        self.assertEqual(canonical_json(error.as_failure().to_dict()), expected)

        for invalid in (None, 1, object()):
            with self.subTest(invalid=repr(invalid)), self.assertRaises(TypeError):
                Failure(FailureCode.INVALID_VALUE, invalid)  # type: ignore[arg-type]
            with self.subTest(api="AutoMLXError", invalid=repr(invalid)), self.assertRaises(TypeError):
                AutoMLXError(invalid)  # type: ignore[arg-type]
        for invalid in ("", "bad\ud800"):
            with self.subTest(invalid=repr(invalid)), self.assertRaises(ValueError):
                Failure(FailureCode.INVALID_VALUE, invalid)
            with self.subTest(api="AutoMLXError", invalid=repr(invalid)), self.assertRaises(ValueError):
                AutoMLXError(invalid)

    def test_error_subclasses_keep_stable_failure_codes(self) -> None:
        unknown = UnknownFieldError("extra")
        unsafe = UnsafePathError("../escape")
        self.assertEqual(unknown.code, FailureCode.UNKNOWN_FIELD)
        self.assertEqual(unsafe.code, FailureCode.UNSAFE_PATH)

    def test_distribution_metadata_declares_all_checked_in_schemas(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with (project_root / "pyproject.toml").open("rb") as handle:
            config = tomllib.load(handle)
        data_files = config["tool"]["setuptools"]["data-files"]
        self.assertIn("schemas/*.json", data_files["schemas"])
        self.assertEqual(
            {path.name for path in (project_root / "schemas").glob("*.json")},
            {
                "artifact.json",
                "candidate_proposal.json",
                "declarative_provider.json",
                "evaluation_policy.json",
                "frozen_workload.json",
                "knob.json",
                "runtime_identity.json",
            },
        )

    def test_candidate_config_is_complete_typed_and_evaluator_identified(self) -> None:
        proposal = CandidateProposal("grid", self.workload, {"mode": "eager", "batch": 2, "cache": False})
        expected_id = sha256_hex(
            {
                "provider_id": "grid",
                "workload_hash": self.workload.workload_hash,
                "config": {"mode": "eager", "batch": 2, "cache": False},
            }
        )
        self.assertEqual(proposal.candidate_id, expected_id)
        with self.assertRaises(FrozenInstanceError):
            proposal.candidate_id = "0" * 64  # type: ignore[misc]
        for config in (
            {"mode": "eager", "batch": 2},
            {"mode": "eager", "batch": 2, "cache": False, "extra": 1},
            {"mode": "eager", "batch": True, "cache": False},
            {"mode": "eager", "batch": 2, "cache": False, "candidate_id": "chosen"},
        ):
            with self.subTest(config=config), self.assertRaises(ContractError):
                CandidateProposal("grid", self.workload, config)

    def test_candidate_identity_changes_with_provider_workload_or_config(self) -> None:
        base = CandidateProposal("grid", self.workload, {"mode": "eager", "batch": 2, "cache": False})
        self.assertNotEqual(base.candidate_id, CandidateProposal("other", self.workload, dict(base.config)).candidate_id)
        self.assertNotEqual(
            base.candidate_id,
            CandidateProposal("grid", self.workload, {"mode": "compiled", "batch": 2, "cache": False}).candidate_id,
        )
        changed_workload = FrozenWorkload("toy-v2", knobs=self.workload.knobs, parameters=self.workload.parameters)
        self.assertNotEqual(base.candidate_id, CandidateProposal("grid", changed_workload, dict(base.config)).candidate_id)

    def test_contract_round_trips_are_exact(self) -> None:
        policy = EvaluationPolicy()
        self.assertEqual(EvaluationPolicy.from_dict(policy.to_dict()), policy)
        runtime = RuntimeIdentity("python", "3.11.0", "Darwin", "arm64")
        self.assertEqual(RuntimeIdentity.from_dict(runtime.to_dict()), runtime)
        self.assertEqual(FrozenWorkload.from_dict(self.workload.to_dict()), self.workload)

    def test_surrogates_and_excessive_parameter_nesting_fail_during_contract_construction(self) -> None:
        with self.assertRaises(ContractError) as context:
            FrozenWorkload("toy", parameters={"bad": "\ud800"})
        self.assertEqual(context.exception.code, FailureCode.INVALID_UNICODE)

        nested: object = "leaf"
        for _ in range(MAX_JSON_DEPTH + 2):
            nested = [nested]
        with self.assertRaises(ContractError) as context:
            FrozenWorkload("toy", parameters={"nested": nested})
        self.assertEqual(context.exception.code, FailureCode.JSON_TOO_DEEP)

    def test_public_workload_mapping_ingress_rejects_surrogate_keys_without_raw_value_errors(self) -> None:
        data = self.workload.to_dict()
        data["parameters"] = {"bad\ud800": 1}
        with self.assertRaises(CanonicalJSONError) as context:
            FrozenWorkload.from_dict(data)
        self.assertEqual(context.exception.code, FailureCode.INVALID_UNICODE)
        self.assertNotIn("\ud800", str(context.exception))

    def test_public_workload_mapping_ingress_applies_document_depth(self) -> None:
        nested: object = "leaf"
        for _ in range(MAX_JSON_DEPTH - 1):
            nested = [nested]
        workload = FrozenWorkload("deep", parameters={"nested": nested})
        with self.assertRaises(CanonicalJSONError) as context:
            FrozenWorkload.from_dict(workload.to_dict())
        self.assertEqual(context.exception.code, FailureCode.JSON_TOO_DEEP)

    def test_schema_string_surfaces_reject_surrogates_independently_of_jsonschema(self) -> None:
        cases = (
            lambda: Knob("mode\ud800", "enum", values=("ok",)),
            lambda: Knob("mode", "enum", values=("bad\ud800",)),
            lambda: FrozenWorkload("toy\ud800"),
            lambda: FrozenWorkload("toy", parameters={"bad\ud800": "ok"}),
            lambda: FrozenWorkload("toy", parameters={"ok": "bad\ud800"}),
            lambda: CandidateProposal("grid\ud800", self.workload, {"mode": "eager", "batch": 2, "cache": False}),
            lambda: CandidateProposal("grid", self.workload, {"mode\ud800": "eager", "batch": 2, "cache": False}),
            lambda: CandidateProposal("grid", self.workload, {"mode": "bad\ud800", "batch": 2, "cache": False}),
            lambda: RuntimeIdentity("python\ud800", "3.11", "Darwin", "arm64"),
        )
        for build in cases:
            with self.subTest(build=build), self.assertRaises(ContractError) as context:
                build()
            self.assertEqual(context.exception.code, FailureCode.INVALID_UNICODE)
            self.assertNotIn("\ud800", str(context.exception))

        with self.assertRaises(ContractError) as context:
            Artifact("model\ud800.bin", "0" * 64, 0)
        self.assertEqual(context.exception.code, FailureCode.UNSAFE_PATH)
        self.assertNotIn("\ud800", str(context.exception))

    def test_frozen_workload_schema_documents_the_loader_depth_gate(self) -> None:
        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "frozen_workload.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertIn("MAX_JSON_DEPTH=64", schema["$comment"])

        nested: object = "leaf"
        for _ in range(MAX_JSON_DEPTH + 1):
            nested = [nested]
        with self.assertRaises(ContractError) as context:
            FrozenWorkload("toy", parameters={"nested": nested})
        self.assertEqual(context.exception.code, FailureCode.JSON_TOO_DEEP)

    def test_policy_and_config_bounds_are_explicit_and_stable(self) -> None:
        for kwargs in (
            {"warmup_runs": MAX_WARMUP_RUNS + 1},
            {"measurement_runs": MAX_MEASUREMENT_RUNS + 1},
            {"max_output_bytes": MAX_POLICY_OUTPUT_BYTES + 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ContractError) as context:
                EvaluationPolicy(**kwargs)
            self.assertEqual(context.exception.code, FailureCode.INVALID_POLICY)
        with self.assertRaises(ContractError) as context:
            validate_config(self.workload, {f"extra{i}": i for i in range(MAX_CONFIG_ENTRIES + 1)})
        self.assertEqual(context.exception.code, FailureCode.CONFIG_MISMATCH)

    def test_workload_knob_count_matches_config_capacity_at_the_boundary(self) -> None:
        knobs = tuple(Knob(f"knob{i}", "bool") for i in range(MAX_CONFIG_ENTRIES))
        workload = FrozenWorkload("max-knobs", knobs=knobs)
        proposal = CandidateProposal("grid", workload, {knob.name: False for knob in knobs})
        self.assertEqual(len(proposal.config), MAX_CONFIG_ENTRIES)

        with self.assertRaises(ContractError) as context:
            FrozenWorkload("too-many-knobs", knobs=knobs + (Knob("overflow", "bool"),))
        self.assertEqual(context.exception.code, FailureCode.CONFIG_MISMATCH)
