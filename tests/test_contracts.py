from __future__ import annotations

import copy
import sys
import tomllib
from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx import Artifact, CandidateProposal, EvaluationPolicy, FrozenWorkload, Knob, RuntimeIdentity, sha256_hex
from auto_mlx.errors import ContractError, Failure, FailureCode, UnknownFieldError, UnsafePathError


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
