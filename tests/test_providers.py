from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx import CandidateProvider, DeclarativeProvider, FrozenWorkload, Knob
from auto_mlx.providers import MAX_PROVIDER_CONFIGS
from auto_mlx.errors import ContractError, FailureCode, UnknownFieldError


class ProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workload = FrozenWorkload(
            "provider-test",
            knobs=(Knob("mode", "enum", values=("a", "b")), Knob("threads", "integer", minimum=1, maximum=2)),
        )

    def test_protocol_and_provider_emit_derived_proposals_only(self) -> None:
        provider = DeclarativeProvider("declarative-grid", ({"mode": "a", "threads": 1}, {"mode": "b", "threads": 2}))
        self.assertIsInstance(provider, CandidateProvider)
        proposals = provider.propose(self.workload)
        self.assertEqual([proposal.config["mode"] for proposal in proposals], ["a", "b"])
        self.assertEqual(len({proposal.candidate_id for proposal in proposals}), 2)
        self.assertNotIn("command", provider.to_dict())
        self.assertNotIn("source", provider.to_dict())

    def test_provider_rejects_code_like_values_and_incomplete_configs(self) -> None:
        with self.assertRaises(ContractError):
            DeclarativeProvider("bad", ({"mode": "a", "threads": {"command": "rm -rf"}},))
        provider = DeclarativeProvider("grid", ({"mode": "a"},))
        with self.assertRaises(ContractError):
            provider.propose(self.workload)

    def test_provider_schema_is_strict_and_source_maps_are_frozen(self) -> None:
        configs = [{"mode": "a", "threads": 1}]
        provider = DeclarativeProvider("grid", configs)
        configs[0]["threads"] = 2
        self.assertEqual(provider.configs[0]["threads"], 1)
        with self.assertRaises(UnknownFieldError):
            DeclarativeProvider.from_dict({"provider_id": "grid", "configs": [], "command": "x"})
        with self.assertRaises(ContractError):
            DeclarativeProvider.from_dict({"provider_id": "grid", "configs": [{"mode": 1, "threads": 1.0}]})

    def test_reserved_candidate_id_is_rejected_before_provider_serialization(self) -> None:
        with self.assertRaises(ContractError) as context:
            DeclarativeProvider("bad", ({"candidate_id": "attacker-choice"},))
        self.assertEqual(context.exception.code, FailureCode.CONFIG_MISMATCH)
        with self.assertRaises(ContractError) as context:
            DeclarativeProvider.from_json('{"provider_id":"bad","configs":[{"candidate_id":"attacker-choice"}]}')
        self.assertEqual(context.exception.code, FailureCode.CONFIG_MISMATCH)

    def test_wrong_shaped_provider_fields_fail_as_contract_errors(self) -> None:
        for configs in (None, {"mode": "a"}, "not-an-array"):
            with self.subTest(configs=configs), self.assertRaises(ContractError):
                DeclarativeProvider("bad", configs)  # type: ignore[arg-type]

    def test_duplicate_normalized_configs_are_rejected_deterministically(self) -> None:
        with self.assertRaises(ContractError) as context:
            DeclarativeProvider(
                "grid",
                (
                    {"mode": "a", "threads": 1},
                    {"threads": 1, "mode": "a"},
                ),
            )
        self.assertEqual(context.exception.code, FailureCode.CONFIG_MISMATCH)

    def test_provider_schema_rejects_duplicate_configurations(self) -> None:
        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "declarative_provider.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertTrue(schema["properties"]["configs"]["uniqueItems"])
        self.assertEqual(
            schema["properties"]["configs"]["items"]["additionalProperties"]["oneOf"],
            [{"type": "string"}, {"type": "integer"}, {"type": "boolean"}],
        )

        with self.assertRaises(ContractError) as context:
            DeclarativeProvider.from_dict(
                {
                    "provider_id": "grid",
                    "configs": [{"mode": "a", "threads": 1}, {"threads": 1, "mode": "a"}],
                }
            )
        self.assertEqual(context.exception.code, FailureCode.CONFIG_MISMATCH)

    def test_provider_config_count_is_bounded_before_materialization(self) -> None:
        with self.assertRaises(ContractError) as context:
            DeclarativeProvider("grid", ({"mode": "a", "threads": 1},) * (MAX_PROVIDER_CONFIGS + 1))
        self.assertEqual(context.exception.code, FailureCode.PROVIDER_ERROR)
