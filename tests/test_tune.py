"""Tests for the Wave C search layer (auto_mlx.tune): pre-filter, racing,
budget accounting, warm start, and the content-addressed tuning summary.

These are synthetic/unit tests -- they never need MLX or the local sandbox:
``run_rung`` is a hand-built, genuinely-accepted ``ObservationBundle`` (the
same construction ``tests/_wave_b_fixtures.py`` uses for promotion/dispatch
tests), so racing, elimination, and receipt storage all run through their
real code paths against real ``ContentAddressedStore``/``Receipt`` objects,
with only the underlying MLX subprocess measurement replaced. The real,
sandboxed end-to-end ``auto-mlx tune`` run lives in
tests/test_tune_cli_loop.py.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx.canonical import sha256_hex
from auto_mlx.contracts import CandidateProposal, EvaluationPolicy, FrozenWorkload, Knob, RuntimeIdentity
from auto_mlx.errors import ContractError, FailureCode
from auto_mlx.evaluator import Observation, ObservationBundle, _execution_policy_digest, _execution_policy_from_contract
from auto_mlx.executor import ExecutionRecord, ExecutionStatus
from auto_mlx.measurement import MeasurementSample, PairedMeasurementPlan, assemble_measurement_bundle
from auto_mlx.oracle import ExactOutputOracle
from auto_mlx.providers import DeclarativeProvider
from auto_mlx.receipts import ContentAddressedStore, Receipt
from auto_mlx.statistics import compute_sample_timing, compute_statistics_verdict
import auto_mlx.tune as tune

from _wave_b_fixtures import FixtureIsolationAuthority, FixtureIsolationProvider, _iter_timings_stderr, _nominal_thermal_blocks


def _toy_workload() -> FrozenWorkload:
    return FrozenWorkload(
        "toy-matmul",
        knobs=(
            Knob("mode", "enum", values=("eager", "compiled")),
            Knob("tile", "integer", minimum=16, maximum=32),
        ),
        parameters={"dtype": "float32", "shape": [1, 3072, 3072]},
    )


def _build_bundle(
    workload: FrozenWorkload,
    candidate: CandidateProposal,
    policy: EvaluationPolicy,
    runtime: RuntimeIdentity,
    *,
    baseline_ns: int,
    candidate_ns: int,
    bootstrap_seed: int = 1,
) -> ObservationBundle:
    """A real, accepted ``ObservationBundle`` at exactly ``policy.measurement_runs`` blocks.

    Mirrors ``tests/_wave_b_fixtures.py``'s ``build_evaluator_bundle_receipt``
    but returns the bundle itself (what a racing rung's ``run_rung`` must
    return) rather than the final receipt -- callers build the receipt
    themselves, exactly like ``auto_mlx.cli``'s real ``store_rung_receipt``
    closure does.
    """

    oracle = ExactOutputOracle(b"ok\n", label="tune-fixture-oracle")
    provider = FixtureIsolationProvider()
    authority = FixtureIsolationAuthority()
    plan = PairedMeasurementPlan.create(
        policy.measurement_runs,
        candidate_id=candidate.candidate_id,
        workload_hash=workload.workload_hash,
        baseline_runner_id="baseline",
        baseline_runner_digest="1" * 64,
        candidate_runner_id="candidate",
        candidate_runner_digest="2" * 64,
        oracle=oracle,
        isolation_provider_id=provider.provider_id,
        isolation_identity=provider.identity,
        isolation_verifier_id=authority.verifier_id,
        isolation_verifier_identity=authority.identity,
        isolation_requirements=frozenset({"network_denial", "descendant_containment"}),
    )

    def record(sample_id: str, arm: str) -> ExecutionRecord:
        runner_id = "baseline" if arm == "baseline" else "candidate"
        runner_digest = "1" * 64 if arm == "baseline" else "2" * 64
        iteration_ns = baseline_ns if arm == "baseline" else candidate_ns
        return ExecutionRecord(
            candidate_id=candidate.candidate_id,
            workload_hash=workload.workload_hash,
            runner_id=runner_id,
            runner_digest=runner_digest,
            status=ExecutionStatus.SUCCESS,
            parent_elapsed_ns=iteration_ns + 1_000,
            runner_elapsed_ns=iteration_ns,
            observation_id=sample_id,
            arm=arm,
            returncode=0,
            stdout=b"ok\n",
            stderr=_iter_timings_stderr(iteration_ns),
            isolation=authority._attest(provider, provider._claim("d" * 64)),
        )

    warmups = tuple(
        Observation(f"warmup-{index + 1:04d}-{arm}", arm, record(f"warmup-{index + 1:04d}-{arm}", arm), oracle.evaluate(b"ok\n"))
        for index in range(policy.warmup_runs)
        for arm in ("baseline", "candidate")
    )
    samples = []
    for block in plan.blocks:
        for slot in block.slots:
            current = record(slot.sample_id, slot.arm)
            samples.append(MeasurementSample(slot.sample_id, slot.block_id, slot.slot_index, slot.arm, current, oracle.evaluate(current.stdout)))
    measurements = assemble_measurement_bundle(plan, samples)
    if not measurements.accepted:
        raise AssertionError(f"fixture measurement bundle was not accepted: {measurements.rejection_reasons}")

    baseline_points: list[list[int]] = []
    candidate_points: list[list[int]] = []
    for block in measurements.blocks:
        block_baseline: list[int] = []
        block_candidate: list[int] = []
        for sample in block.samples:
            timing = compute_sample_timing(sample.record.runner_elapsed_ns, sample.record.stderr)
            (block_baseline if sample.arm == "baseline" else block_candidate).append(timing.point_estimate_ns)
        baseline_points.append(block_baseline)
        candidate_points.append(block_candidate)
    verdict = compute_statistics_verdict(
        block_baseline_points=baseline_points,
        block_candidate_points=candidate_points,
        k_repetitions=policy.k_repetitions,
        measurement_runs=policy.measurement_runs,
        max_measurement_runs=policy.max_measurement_runs,
        min_effect_bps=policy.min_effect_bps,
        bootstrap_resamples=policy.bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        calibration=policy.calibration,
    )
    return ObservationBundle(
        candidate_id=candidate.candidate_id,
        workload_hash=workload.workload_hash,
        runtime=runtime,
        baseline_runner_id="baseline",
        baseline_runner_digest="1" * 64,
        candidate_runner_id="candidate",
        candidate_runner_digest="2" * 64,
        isolation_provider_id=provider.provider_id,
        isolation_identity=provider.identity,
        isolation_verifier_id=authority.verifier_id,
        isolation_verifier_identity=authority.identity,
        isolation_requirements=frozenset({"network_denial", "descendant_containment"}),
        warmups=warmups,
        measurements=measurements,
        thermal_blocks=_nominal_thermal_blocks(plan, policy=policy),
        policy_digest=sha256_hex(policy.to_dict()),
        execution_policy_digest=_execution_policy_digest(_execution_policy_from_contract(policy)),
        measurement_block_count=policy.measurement_runs,
        evaluation_policy=policy,
        execution_policy=_execution_policy_from_contract(policy),
        oracle=oracle,
        oracle_descriptor=oracle.descriptor,
        statistics=verdict.to_dict(),
    )


class PrefilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workload = _toy_workload()

    def test_full_grid_enumeration_is_all_legal(self) -> None:
        configs = [{"mode": mode, "tile": tile} for mode in ("eager", "compiled") for tile in range(16, 33)]
        provider = DeclarativeProvider("grid", tuple(configs))
        legal, pruned = tune.prefilter_candidates(provider, self.workload)
        self.assertEqual(len(legal), 34)
        self.assertEqual(pruned, ())

    def test_out_of_range_tile_is_pruned(self) -> None:
        provider = DeclarativeProvider(
            "grid",
            ({"mode": "eager", "tile": 16}, {"mode": "eager", "tile": 99}),
        )
        legal, pruned = tune.prefilter_candidates(provider, self.workload)
        self.assertEqual(len(legal), 1)
        self.assertEqual(len(pruned), 1)
        self.assertEqual(pruned[0]["config"], {"mode": "eager", "tile": 99})
        self.assertEqual(pruned[0]["reason"], FailureCode.CONFIG_MISMATCH.value)

    def test_unknown_knob_name_is_pruned(self) -> None:
        provider = DeclarativeProvider("grid", ({"mode": "eager", "bogus": 1},))
        legal, pruned = tune.prefilter_candidates(provider, self.workload)
        self.assertEqual(legal, ())
        self.assertEqual(len(pruned), 1)
        self.assertEqual(pruned[0]["reason"], FailureCode.CONFIG_MISMATCH.value)

    def test_wrong_type_is_pruned(self) -> None:
        provider = DeclarativeProvider("grid", ({"mode": "eager", "tile": "sixteen"},))
        legal, pruned = tune.prefilter_candidates(provider, self.workload)
        self.assertEqual(legal, ())
        self.assertEqual(len(pruned), 1)


class MaxCandidatesTests(unittest.TestCase):
    def test_none_keeps_everything(self) -> None:
        workload = _toy_workload()
        candidates = tuple(CandidateProposal("p", workload, {"mode": "eager", "tile": t}) for t in (16, 17, 18))
        kept, dropped = tune.apply_max_candidates(candidates, None)
        self.assertEqual(kept, candidates)
        self.assertEqual(dropped, 0)

    def test_truncates_in_order_and_reports_dropped_count(self) -> None:
        workload = _toy_workload()
        candidates = tuple(CandidateProposal("p", workload, {"mode": "eager", "tile": t}) for t in (16, 17, 18))
        kept, dropped = tune.apply_max_candidates(candidates, 2)
        self.assertEqual(kept, candidates[:2])
        self.assertEqual(dropped, 1)


class RacingEliminationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workload = _toy_workload()
        self.runtime = RuntimeIdentity("python", "3.11.0", "Darwin", "arm64")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ContentAddressedStore(str(Path(self.temp.name).resolve()))

    def _store_receipt(self, bundle: ObservationBundle, candidate: CandidateProposal, policy: EvaluationPolicy) -> tuple[str, bool]:
        receipt = Receipt.from_observation_bundle(bundle, self.workload, candidate, policy, oracle=bundle.oracle, created_at_ns=1)
        self.store.put_receipt(receipt, require_durable=True)
        return receipt.receipt_id, True

    def test_clearly_inferior_candidate_is_eliminated_before_the_block_cap(self) -> None:
        # A candidate identical to baseline (zero true effect): its CI is a
        # zero-width point at 0ns, strictly below the (positive) min-effect
        # threshold -- futile from the very first rung.
        winner = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})
        loser = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 24})

        def run_rung(candidate: CandidateProposal, policy: EvaluationPolicy) -> ObservationBundle:
            if candidate.candidate_id == winner.candidate_id:
                return _build_bundle(self.workload, candidate, policy, self.runtime, baseline_ns=20_000_000, candidate_ns=10_000_000)
            return _build_bundle(self.workload, candidate, policy, self.runtime, baseline_ns=20_000_000, candidate_ns=20_000_000)

        base_policy = EvaluationPolicy(warmup_runs=1, measurement_runs=3, max_measurement_runs=8, k_repetitions=1, bootstrap_resamples=200)
        outcome = tune.race_candidates(
            candidates=[winner, loser], base_policy=base_policy, run_rung=run_rung, store_receipt=self._store_receipt
        )
        entrants = {entrant.candidate.candidate_id: entrant for entrant in outcome.entrants}

        loser_entrant = entrants[loser.candidate_id]
        self.assertEqual(loser_entrant.status, tune.STATUS_ELIMINATED_FUTILE)
        # Eliminated at the MINIMUM block count -- never ran anywhere near
        # the 8-block cap. This is racing's whole point.
        self.assertEqual(loser_entrant.block_count_used, base_policy.measurement_runs)
        self.assertLess(loser_entrant.block_count_used, base_policy.max_measurement_runs)
        self.assertIsNotNone(loser_entrant.receipt_id)
        stored = self.store.get_receipt(loser_entrant.receipt_id)
        self.assertEqual(stored.status, "complete")

        winner_entrant = entrants[winner.candidate_id]
        self.assertEqual(winner_entrant.status, tune.STATUS_IMPROVED)
        self.assertIsNotNone(outcome.incumbent)
        self.assertEqual(outcome.incumbent.candidate.candidate_id, winner.candidate_id)

    def test_regressed_candidate_resolves_decisively_without_reaching_the_cap(self) -> None:
        slower = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})

        def run_rung(candidate: CandidateProposal, policy: EvaluationPolicy) -> ObservationBundle:
            return _build_bundle(self.workload, candidate, policy, self.runtime, baseline_ns=10_000_000, candidate_ns=20_000_000)

        base_policy = EvaluationPolicy(warmup_runs=1, measurement_runs=3, max_measurement_runs=10, k_repetitions=1, bootstrap_resamples=200)
        outcome = tune.race_candidates(candidates=[slower], base_policy=base_policy, run_rung=run_rung, store_receipt=self._store_receipt)
        self.assertEqual(len(outcome.entrants), 1)
        self.assertEqual(outcome.entrants[0].status, tune.STATUS_REGRESSED)
        self.assertEqual(outcome.entrants[0].block_count_used, 3)
        self.assertIsNone(outcome.incumbent)

    def test_genuinely_ambiguous_candidate_extends_to_the_cap_and_reports_inconclusive(self) -> None:
        # Effect just under the noise this fixture's zero-dispersion samples
        # can express: use a tiny nonzero delta straddling the threshold
        # band so it is neither clearly futile nor decisive at any rung --
        # it must legitimately ride out every rung to the declared cap.
        borderline = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 20})

        def run_rung(candidate: CandidateProposal, policy: EvaluationPolicy) -> ObservationBundle:
            # baseline 20_000_000ns, min_effect_bps default 200 (2%) -> 400_000ns.
            # candidate 19_700_000ns -> point delta 300_000ns: below the
            # improvement threshold but with ci_upper (a zero-width point
            # here) sitting ABOVE it once rounded -- keep it deterministic
            # and just above the futility line so it never eliminates.
            return _build_bundle(self.workload, candidate, policy, self.runtime, baseline_ns=20_000_000, candidate_ns=19_600_000)

        base_policy = EvaluationPolicy(warmup_runs=1, measurement_runs=3, max_measurement_runs=5, k_repetitions=1, bootstrap_resamples=200)
        outcome = tune.race_candidates(candidates=[borderline], base_policy=base_policy, run_rung=run_rung, store_receipt=self._store_receipt)
        self.assertEqual(len(outcome.entrants), 1)
        entrant = outcome.entrants[0]
        self.assertEqual(entrant.status, tune.STATUS_INCONCLUSIVE_AT_CAP)
        self.assertEqual(entrant.block_count_used, base_policy.max_measurement_runs)


class BudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workload = _toy_workload()
        self.runtime = RuntimeIdentity("python", "3.11.0", "Darwin", "arm64")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ContentAddressedStore(str(Path(self.temp.name).resolve()))

    def _store_receipt(self, bundle: ObservationBundle, candidate: CandidateProposal, policy: EvaluationPolicy) -> tuple[str, bool]:
        receipt = Receipt.from_observation_bundle(bundle, self.workload, candidate, policy, oracle=bundle.oracle, created_at_ns=1)
        self.store.put_receipt(receipt, require_durable=True)
        return receipt.receipt_id, True

    def test_budget_measurements_exhaustion_is_reported_honestly(self) -> None:
        c1 = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})
        c2 = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 17})

        def run_rung(candidate: CandidateProposal, policy: EvaluationPolicy) -> ObservationBundle:
            # Ambiguous-forever: never decisive, never futile, so the only
            # way either candidate resolves is by budget or cap.
            return _build_bundle(self.workload, candidate, policy, self.runtime, baseline_ns=20_000_000, candidate_ns=19_600_000)

        base_policy = EvaluationPolicy(warmup_runs=1, measurement_runs=3, max_measurement_runs=20, k_repetitions=1, bootstrap_resamples=200)
        # Budget affords exactly one rung of 3 blocks -- c1's first rung, and
        # nothing else.
        outcome = tune.race_candidates(
            candidates=[c1, c2], base_policy=base_policy, run_rung=run_rung, store_receipt=self._store_receipt, budget_measurements=3,
        )
        self.assertTrue(outcome.budget_exhausted)
        self.assertEqual(outcome.blocks_spent, 3)
        entrants = {entrant.candidate.candidate_id: entrant for entrant in outcome.entrants}
        self.assertEqual(entrants[c1.candidate_id].status, tune.STATUS_UNRESOLVED_BUDGET)
        self.assertEqual(entrants[c1.candidate_id].block_count_used, 3)
        self.assertEqual(entrants[c1.candidate_id].reason, "budget_exhausted")
        self.assertEqual(entrants[c2.candidate_id].status, tune.STATUS_UNRESOLVED_BUDGET)
        self.assertIsNone(entrants[c2.candidate_id].block_count_used)
        self.assertEqual(entrants[c2.candidate_id].reason, "budget_exhausted_before_first_rung")

    def test_budget_seconds_exhaustion_stops_before_any_rung(self) -> None:
        c1 = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})
        called = {"count": 0}

        def run_rung(candidate: CandidateProposal, policy: EvaluationPolicy) -> ObservationBundle:
            called["count"] += 1
            return _build_bundle(self.workload, candidate, policy, self.runtime, baseline_ns=20_000_000, candidate_ns=19_600_000)

        base_policy = EvaluationPolicy(warmup_runs=1, measurement_runs=3, max_measurement_runs=20, k_repetitions=1, bootstrap_resamples=200)
        outcome = tune.race_candidates(
            candidates=[c1], base_policy=base_policy, run_rung=run_rung, store_receipt=self._store_receipt, budget_seconds=0,
        )
        self.assertTrue(outcome.budget_exhausted)
        self.assertEqual(called["count"], 0)
        self.assertEqual(outcome.entrants[0].status, tune.STATUS_UNRESOLVED_BUDGET)
        self.assertEqual(outcome.entrants[0].reason, "budget_exhausted_before_first_rung")

    def test_baseline_floor_is_always_present_in_the_summary_regardless_of_race_outcome(self) -> None:
        c1 = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})

        def run_rung(candidate: CandidateProposal, policy: EvaluationPolicy) -> ObservationBundle:
            return _build_bundle(self.workload, candidate, policy, self.runtime, baseline_ns=10_000_000, candidate_ns=20_000_000)

        base_policy = EvaluationPolicy(warmup_runs=1, measurement_runs=3, max_measurement_runs=6, k_repetitions=1, bootstrap_resamples=200)
        outcome = tune.race_candidates(candidates=[c1], base_policy=base_policy, run_rung=run_rung, store_receipt=self._store_receipt)
        summary = tune.build_tuning_summary(
            workload_hash=self.workload.workload_hash, runtime=self.runtime, provider_id="grid", base_policy=base_policy,
            considered=1, pruned=(), max_candidates=None, max_candidates_dropped=0, outcome=outcome,
        )
        self.assertEqual(summary.baseline["status"], "floor")
        self.assertIsNone(summary.winner)


class WarmStartTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workload = _toy_workload()
        self.runtime = RuntimeIdentity("python", "3.11.0", "Darwin", "arm64")
        self.other_runtime = RuntimeIdentity("python", "3.11.0", "Darwin", "x86_64")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ContentAddressedStore(str(Path(self.temp.name).resolve()))
        self.winner_config = {"mode": "compiled", "tile": 24}

    def _seed_prior_winner(self, runtime: RuntimeIdentity) -> None:
        winner_candidate = CandidateProposal("grid", self.workload, self.winner_config)
        entrant = tune.RaceEntrant(
            winner_candidate, tune.STATUS_IMPROVED, 3, "a" * 64, True,
            {
                "schema": "auto_mlx.statistics.v1", "k_repetitions": 1, "block_count_used": 3, "verdict": "improved",
                "ci_lower_ns": 500_000, "ci_upper_ns": 600_000, "min_effect_ns": 400_000, "baseline_reference_ns": 20_000_000,
                "bootstrap_seed": 1, "bootstrap_resamples": 200, "confidence_bps": 9500, "calibration": False,
            },
            "decisive_improved",
        )
        outcome = tune.RaceOutcome(
            entrants=(entrant,), incumbent=entrant, blocks_spent=3, seconds_spent_ns=1, budget_measurements=None,
            budget_seconds=None, budget_exhausted=False,
        )
        summary = tune.build_tuning_summary(
            workload_hash=self.workload.workload_hash, runtime=runtime, provider_id="grid",
            base_policy=EvaluationPolicy(measurement_runs=3, max_measurement_runs=6), considered=1, pruned=(),
            max_candidates=None, max_candidates_dropped=0, outcome=outcome,
        )
        self.store.put_tuning_summary(summary, require_durable=True)
        self.store.append_tuning_history(self.workload.workload_hash, runtime.identity, summary.summary_id, require_durable=True)

    def test_prior_winner_is_moved_to_the_front(self) -> None:
        self._seed_prior_winner(self.runtime)
        candidates = tuple(
            CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": t}) for t in (16, 17, 24, 30)
        )
        ordered = tune.warm_start_order(
            candidates, store=self.store, workload_hash=self.workload.workload_hash, runtime_identity=self.runtime.identity
        )
        self.assertEqual(dict(ordered[0].config), self.winner_config)
        self.assertEqual({c.candidate_id for c in ordered}, {c.candidate_id for c in candidates})

    def test_mismatched_runtime_identity_is_ignored(self) -> None:
        self._seed_prior_winner(self.other_runtime)
        candidates = tuple(
            CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": t}) for t in (16, 17, 24, 30)
        )
        ordered = tune.warm_start_order(
            candidates, store=self.store, workload_hash=self.workload.workload_hash, runtime_identity=self.runtime.identity
        )
        self.assertEqual(ordered, candidates)

    def test_no_history_leaves_order_untouched(self) -> None:
        candidates = tuple(CandidateProposal("grid", self.workload, {"mode": "eager", "tile": t}) for t in (16, 17))
        ordered = tune.warm_start_order(
            candidates, store=self.store, workload_hash=self.workload.workload_hash, runtime_identity=self.runtime.identity
        )
        self.assertEqual(ordered, candidates)


class TuningSummaryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workload = _toy_workload()
        self.runtime = RuntimeIdentity("python", "3.11.0", "Darwin", "arm64")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ContentAddressedStore(str(Path(self.temp.name).resolve()))

    def _summary(self) -> tune.TuningSummary:
        outcome = tune.RaceOutcome(
            entrants=(), incumbent=None, blocks_spent=0, seconds_spent_ns=0, budget_measurements=None,
            budget_seconds=None, budget_exhausted=False,
        )
        return tune.build_tuning_summary(
            workload_hash=self.workload.workload_hash, runtime=self.runtime, provider_id="grid",
            base_policy=EvaluationPolicy(), considered=0, pruned=(), max_candidates=None, max_candidates_dropped=0,
            outcome=outcome,
        )

    def test_content_addressed_round_trip(self) -> None:
        summary = self._summary()
        self.store.put_tuning_summary(summary, require_durable=True)
        reread = self.store.get_tuning_summary(summary.summary_id)
        self.assertEqual(reread["summary_id"], summary.summary_id)
        self.assertEqual(tune.TuningSummary.from_dict(reread).summary_id, summary.summary_id)

    def test_tampered_summary_on_disk_is_rejected_on_read(self) -> None:
        summary = self._summary()
        path = self.store.put_tuning_summary(summary, require_durable=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["provider_id"] = "tampered"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ContractError):
            self.store.get_tuning_summary(summary.summary_id)

    def test_missing_summary_is_a_typed_error(self) -> None:
        with self.assertRaises(ContractError):
            self.store.get_tuning_summary("0" * 64)

    def test_history_index_append_and_list(self) -> None:
        first = self._summary()
        self.store.put_tuning_summary(first, require_durable=True)
        self.store.append_tuning_history(self.workload.workload_hash, self.runtime.identity, first.summary_id, require_durable=True)
        self.assertEqual(
            self.store.list_tuning_history(self.workload.workload_hash, self.runtime.identity), (first.summary_id,)
        )
        # Appending the same id again does not duplicate it.
        self.store.append_tuning_history(self.workload.workload_hash, self.runtime.identity, first.summary_id, require_durable=True)
        self.assertEqual(
            self.store.list_tuning_history(self.workload.workload_hash, self.runtime.identity), (first.summary_id,)
        )

    def test_unknown_workload_runtime_pair_has_empty_history(self) -> None:
        self.assertEqual(self.store.list_tuning_history("a" * 64, "b" * 64), ())


if __name__ == "__main__":
    unittest.main()
