"""Invariant tests for the pre-race advisor (auto_mlx.advisor).

Each test pins one of the advice contract's non-negotiables:

- a prior decisive regression at the exact identity prunes -- WITH a receipt
  citation that still independently loads from the store;
- ``inconclusive``/futile history never prunes, it only demotes order;
- a mismatched runtime identity structurally finds no history at all;
- a prior winner is only *seeded* (ordering), never promoted;
- an uncitable or unattested shortcut is refused and the candidate races;
- contradictory history disables the shortcut and surfaces the conflict.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx.contracts import CandidateProposal, EvaluationPolicy, FrozenWorkload, Knob, RuntimeIdentity
from auto_mlx.receipts import ContentAddressedStore, Receipt
import auto_mlx.advisor as advisor
import auto_mlx.tune as tune

from test_tune import _build_bundle, _toy_workload


_POLICY = EvaluationPolicy(warmup_runs=1, measurement_runs=3, max_measurement_runs=8, k_repetitions=1, bootstrap_resamples=200)


class AdvisorFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.workload = _toy_workload()
        self.runtime = RuntimeIdentity("python", "3.11.0", "Darwin", "arm64")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ContentAddressedStore(str(Path(self.temp.name).resolve()))

    def _store_receipt(self, bundle, candidate, policy) -> tuple[str, bool]:
        receipt = Receipt.from_observation_bundle(bundle, self.workload, candidate, policy, oracle=bundle.oracle, created_at_ns=1)
        self.store.put_receipt(receipt, require_durable=True)
        return receipt.receipt_id, True

    def _race_and_store(self, candidates, run_rung) -> tune.TuningSummary:
        outcome = tune.race_candidates(
            candidates=candidates, base_policy=_POLICY, run_rung=run_rung, store_receipt=self._store_receipt
        )
        summary = tune.build_tuning_summary(
            workload_hash=self.workload.workload_hash,
            runtime=self.runtime,
            provider_id="grid",
            base_policy=_POLICY,
            considered=len(candidates),
            pruned=(),
            max_candidates=None,
            max_candidates_dropped=0,
            outcome=outcome,
            created_at_ns=1,
        )
        self.store.put_tuning_summary(summary, require_durable=True)
        self.store.append_tuning_history(self.workload.workload_hash, self.runtime.identity, summary.summary_id, require_durable=True)
        return summary

    def _advise(self, candidates, *, runtime_identity: str | None = None) -> advisor.RaceAdvice:
        return advisor.advise_candidates(
            candidates,
            store=self.store,
            workload_hash=self.workload.workload_hash,
            runtime_identity=self.runtime.identity if runtime_identity is None else runtime_identity,
        )

    def _regressed_rung(self, candidate, policy):
        return _build_bundle(self.workload, candidate, policy, self.runtime, baseline_ns=10_000_000, candidate_ns=20_000_000)

    def _improved_rung(self, candidate, policy):
        return _build_bundle(self.workload, candidate, policy, self.runtime, baseline_ns=20_000_000, candidate_ns=10_000_000)

    def _futile_rung(self, candidate, policy):
        return _build_bundle(self.workload, candidate, policy, self.runtime, baseline_ns=20_000_000, candidate_ns=20_000_000)


class RegressionPruneTests(AdvisorFixture):
    def test_prior_decisive_regression_is_pruned_with_receipt_citation(self) -> None:
        slower = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})
        summary = self._race_and_store([slower], self._regressed_rung)
        stored_entrant = summary.entrants[0]
        self.assertEqual(stored_entrant["status"], tune.STATUS_REGRESSED)

        advice = self._advise([CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})])
        self.assertEqual(len(advice.prunes), 1)
        prune = advice.prunes[0]
        self.assertEqual(prune.reason, advisor.PRUNE_REASON)
        self.assertEqual(prune.receipt_id, stored_entrant["receipt_id"])
        self.assertEqual(prune.summary_id, summary.summary_id)
        # The cited receipt still independently loads from the store.
        self.assertEqual(self.store.get_receipt(prune.receipt_id).receipt_id, prune.receipt_id)

    def test_apply_advice_removes_pruned_candidate_and_emits_cited_prune_entry(self) -> None:
        slower = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})
        other = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 24})
        summary = self._race_and_store([slower], self._regressed_rung)

        fresh = [
            CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16}),
            other,
        ]
        advice = self._advise(fresh)
        kept, prunes = advisor.apply_advice(fresh, advice)
        self.assertEqual([dict(candidate.config) for candidate in kept], [dict(other.config)])
        self.assertEqual(len(prunes), 1)
        self.assertEqual(prunes[0]["reason"], advisor.PRUNE_REASON)
        self.assertEqual(prunes[0]["summary_id"], summary.summary_id)

        # The cited prune shape is accepted by the tuning-summary contract,
        # so the shortcut's evidence trail survives in the stored record.
        recorded = tune.TuningSummary(
            workload_hash=self.workload.workload_hash,
            runtime=self.runtime,
            provider_id="grid",
            policy=_POLICY,
            created_at_ns=2,
            prefilter={
                "considered": 2,
                "pruned": [dict(entry) for entry in prunes],
                "max_candidates": None,
                "max_candidates_dropped": 0,
                "raced_count": 0,
            },
            budget={
                "budget_measurements": None,
                "budget_seconds": None,
                "blocks_spent": 0,
                "seconds_spent_ns": 0,
                "exhausted": False,
            },
            entrants=[],
            winner=None,
        )
        self.assertEqual(recorded.prefilter["pruned"][0]["receipt_id"], prunes[0]["receipt_id"])

    def test_warm_rerun_spends_fewer_blocks_for_the_same_conclusion(self) -> None:
        # AC1 in miniature: a cold race spends blocks on the known loser; a
        # warm race with advice spends zero blocks on it and reaches the
        # identical conclusion (loser excluded, with citation instead of
        # silence).
        slower = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})
        summary = self._race_and_store([slower], self._regressed_rung)
        cold_blocks = summary.budget["blocks_spent"]
        self.assertGreater(cold_blocks, 0)

        fresh = [CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})]
        advice = self._advise(fresh)
        kept, prunes = advisor.apply_advice(fresh, advice)
        self.assertEqual(kept, ())
        outcome = tune.race_candidates(
            candidates=kept, base_policy=_POLICY, run_rung=self._regressed_rung, store_receipt=self._store_receipt
        )
        self.assertEqual(outcome.blocks_spent, 0)
        self.assertEqual(len(prunes), 1)

    def test_unattested_regression_refuses_the_shortcut(self) -> None:
        slower = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})
        summary = self._race_and_store([slower], self._regressed_rung)
        # Rewrite the stored history to an unattested variant of the same
        # summary (a legal, validated summary -- just without attestation).
        entrant = dict(summary.entrants[0])
        entrant["attested"] = False
        unattested = tune.TuningSummary(
            workload_hash=summary.workload_hash,
            runtime=summary.runtime,
            provider_id=summary.provider_id,
            policy=summary.policy,
            created_at_ns=3,
            prefilter=summary.prefilter,
            budget=summary.budget,
            entrants=[entrant],
            winner=None,
        )
        self.store.put_tuning_summary(unattested, require_durable=True)
        self.store.append_tuning_history(self.workload.workload_hash, self.runtime.identity, unattested.summary_id, require_durable=True)

        advice = self._advise([CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})])
        self.assertEqual(advice.prunes, ())

    def test_missing_receipt_refuses_the_shortcut(self) -> None:
        slower = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})
        summary = self._race_and_store([slower], self._regressed_rung)
        entrant = dict(summary.entrants[0])
        entrant["receipt_id"] = "f" * 64  # valid shape, absent from the store
        rewritten = tune.TuningSummary(
            workload_hash=summary.workload_hash,
            runtime=summary.runtime,
            provider_id=summary.provider_id,
            policy=summary.policy,
            created_at_ns=3,
            prefilter=summary.prefilter,
            budget=summary.budget,
            entrants=[entrant],
            winner=None,
        )
        self.store.put_tuning_summary(rewritten, require_durable=True)
        self.store.append_tuning_history(self.workload.workload_hash, self.runtime.identity, rewritten.summary_id, require_durable=True)

        advice = self._advise([CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})])
        self.assertEqual(advice.prunes, ())


class NeverPromoteNeverPruneInconclusiveTests(AdvisorFixture):
    def test_inconclusive_history_never_prunes_only_demotes(self) -> None:
        borderline = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})
        self._race_and_store([borderline], self._futile_rung)

        fresh_same = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})
        fresh_new = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 24})
        advice = self._advise([fresh_same, fresh_new])
        self.assertEqual(advice.prunes, ())
        self.assertEqual(len(advice.demotions), 1)
        kept, prunes = advisor.apply_advice([fresh_same, fresh_new], advice)
        # Still racing -- demoted to the back, never removed.
        self.assertEqual([dict(c.config) for c in kept], [dict(fresh_new.config), dict(fresh_same.config)])
        self.assertEqual(prunes, ())

    def test_prior_winner_is_seeded_first_and_only_seeded(self) -> None:
        winner = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})
        summary = self._race_and_store([winner], self._improved_rung)
        self.assertIsNotNone(summary.winner)

        fresh_other = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 24})
        fresh_winner = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})
        advice = self._advise([fresh_other, fresh_winner])
        self.assertEqual(len(advice.seeds), 1)
        seed = advice.seeds[0]
        self.assertEqual(seed.reason, advisor.SEED_REASON)
        self.assertEqual(seed.receipt_id, summary.winner["receipt_id"])
        self.assertEqual(seed.summary_id, summary.summary_id)

        kept, prunes = advisor.apply_advice([fresh_other, fresh_winner], advice)
        # Seeded FIRST -- but still present, still racing from the bottom
        # rung: advice offers no way to skip measurement or activate.
        self.assertEqual([dict(c.config) for c in kept], [dict(fresh_winner.config), dict(fresh_other.config)])
        self.assertEqual(prunes, ())
        # The advice vocabulary itself is closed to ordering/pruning: there
        # is no promote/activate action for a graph or history entry to take.
        for entry in advice.entries:
            self.assertIn(entry.action, {advisor.ACTION_PRUNE, advisor.ACTION_SEED, advisor.ACTION_DEMOTE})


class IdentityGatingTests(AdvisorFixture):
    def test_identity_mismatch_finds_no_history_and_no_advice(self) -> None:
        slower = CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})
        self._race_and_store([slower], self._regressed_rung)

        other_runtime = RuntimeIdentity("python", "3.12.0", "Darwin", "arm64")
        self.assertNotEqual(other_runtime.identity, self.runtime.identity)
        advice = self._advise(
            [CandidateProposal("grid", self.workload, {"mode": "compiled", "tile": 16})],
            runtime_identity=other_runtime.identity,
        )
        self.assertEqual(advice.considered_summaries, ())
        self.assertEqual(advice.entries, ())
        self.assertEqual(advice.conflicts, ())


class ConflictTests(AdvisorFixture):
    def test_contradictory_history_disables_prune_and_surfaces_conflict(self) -> None:
        config = {"mode": "compiled", "tile": 16}
        # Older run: decisively improved. Newer run: decisively regressed.
        self._race_and_store([CandidateProposal("grid", self.workload, config)], self._improved_rung)
        self._race_and_store([CandidateProposal("grid", self.workload, config)], self._regressed_rung)

        fresh = CandidateProposal("grid", self.workload, config)
        advice = self._advise([fresh])
        self.assertEqual(advice.prunes, ())
        self.assertEqual(advice.conflicts, (fresh.candidate_id,))
        kept, prunes = advisor.apply_advice([fresh], advice)
        self.assertEqual(kept, (fresh,))
        self.assertEqual(prunes, ())


if __name__ == "__main__":
    unittest.main()
