"""The Wave C ``racing`` relaxation must not weaken the Wave B stop rule.

Wave B guarantees that an ``inconclusive`` verdict is only a legitimate
stopping point at the sequential-sampling cap: anything less means
sequential extension should have kept going, and a receipt claiming
otherwise is rejected.

Wave C's racing search needs a narrow exception -- a candidate whose CI
upper bound has already fallen below the min-effect threshold can never
become a winner, so spending the remaining budget confirming a foregone
conclusion is waste.  That exception is gated on ``EvaluationPolicy.racing``
AND on an independently recomputed CI bound.

These tests pin the property that matters: the exception is scoped to
racing policies only, so an ordinary ``evaluate`` receipt (racing=False,
the default) is still held to the strict Wave B rule.  Identical
measurement evidence, differing only by the policy flag, must validate
in one case and be refused in the other.
"""

from __future__ import annotations

import unittest

from auto_mlx.contracts import CandidateProposal, EvaluationPolicy, FrozenWorkload, Knob, RuntimeIdentity
from auto_mlx.errors import ContractError
from auto_mlx.receipts import Receipt
from auto_mlx.statistics import VERDICT_INCONCLUSIVE

from _wave_b_fixtures import build_evaluator_bundle_receipt


def _workload() -> FrozenWorkload:
    return FrozenWorkload("racing-integrity", knobs=(Knob("mode", "enum", values=("eager", "compiled")),))


def _policy(*, racing: bool) -> EvaluationPolicy:
    # measurement_runs < max_measurement_runs, so stopping at
    # measurement_runs is an early stop, not the cap.
    return EvaluationPolicy(measurement_runs=3, max_measurement_runs=20, racing=racing)


def _receipt(*, racing: bool) -> Receipt:
    workload = _workload()
    policy = _policy(racing=racing)
    candidate = CandidateProposal("racing-integrity-provider", workload, {"mode": "compiled"})
    return build_evaluator_bundle_receipt(
        workload,
        candidate,
        policy,
        RuntimeIdentity.current(),
        # Identical per-arm timings: no measurable difference, so the
        # verdict is inconclusive and the CI collapses well below the
        # min-effect threshold -- i.e. genuinely eliminable.
        baseline_iteration_ns=20_000_000,
        candidate_iteration_ns=20_000_000,
    )


class RacingRelaxationIsScopedTests(unittest.TestCase):
    def test_racing_policy_may_stop_early_when_the_candidate_cannot_win(self) -> None:
        receipt = _receipt(racing=True)
        statistics = receipt.to_dict()["evaluator_bundle"]["statistics"]
        self.assertEqual(statistics["verdict"], VERDICT_INCONCLUSIVE)
        self.assertLess(statistics["ci_upper_ns"], statistics["min_effect_ns"])
        self.assertLess(statistics["block_count_used"], 20)
        # Round-trips through independent re-validation.
        self.assertEqual(Receipt.from_dict(receipt.to_dict()).receipt_id, receipt.receipt_id)

    def test_ordinary_evaluate_receipt_still_cannot_stop_early_on_inconclusive(self) -> None:
        # Same evidence, same early stop, racing=False (the default for
        # every ordinary `auto-mlx evaluate` run).  The Wave B rule must
        # still refuse it.
        with self.assertRaises(ContractError) as caught:
            _receipt(racing=False)
        self.assertIn("inconclusive", str(caught.exception))

    def test_racing_defaults_off_so_the_relaxation_is_opt_in(self) -> None:
        self.assertFalse(EvaluationPolicy().racing)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
