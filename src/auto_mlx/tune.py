"""Wave C search layer: candidate racing, tuning summaries, and warm start.

Trust model, unchanged from Waves A/B: the evaluator owns execution and all
evidentiary timing; receipts are immutable and attested; a calibration
receipt is never promotable; a wrong-output candidate never enters timing
comparison. This module adds no new execution surface of its own -- it only
orchestrates repeated, real ``auto_mlx.evaluator.Evaluator`` runs (injected
via ``run_rung``) and stores their resulting receipts through the existing
pipeline (injected via ``store_receipt``). No candidate here ever selects a
command or code: candidates only ever come from a
:class:`auto_mlx.providers.DeclarativeProvider`'s closed, scalar-only
config space.

Racing design
-------------

Baseline is a permanent, unremovable entrant: it is what every candidate is
measured against (each racing "rung" is an ordinary paired baseline-vs-
candidate Wave B evaluation), and the race can always honestly conclude
"keep baseline" -- this is reported explicitly, never silently.

Each surviving candidate is measured in a lockstep ladder of rungs, one
additional block at a time, starting at ``policy.measurement_runs`` and
capped at ``policy.max_measurement_runs``. Each rung is a *self-contained*
:class:`auto_mlx.contracts.EvaluationPolicy` with ``measurement_runs ==
max_measurement_runs == <rung's block count>`` -- i.e. Wave B's own
internal sequential extension is pinned off for that one call, so its
verdict (decisive or inconclusive) is always legitimate evidence at
*exactly* that block count, with no change to the existing receipt schema.
A rung re-measures its candidate from block 1 (Evaluator.evaluate has no
cross-call state), trading a bounded amount of redundant measurement for a
zero-touch reuse of the fully audited Wave A/B evidence machinery.

After the minimum ``policy.measurement_runs`` blocks, and re-checked after
every additional rung, a candidate is dropped the moment its CI upper bound
falls below the min-effect threshold (it cannot statistically become a
winner) -- never on a single noisy delta, always on a real bootstrap CI. A
candidate only ever becomes the race's winner if it reaches a decisive
``improved`` verdict *and*, when replacing a prior incumbent, does so at a
block depth at least as deep as the incumbent's own (SMAC intensification
discipline) with a strictly better guaranteed lower bound.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from .canonical import canonical_json, sha256_hex, strict_json_loads, validate_json_value
from .contracts import CandidateProposal, EvaluationPolicy, FrozenWorkload, RuntimeIdentity
from .errors import ContractError, FailureCode, UnknownFieldError
from .evaluator import ObservationBundle
from .paths import validate_sha256
from .providers import DeclarativeProvider
from .receipts import ContentAddressedStore
from .statistics import VERDICT_IMPROVED, VERDICT_REGRESSED


TUNING_SUMMARY_SCHEMA: Final = "auto_mlx.tuning_summary.v1"

STATUS_IMPROVED: Final = "improved"
STATUS_REGRESSED: Final = "regressed"
STATUS_INCONCLUSIVE_AT_CAP: Final = "inconclusive_at_cap"
STATUS_ELIMINATED_FUTILE: Final = "eliminated_futile"
STATUS_FAILED: Final = "failed"
STATUS_UNRESOLVED_BUDGET: Final = "unresolved_budget"
ENTRANT_STATUSES: Final = frozenset(
    {
        STATUS_IMPROVED,
        STATUS_REGRESSED,
        STATUS_INCONCLUSIVE_AT_CAP,
        STATUS_ELIMINATED_FUTILE,
        STATUS_FAILED,
        STATUS_UNRESOLVED_BUDGET,
    }
)

BASELINE_FLOOR_NOTE: Final = (
    "baseline is a permanent, unremovable entrant; the race concludes 'keep baseline' "
    "unless a candidate reaches a decisive improved verdict"
)


def _object(value: Any, *, label: str) -> dict[str, Any]:
    validate_json_value(value)
    if type(value) is not dict:
        raise ContractError(f"{label} must be a JSON object", code=FailureCode.WRONG_TYPE)
    return value


def _exact(value: dict[str, Any], expected: set[str], *, label: str) -> None:
    if any(type(key) is not str for key in value):
        raise ContractError(f"{label} field names must be strings", code=FailureCode.WRONG_TYPE)
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise UnknownFieldError(f"{label} has unknown field(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ContractError(f"{label} is missing field(s): {', '.join(sorted(missing))}", code=FailureCode.INVALID_VALUE)


def _string(value: Any, *, label: str, non_empty: bool = True) -> str:
    if type(value) is not str or (non_empty and not value):
        raise ContractError(f"{label} must be a {'non-empty ' if non_empty else ''}string", code=FailureCode.WRONG_TYPE)
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ContractError(f"{label} must not contain unpaired surrogates", code=FailureCode.INVALID_UNICODE)
    return value


def _integer(value: Any, *, label: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise ContractError(f"{label} must be an integer{suffix}", code=FailureCode.WRONG_TYPE)
    return value


def _boolean(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise ContractError(f"{label} must be a boolean", code=FailureCode.WRONG_TYPE)
    return value


def _config_dict(value: Any, *, label: str) -> dict[str, Any]:
    data = _object(value, label=label)
    for key, item in data.items():
        if type(item) not in {str, int, bool}:
            raise ContractError(f"{label} values must be strings, integers, or booleans", code=FailureCode.WRONG_TYPE)
    return data


# ---------------------------------------------------------------------------
# Pre-filter: reject configs the workload's own knob contract excludes,
# before spending any measurement (MetaSchedule postproc analog).
# ---------------------------------------------------------------------------


def prefilter_candidates(
    provider: DeclarativeProvider, workload: FrozenWorkload
) -> tuple[tuple[CandidateProposal, ...], tuple[Mapping[str, Any], ...]]:
    """Split a provider's declarative configs into legal candidates and prunes.

    ``DeclarativeProvider.propose`` raises on the first config that fails
    ``validate_config`` against ``workload``'s knob contract; this pre-filter
    instead evaluates every config independently and reports each rejection
    -- a config never spends a single measurement.
    """

    if not isinstance(provider, DeclarativeProvider):
        raise ContractError("prefilter_candidates requires a DeclarativeProvider", code=FailureCode.PROVIDER_ERROR)
    if not isinstance(workload, FrozenWorkload):
        raise ContractError("prefilter_candidates requires a FrozenWorkload", code=FailureCode.PROVIDER_ERROR)
    legal: list[CandidateProposal] = []
    pruned: list[dict[str, Any]] = []
    for config in provider.configs:
        try:
            candidate = CandidateProposal(provider.provider_id, workload, dict(config))
        except ContractError as exc:
            pruned.append({"config": dict(config), "reason": exc.code.value, "message": str(exc)})
            continue
        legal.append(candidate)
    return tuple(legal), tuple(pruned)


def apply_max_candidates(
    candidates: Sequence[CandidateProposal], max_candidates: int | None
) -> tuple[tuple[CandidateProposal, ...], int]:
    """Truncate to at most ``max_candidates`` (in order); report how many were dropped."""

    if max_candidates is None:
        return tuple(candidates), 0
    if type(max_candidates) is not int or max_candidates < 0:
        raise ContractError("max_candidates must be a non-negative integer or null", code=FailureCode.INVALID_POLICY)
    kept = tuple(candidates[:max_candidates])
    return kept, max(0, len(candidates) - len(kept))


# ---------------------------------------------------------------------------
# Warm start: seed racing order from a prior winner for this exact
# (workload_hash, runtime identity) pair. A mismatched identity finds no
# history at all (list_tuning_history is itself keyed by the exact pair),
# so "ignore prior data on mismatch" is structural, not a separate check.
# ---------------------------------------------------------------------------


def warm_start_order(
    candidates: Sequence[CandidateProposal],
    *,
    store: ContentAddressedStore,
    workload_hash: str,
    runtime_identity: str,
) -> tuple[CandidateProposal, ...]:
    history = store.list_tuning_history(workload_hash, runtime_identity)
    ordered = list(candidates)
    if not history:
        return tuple(ordered)
    try:
        latest = store.get_tuning_summary(history[-1])
    except ContractError:
        return tuple(ordered)
    winner = latest.get("winner")
    if not isinstance(winner, Mapping):
        return tuple(ordered)
    winner_config = winner.get("config")
    if not isinstance(winner_config, Mapping):
        return tuple(ordered)
    for index, candidate in enumerate(ordered):
        if dict(candidate.config) == dict(winner_config):
            ordered.insert(0, ordered.pop(index))
            break
    return tuple(ordered)


# ---------------------------------------------------------------------------
# Racing.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RaceEntrant:
    """One candidate's final, terminal racing outcome."""

    candidate: CandidateProposal
    status: str
    block_count_used: int | None
    receipt_id: str | None
    attested: bool
    statistics: Mapping[str, Any] | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, CandidateProposal):
            raise ContractError("race entrant requires a CandidateProposal", code=FailureCode.WRONG_TYPE)
        if self.status not in ENTRANT_STATUSES:
            raise ContractError("race entrant status is not closed", code=FailureCode.INVALID_VALUE)
        if self.block_count_used is not None and (type(self.block_count_used) is not int or self.block_count_used < 1):
            raise ContractError("race entrant block_count_used must be a positive integer or null", code=FailureCode.WRONG_TYPE)
        if self.receipt_id is not None:
            validate_sha256(self.receipt_id)
        if type(self.attested) is not bool:
            raise ContractError("race entrant attested must be a boolean", code=FailureCode.WRONG_TYPE)
        _string(self.reason, label="race entrant reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "provider_id": self.candidate.provider_id,
            "config": dict(self.candidate.config),
            "status": self.status,
            "block_count_used": self.block_count_used,
            "receipt_id": self.receipt_id,
            "attested": self.attested,
            "statistics": None if self.statistics is None else dict(self.statistics),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RaceOutcome:
    entrants: tuple[RaceEntrant, ...]
    incumbent: RaceEntrant | None
    blocks_spent: int
    seconds_spent_ns: int
    budget_measurements: int | None
    budget_seconds: int | None
    budget_exhausted: bool


def _rung_policy(base_policy: EvaluationPolicy, block_count: int) -> EvaluationPolicy:
    fields = base_policy.to_dict()
    fields["measurement_runs"] = block_count
    fields["max_measurement_runs"] = block_count
    return EvaluationPolicy.from_dict(fields)


def race_candidates(
    *,
    candidates: Sequence[CandidateProposal],
    base_policy: EvaluationPolicy,
    run_rung: Callable[[CandidateProposal, EvaluationPolicy], ObservationBundle],
    store_receipt: Callable[[ObservationBundle, CandidateProposal, EvaluationPolicy], tuple[str, bool]],
    budget_measurements: int | None = None,
    budget_seconds: int | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RaceOutcome:
    """Race every candidate against the (implicit) baseline; honor the budget strictly.

    ``run_rung(candidate, rung_policy)`` must perform one real, self-
    contained evaluation at exactly ``rung_policy.measurement_runs ==
    rung_policy.max_measurement_runs`` blocks and return its
    ``ObservationBundle`` (never store anything itself).
    ``store_receipt(bundle, candidate, rung_policy)`` must build, store, and
    attempt attestation for the resulting receipt, returning
    ``(receipt_id, attested)`` -- called only for a candidate's terminal
    rung, exactly once per candidate that was measured at all.
    """

    if not isinstance(base_policy, EvaluationPolicy):
        raise ContractError("race_candidates requires an EvaluationPolicy", code=FailureCode.WRONG_TYPE)
    start = base_policy.measurement_runs
    cap = base_policy.max_measurement_runs
    if budget_measurements is not None and (type(budget_measurements) is not int or budget_measurements < 0):
        raise ContractError("budget_measurements must be a non-negative integer or null", code=FailureCode.INVALID_POLICY)
    if budget_seconds is not None and (type(budget_seconds) is not int or budget_seconds < 0):
        raise ContractError("budget_seconds must be a non-negative integer or null", code=FailureCode.INVALID_POLICY)

    order = [candidate.candidate_id for candidate in candidates]
    active: dict[str, CandidateProposal] = {candidate.candidate_id: candidate for candidate in candidates}
    resolved: dict[str, RaceEntrant] = {}
    last_seen: dict[str, tuple[int, Mapping[str, Any] | None]] = {}
    incumbent: RaceEntrant | None = None

    blocks_spent = 0
    started_at = clock()
    exhausted = False

    def _budget_ok() -> bool:
        if budget_measurements is not None and blocks_spent >= budget_measurements:
            return False
        if budget_seconds is not None and (clock() - started_at) >= budget_seconds:
            return False
        return True

    n = start
    while n <= cap and active:
        if not _budget_ok():
            exhausted = True
            break
        rung_policy = _rung_policy(base_policy, n)
        for candidate_id in list(order):
            if candidate_id not in active:
                continue
            if not _budget_ok():
                exhausted = True
                break
            candidate = active[candidate_id]
            bundle = run_rung(candidate, rung_policy)
            blocks_spent += n
            if not isinstance(bundle, ObservationBundle):
                raise ContractError("run_rung must return an ObservationBundle", code=FailureCode.WRONG_TYPE)
            statistics = bundle.statistics if bundle.accepted else None
            verdict = statistics.get("verdict") if statistics is not None else None
            if verdict is None:
                receipt_id, attested = store_receipt(bundle, candidate, rung_policy)
                resolved[candidate_id] = RaceEntrant(
                    candidate, STATUS_FAILED, n, receipt_id, attested, None,
                    "execution_not_accepted" if not bundle.accepted else "no_statistics",
                )
                del active[candidate_id]
                continue
            if verdict == VERDICT_IMPROVED:
                receipt_id, attested = store_receipt(bundle, candidate, rung_policy)
                entrant = RaceEntrant(candidate, STATUS_IMPROVED, n, receipt_id, attested, statistics, "decisive_improved")
                resolved[candidate_id] = entrant
                del active[candidate_id]
                if incumbent is None:
                    incumbent = entrant
                elif (
                    entrant.block_count_used is not None
                    and incumbent.block_count_used is not None
                    and entrant.block_count_used >= incumbent.block_count_used
                    and entrant.statistics["ci_lower_ns"] > incumbent.statistics["ci_lower_ns"]
                ):
                    incumbent = entrant
                continue
            if verdict == VERDICT_REGRESSED:
                receipt_id, attested = store_receipt(bundle, candidate, rung_policy)
                resolved[candidate_id] = RaceEntrant(candidate, STATUS_REGRESSED, n, receipt_id, attested, statistics, "decisive_regressed")
                del active[candidate_id]
                continue
            # Inconclusive: eliminate on futility (cannot statistically win),
            # finalize at the cap, or keep racing to the next rung.
            futile = statistics["ci_upper_ns"] < statistics["min_effect_ns"]
            if futile:
                receipt_id, attested = store_receipt(bundle, candidate, rung_policy)
                resolved[candidate_id] = RaceEntrant(
                    candidate, STATUS_ELIMINATED_FUTILE, n, receipt_id, attested, statistics,
                    "ci_upper_bound_below_min_effect_threshold",
                )
                del active[candidate_id]
                continue
            if n >= cap:
                receipt_id, attested = store_receipt(bundle, candidate, rung_policy)
                resolved[candidate_id] = RaceEntrant(
                    candidate, STATUS_INCONCLUSIVE_AT_CAP, n, receipt_id, attested, statistics,
                    "reached_block_cap_still_inconclusive",
                )
                del active[candidate_id]
                continue
            last_seen[candidate_id] = (n, statistics)
        if exhausted:
            break
        n += 1

    for candidate_id in order:
        if candidate_id in active:
            seen = last_seen.get(candidate_id)
            block_count_used, statistics = seen if seen is not None else (None, None)
            reason = "budget_exhausted" if seen is not None else "budget_exhausted_before_first_rung"
            resolved[candidate_id] = RaceEntrant(
                active[candidate_id], STATUS_UNRESOLVED_BUDGET, block_count_used, None, False, statistics, reason
            )
            exhausted = True

    seconds_spent_ns = max(0, int((clock() - started_at) * 1_000_000_000))
    entrants = tuple(resolved[candidate_id] for candidate_id in order)
    return RaceOutcome(
        entrants=entrants,
        incumbent=incumbent,
        blocks_spent=blocks_spent,
        seconds_spent_ns=seconds_spent_ns,
        budget_measurements=budget_measurements,
        budget_seconds=budget_seconds,
        budget_exhausted=exhausted,
    )


# ---------------------------------------------------------------------------
# Tuning summary: a closed, content-addressed report of one tune run.
# ---------------------------------------------------------------------------


_TUNING_SUMMARY_FIELDS: Final = {
    "schema", "workload_hash", "runtime", "provider_id", "policy", "created_at_ns",
    "prefilter", "budget", "entrants", "winner", "baseline", "summary_id",
}
_PRUNE_FIELDS: Final = {"config", "reason", "message"}
_PREFILTER_FIELDS: Final = {"considered", "pruned", "max_candidates", "max_candidates_dropped", "raced_count"}
_BUDGET_FIELDS: Final = {"budget_measurements", "budget_seconds", "blocks_spent", "seconds_spent_ns", "exhausted"}
_ENTRANT_FIELDS: Final = {
    "candidate_id", "provider_id", "config", "status", "block_count_used", "receipt_id", "attested", "statistics", "reason",
}
_WINNER_FIELDS: Final = {"candidate_id", "provider_id", "config", "receipt_id", "block_count_used", "ci_lower_ns"}
_BASELINE_FIELDS: Final = {"status", "note"}


def _validate_prune_entry(value: Any, *, index: int) -> None:
    data = _object(value, label=f"prefilter.pruned[{index}]")
    _exact(data, _PRUNE_FIELDS, label=f"prefilter.pruned[{index}]")
    _config_dict(data["config"], label=f"prefilter.pruned[{index}].config")
    _string(data["reason"], label=f"prefilter.pruned[{index}].reason")
    _string(data["message"], label=f"prefilter.pruned[{index}].message", non_empty=False)


def _validate_entrant_entry(value: Any, *, index: int) -> None:
    data = _object(value, label=f"entrants[{index}]")
    _exact(data, _ENTRANT_FIELDS, label=f"entrants[{index}]")
    validate_sha256(data["candidate_id"])
    _string(data["provider_id"], label=f"entrants[{index}].provider_id")
    _config_dict(data["config"], label=f"entrants[{index}].config")
    if data["status"] not in ENTRANT_STATUSES:
        raise ContractError(f"entrants[{index}].status is not closed", code=FailureCode.INVALID_VALUE)
    if data["block_count_used"] is not None:
        _integer(data["block_count_used"], label=f"entrants[{index}].block_count_used", minimum=1)
    if data["receipt_id"] is not None:
        validate_sha256(data["receipt_id"])
    _boolean(data["attested"], label=f"entrants[{index}].attested")
    if data["statistics"] is not None:
        _object(data["statistics"], label=f"entrants[{index}].statistics")
    _string(data["reason"], label=f"entrants[{index}].reason")


def _validate_winner(value: Any) -> None:
    if value is None:
        return
    data = _object(value, label="winner")
    _exact(data, _WINNER_FIELDS, label="winner")
    validate_sha256(data["candidate_id"])
    _string(data["provider_id"], label="winner.provider_id")
    _config_dict(data["config"], label="winner.config")
    validate_sha256(data["receipt_id"])
    _integer(data["block_count_used"], label="winner.block_count_used", minimum=1)
    _integer(data["ci_lower_ns"], label="winner.ci_lower_ns")


@dataclass(frozen=True, slots=True, init=False)
class TuningSummary:
    """A closed, immutable, content-addressed report of one `auto-mlx tune` run."""

    workload_hash: str
    runtime: RuntimeIdentity
    provider_id: str
    policy: EvaluationPolicy
    created_at_ns: int
    prefilter: Mapping[str, Any]
    budget: Mapping[str, Any]
    entrants: tuple[Mapping[str, Any], ...]
    winner: Mapping[str, Any] | None
    baseline: Mapping[str, Any]
    summary_id: str

    def __init__(
        self,
        *,
        workload_hash: str,
        runtime: RuntimeIdentity | Mapping[str, Any],
        provider_id: str,
        policy: EvaluationPolicy | Mapping[str, Any],
        prefilter: Mapping[str, Any],
        budget: Mapping[str, Any],
        entrants: Sequence[Mapping[str, Any]],
        winner: Mapping[str, Any] | None,
        created_at_ns: int | None = None,
        baseline: Mapping[str, Any] | None = None,
        summary_id: str | None = None,
    ) -> None:
        validate_sha256(workload_hash)
        frozen_runtime = runtime if isinstance(runtime, RuntimeIdentity) else RuntimeIdentity.from_dict(runtime)
        _string(provider_id, label="provider_id")
        frozen_policy = policy if isinstance(policy, EvaluationPolicy) else EvaluationPolicy.from_dict(policy)

        prefilter_data = _object(dict(prefilter), label="prefilter")
        _exact(prefilter_data, _PREFILTER_FIELDS, label="prefilter")
        _integer(prefilter_data["considered"], label="prefilter.considered", minimum=0)
        if type(prefilter_data["pruned"]) is not list:
            raise ContractError("prefilter.pruned must be an array", code=FailureCode.WRONG_TYPE)
        for index, entry in enumerate(prefilter_data["pruned"]):
            _validate_prune_entry(entry, index=index)
        if prefilter_data["max_candidates"] is not None:
            _integer(prefilter_data["max_candidates"], label="prefilter.max_candidates", minimum=0)
        _integer(prefilter_data["max_candidates_dropped"], label="prefilter.max_candidates_dropped", minimum=0)
        _integer(prefilter_data["raced_count"], label="prefilter.raced_count", minimum=0)

        budget_data = _object(dict(budget), label="budget")
        _exact(budget_data, _BUDGET_FIELDS, label="budget")
        if budget_data["budget_measurements"] is not None:
            _integer(budget_data["budget_measurements"], label="budget.budget_measurements", minimum=0)
        if budget_data["budget_seconds"] is not None:
            _integer(budget_data["budget_seconds"], label="budget.budget_seconds", minimum=0)
        _integer(budget_data["blocks_spent"], label="budget.blocks_spent", minimum=0)
        _integer(budget_data["seconds_spent_ns"], label="budget.seconds_spent_ns", minimum=0)
        _boolean(budget_data["exhausted"], label="budget.exhausted")

        entrant_list = [dict(entry) for entry in entrants]
        for index, entry in enumerate(entrant_list):
            _validate_entrant_entry(entry, index=index)
        if len(entrant_list) != prefilter_data["raced_count"]:
            raise ContractError("prefilter.raced_count does not match entrants", code=FailureCode.IDENTITY_MISMATCH)

        winner_data = None if winner is None else dict(winner)
        _validate_winner(winner_data)
        if winner_data is not None:
            matching = [
                entry for entry in entrant_list
                if entry["candidate_id"] == winner_data["candidate_id"] and entry["status"] == STATUS_IMPROVED
            ]
            if not matching:
                raise ContractError("winner does not match a decisively improved entrant", code=FailureCode.IDENTITY_MISMATCH)

        baseline_data = dict(baseline) if baseline is not None else {"status": "floor", "note": BASELINE_FLOOR_NOTE}
        _exact(baseline_data, _BASELINE_FIELDS, label="baseline")
        _string(baseline_data["status"], label="baseline.status")
        if baseline_data["status"] != "floor":
            raise ContractError("baseline.status must be floor", code=FailureCode.INVALID_VALUE)
        _string(baseline_data["note"], label="baseline.note", non_empty=False)

        timestamp = time.time_ns() if created_at_ns is None else _integer(created_at_ns, label="created_at_ns", minimum=0)

        object.__setattr__(self, "workload_hash", workload_hash)
        object.__setattr__(self, "runtime", frozen_runtime)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "policy", frozen_policy)
        object.__setattr__(self, "created_at_ns", timestamp)
        object.__setattr__(self, "prefilter", prefilter_data)
        object.__setattr__(self, "budget", budget_data)
        object.__setattr__(self, "entrants", tuple(entrant_list))
        object.__setattr__(self, "winner", winner_data)
        object.__setattr__(self, "baseline", baseline_data)

        expected_id = sha256_hex(self._body_dict())
        if summary_id is None:
            final_id = expected_id
        else:
            final_id = validate_sha256(summary_id)
            if final_id != expected_id:
                raise ContractError("summary_id does not match canonical summary body", code=FailureCode.IDENTITY_MISMATCH)
        object.__setattr__(self, "summary_id", final_id)

    def _body_dict(self) -> dict[str, Any]:
        return {
            "schema": TUNING_SUMMARY_SCHEMA,
            "workload_hash": self.workload_hash,
            "runtime": self.runtime.to_dict(),
            "provider_id": self.provider_id,
            "policy": self.policy.to_dict(),
            "created_at_ns": self.created_at_ns,
            "prefilter": dict(self.prefilter),
            "budget": dict(self.budget),
            "entrants": [dict(entry) for entry in self.entrants],
            "winner": None if self.winner is None else dict(self.winner),
            "baseline": dict(self.baseline),
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._body_dict()
        result["summary_id"] = self.summary_id
        return result

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Any) -> "TuningSummary":
        data = _object(value, label="tuning summary")
        _exact(data, _TUNING_SUMMARY_FIELDS, label="tuning summary")
        if data["schema"] != TUNING_SUMMARY_SCHEMA:
            raise ContractError("tuning summary schema is incompatible", code=FailureCode.INVALID_VALUE)
        return cls(
            workload_hash=data["workload_hash"],
            runtime=data["runtime"],
            provider_id=data["provider_id"],
            policy=data["policy"],
            created_at_ns=data["created_at_ns"],
            prefilter=data["prefilter"],
            budget=data["budget"],
            entrants=data["entrants"],
            winner=data["winner"],
            baseline=data["baseline"],
            summary_id=data["summary_id"],
        )

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "TuningSummary":
        return cls.from_dict(strict_json_loads(value))


def build_tuning_summary(
    *,
    workload_hash: str,
    runtime: RuntimeIdentity,
    provider_id: str,
    base_policy: EvaluationPolicy,
    considered: int,
    pruned: Sequence[Mapping[str, Any]],
    max_candidates: int | None,
    max_candidates_dropped: int,
    outcome: RaceOutcome,
    created_at_ns: int | None = None,
) -> TuningSummary:
    """Assemble the closed :class:`TuningSummary` from a completed race."""

    winner_data: dict[str, Any] | None = None
    if outcome.incumbent is not None:
        entrant = outcome.incumbent
        winner_data = {
            "candidate_id": entrant.candidate.candidate_id,
            "provider_id": entrant.candidate.provider_id,
            "config": dict(entrant.candidate.config),
            "receipt_id": entrant.receipt_id,
            "block_count_used": entrant.block_count_used,
            "ci_lower_ns": entrant.statistics["ci_lower_ns"],
        }
    return TuningSummary(
        workload_hash=workload_hash,
        runtime=runtime,
        provider_id=provider_id,
        policy=base_policy,
        created_at_ns=created_at_ns,
        prefilter={
            "considered": considered,
            "pruned": [dict(entry) for entry in pruned],
            "max_candidates": max_candidates,
            "max_candidates_dropped": max_candidates_dropped,
            "raced_count": len(outcome.entrants),
        },
        budget={
            "budget_measurements": outcome.budget_measurements,
            "budget_seconds": outcome.budget_seconds,
            "blocks_spent": outcome.blocks_spent,
            "seconds_spent_ns": outcome.seconds_spent_ns,
            "exhausted": outcome.budget_exhausted,
        },
        entrants=[entrant.to_dict() for entrant in outcome.entrants],
        winner=winner_data,
    )


__all__: Final = [
    "BASELINE_FLOOR_NOTE",
    "ENTRANT_STATUSES",
    "RaceEntrant",
    "RaceOutcome",
    "STATUS_ELIMINATED_FUTILE",
    "STATUS_FAILED",
    "STATUS_IMPROVED",
    "STATUS_INCONCLUSIVE_AT_CAP",
    "STATUS_REGRESSED",
    "STATUS_UNRESOLVED_BUDGET",
    "TUNING_SUMMARY_SCHEMA",
    "TuningSummary",
    "apply_max_candidates",
    "build_tuning_summary",
    "prefilter_candidates",
    "race_candidates",
    "warm_start_order",
]
