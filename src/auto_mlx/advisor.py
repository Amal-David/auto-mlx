"""Pre-race advice derived from stored, attested prior evidence.

``auto-mlx tune`` re-races every legal candidate from the bottom rung on every
run, even when this exact ``(workload_hash, runtime_identity)`` pair already
holds attested terminal verdicts for some configurations. This module turns
that stored history into *advice* for the racer -- and nothing stronger:

- **Advice never promotes.** A prior ``improved`` verdict only seeds racing
  order; becoming a winner still requires a fresh, decisive, receipt-backed
  ``improved`` verdict in the current race, through the unchanged racing and
  promotion lanes.
- **Every shortcut cites its receipt.** A candidate is pruned only when its
  prior decisive regression names a stored receipt, the entrant was attested,
  and that receipt still loads from the content-addressed store. If the
  citation cannot be produced and re-verified, the shortcut is refused and
  the candidate races normally -- an uncited shortcut is not an optimization,
  it is a contract violation.
- **``inconclusive`` never hardens into a prune.** Futile eliminations and
  at-cap inconclusive verdicts only demote racing order; they never remove a
  candidate. Thresholds and noise floors move between runs; only a decisive
  regression (whole CI below the negative min-effect bound, which implies
  strictly-slower-than-baseline at any threshold) is prune-grade evidence.
- **Identity drift finds no history at all.** History is keyed by the exact
  ``(workload_hash, runtime_identity)`` pair (the same structural gate
  ``warm_start_order`` relies on), so a changed runtime or workload cannot
  inherit prunes -- there is nothing to inherit.
- **Contradictory history disables the shortcut.** If any prior run reached
  a decisive ``improved`` verdict for a configuration whose latest verdict is
  ``regressed``, the conflict is surfaced and the candidate races normally;
  conflicting evidence is a signal, never something to average away.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from .canonical import canonical_json
from .contracts import CandidateProposal
from .errors import ContractError, FailureCode
from .receipts import ContentAddressedStore
from .tune import (
    STATUS_ELIMINATED_FUTILE,
    STATUS_IMPROVED,
    STATUS_INCONCLUSIVE_AT_CAP,
    STATUS_REGRESSED,
)


ACTION_PRUNE: Final = "prune"
ACTION_SEED: Final = "seed"
ACTION_DEMOTE: Final = "demote"
_ACTIONS: Final = frozenset({ACTION_PRUNE, ACTION_SEED, ACTION_DEMOTE})

PRUNE_REASON: Final = "advised_prior_regression"
SEED_REASON: Final = "advised_prior_winner"
DEMOTE_REASON: Final = "advised_prior_inconclusive"

DEFAULT_MAX_SUMMARIES: Final = 64


@dataclass(frozen=True, slots=True)
class AdviceEntry:
    """One cited, per-candidate piece of advice."""

    candidate_id: str
    config: Mapping[str, Any]
    action: str
    reason: str
    receipt_id: str | None
    summary_id: str | None
    detail: str

    def __post_init__(self) -> None:
        if self.action not in _ACTIONS:
            raise ContractError("advice action is not closed", code=FailureCode.INVALID_VALUE)
        if self.action == ACTION_PRUNE and (self.receipt_id is None or self.summary_id is None):
            raise ContractError(
                "a prune advice entry must cite both its receipt and its tuning summary",
                code=FailureCode.INVALID_VALUE,
            )
        object.__setattr__(self, "config", dict(self.config))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "config": dict(self.config),
            "action": self.action,
            "reason": self.reason,
            "receipt_id": self.receipt_id,
            "summary_id": self.summary_id,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class RaceAdvice:
    """The full advice for one upcoming race, with its evidence trail."""

    workload_hash: str
    runtime_identity: str
    considered_summaries: tuple[str, ...]
    entries: tuple[AdviceEntry, ...]
    conflicts: tuple[str, ...]

    @property
    def prunes(self) -> tuple[AdviceEntry, ...]:
        return tuple(entry for entry in self.entries if entry.action == ACTION_PRUNE)

    @property
    def seeds(self) -> tuple[AdviceEntry, ...]:
        return tuple(entry for entry in self.entries if entry.action == ACTION_SEED)

    @property
    def demotions(self) -> tuple[AdviceEntry, ...]:
        return tuple(entry for entry in self.entries if entry.action == ACTION_DEMOTE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_hash": self.workload_hash,
            "runtime_identity": self.runtime_identity,
            "considered_summaries": list(self.considered_summaries),
            "entries": [entry.to_dict() for entry in self.entries],
            "conflicts": list(self.conflicts),
        }


def _config_key(config: Mapping[str, Any]) -> str:
    return canonical_json(dict(config))


def _receipt_still_loads(store: ContentAddressedStore, receipt_id: str) -> bool:
    try:
        store.get_receipt(receipt_id)
    except (ContractError, OSError):
        return False
    return True


def advise_candidates(
    candidates: Sequence[CandidateProposal],
    *,
    store: ContentAddressedStore,
    workload_hash: str,
    runtime_identity: str,
    max_summaries: int = DEFAULT_MAX_SUMMARIES,
) -> RaceAdvice:
    """Derive cited racing advice for ``candidates`` from stored history.

    Reads at most ``max_summaries`` of the most recent tuning summaries for
    the exact ``(workload_hash, runtime_identity)`` pair. A summary that
    fails its own independent re-validation is skipped entirely (fail closed
    to *no* advice from it, never to trusting it).
    """

    if type(max_summaries) is not int or max_summaries < 1:
        raise ContractError("max_summaries must be a positive integer", code=FailureCode.INVALID_VALUE)

    history = store.list_tuning_history(workload_hash, runtime_identity)
    considered: list[str] = []
    # Newest first: index 0 is the most recent summary.
    summaries: list[dict[str, Any]] = []
    for summary_id in reversed(history[-max_summaries:]):
        try:
            summary = store.get_tuning_summary(summary_id)
        except (ContractError, OSError):
            continue
        summaries.append(summary)
        considered.append(summary_id)

    # Per config key, newest-first list of (summary_id, entrant).
    entrant_history: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    latest_winner: tuple[str, Mapping[str, Any]] | None = None
    for summary in summaries:
        summary_id = summary["summary_id"]
        winner = summary.get("winner")
        if latest_winner is None and isinstance(winner, Mapping):
            latest_winner = (summary_id, winner)
        for entrant in summary["entrants"]:
            entrant_history.setdefault(_config_key(entrant["config"]), []).append((summary_id, entrant))

    entries: list[AdviceEntry] = []
    conflicts: list[str] = []
    for candidate in candidates:
        key = _config_key(candidate.config)
        records = entrant_history.get(key, [])
        if not records:
            continue
        latest_terminal: tuple[str, Mapping[str, Any]] | None = None
        ever_improved = False
        for summary_id, entrant in records:
            status = entrant["status"]
            if status == STATUS_IMPROVED:
                ever_improved = True
            if latest_terminal is None and status in {
                STATUS_IMPROVED,
                STATUS_REGRESSED,
                STATUS_ELIMINATED_FUTILE,
                STATUS_INCONCLUSIVE_AT_CAP,
            }:
                latest_terminal = (summary_id, entrant)
        if latest_terminal is None:
            continue
        summary_id, entrant = latest_terminal
        status = entrant["status"]

        if status == STATUS_REGRESSED:
            if ever_improved:
                conflicts.append(candidate.candidate_id)
                continue
            receipt_id = entrant["receipt_id"]
            statistics = entrant["statistics"]
            decisive = (
                receipt_id is not None
                and entrant["attested"] is True
                and isinstance(statistics, Mapping)
                and type(statistics.get("ci_upper_ns")) is int
                and statistics["ci_upper_ns"] < 0
            )
            if not decisive or not _receipt_still_loads(store, receipt_id):
                # An uncited or unverifiable shortcut is refused: race normally.
                continue
            entries.append(
                AdviceEntry(
                    candidate_id=candidate.candidate_id,
                    config=candidate.config,
                    action=ACTION_PRUNE,
                    reason=PRUNE_REASON,
                    receipt_id=receipt_id,
                    summary_id=summary_id,
                    detail=(
                        "prior decisive regression at this exact workload/runtime identity: "
                        f"ci_upper_ns={statistics['ci_upper_ns']}"
                    ),
                )
            )
            continue

        if status in {STATUS_ELIMINATED_FUTILE, STATUS_INCONCLUSIVE_AT_CAP}:
            entries.append(
                AdviceEntry(
                    candidate_id=candidate.candidate_id,
                    config=candidate.config,
                    action=ACTION_DEMOTE,
                    reason=DEMOTE_REASON,
                    receipt_id=entrant["receipt_id"],
                    summary_id=summary_id,
                    detail=f"prior non-decisive verdict {status!r}; racing order only, never a prune",
                )
            )
            continue

        if status == STATUS_IMPROVED and latest_winner is not None:
            winner_summary_id, winner = latest_winner
            if _config_key(winner["config"]) == key:
                entries.append(
                    AdviceEntry(
                        candidate_id=candidate.candidate_id,
                        config=candidate.config,
                        action=ACTION_SEED,
                        reason=SEED_REASON,
                        receipt_id=winner["receipt_id"],
                        summary_id=winner_summary_id,
                        detail="most recent stored winner for this identity; seeded first, still re-raced in full",
                    )
                )

    return RaceAdvice(
        workload_hash=workload_hash,
        runtime_identity=runtime_identity,
        considered_summaries=tuple(considered),
        entries=tuple(entries),
        conflicts=tuple(conflicts),
    )


def apply_advice(
    candidates: Sequence[CandidateProposal], advice: RaceAdvice
) -> tuple[tuple[CandidateProposal, ...], tuple[dict[str, Any], ...]]:
    """Apply ``advice`` to a candidate sequence.

    Returns the surviving candidates in advised order (seeds first, undecided
    candidates in provider order, demotions last) plus the cited prune
    entries in the tuning summary's ``prefilter.pruned`` shape.
    """

    by_candidate: dict[str, AdviceEntry] = {entry.candidate_id: entry for entry in advice.entries}
    seeds: list[CandidateProposal] = []
    middle: list[CandidateProposal] = []
    demoted: list[CandidateProposal] = []
    prunes: list[dict[str, Any]] = []
    for candidate in candidates:
        entry = by_candidate.get(candidate.candidate_id)
        if entry is None:
            middle.append(candidate)
            continue
        if entry.action == ACTION_PRUNE:
            prunes.append(
                {
                    "config": dict(candidate.config),
                    "reason": entry.reason,
                    "message": entry.detail,
                    "receipt_id": entry.receipt_id,
                    "summary_id": entry.summary_id,
                }
            )
            continue
        if entry.action == ACTION_SEED:
            seeds.append(candidate)
            continue
        demoted.append(candidate)
    return tuple(seeds + middle + demoted), tuple(prunes)


__all__: Final = [
    "ACTION_DEMOTE",
    "ACTION_PRUNE",
    "ACTION_SEED",
    "AdviceEntry",
    "DEFAULT_MAX_SUMMARIES",
    "DEMOTE_REASON",
    "PRUNE_REASON",
    "RaceAdvice",
    "SEED_REASON",
    "advise_candidates",
    "apply_advice",
]
