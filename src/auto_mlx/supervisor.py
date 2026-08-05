"""The local supervisor: the sole authority permitted to mint a receipt attestation.

Evaluation (:mod:`auto_mlx.evaluator`) and attestation are separate
authorities (see ``docs/evidence-and-promotion.md``).  The evaluator
retains observations; it never sees, imports, or references attestation
key material (see ``tests/test_supervisor.py::EvaluatorKeyIsolationTests``
for the module-boundary proof).  This module is the only place in the
codebase that is meant to call
:func:`auto_mlx.receipts.receipt_attestation` with a real key drawn from
:mod:`auto_mlx.keys` -- and it only ever does so after independently
re-establishing every layer of evidence :func:`auto_mlx.receipts.validate_receipt`
already knows how to check:

1. Every derived aggregate is recomputed from the receipt's raw samples
   (never trusted from the stored value) -- reusing
   :func:`auto_mlx.receipts.recompute_receipt_fields` via
   :func:`auto_mlx.receipts.validate_receipt`.
2. Identity and canonical form (receipt_id vs. body hash, workload/
   candidate/policy/runtime binding, evaluator-bundle identity) --
   likewise reused, not reimplemented.
3. The evidence chain: the receipt represents accepted, promotion-eligible
   evidence (``receipt.status == "complete"``, no failure), oracle parity
   across every sample (``recomputed.oracle.all_match``), and structurally
   complete measurement pairs (``recomputed.compatibility.compatible``).

Only when every one of those holds does this module apply the supervisor
HMAC.  Any mismatch is a typed, fail-closed :class:`auto_mlx.errors.SupervisorRefusalError`
that states exactly what failed -- this function never attests on a guess.
"""

from __future__ import annotations

import os
from typing import Final

from .errors import FailureCode, SupervisorRefusalError
from .receipts import Receipt, receipt_attestation, validate_receipt


def attest_receipt(
    receipt: Receipt,
    key: bytes | bytearray,
    *,
    artifact_root: str | os.PathLike[str] | None = None,
) -> str:
    """Independently verify ``receipt``'s evidence chain, then mint its HMAC.

    ``artifact_root``, when given, additionally requires every declared
    artifact to verify against real files under that root before attesting
    (see :func:`auto_mlx.paths.verify_artifact`).  Omitting it attests the
    receipt's execution evidence only; activation-time artifact
    verification is a separate, later gate (see
    :func:`auto_mlx.promotion.activate`).
    """

    if not isinstance(receipt, Receipt):
        raise SupervisorRefusalError("supervisor requires a closed Receipt value", code=FailureCode.WRONG_TYPE)

    validation = validate_receipt(receipt, artifact_root=artifact_root)

    if not validation.valid:
        reasons = "; ".join(f"{failure.code.value}: {failure.message}" for failure in validation.failures)
        raise SupervisorRefusalError(
            f"receipt did not independently recompute to its stored evidence: {reasons}",
            code=FailureCode.SUPERVISOR_REFUSED,
        )
    if not validation.local:
        raise SupervisorRefusalError("receipt is not local-evaluation provenance", code=FailureCode.SUPERVISOR_REFUSED)
    if not validation.complete:
        raise SupervisorRefusalError("receipt does not represent complete, closed evidence", code=FailureCode.SUPERVISOR_REFUSED)
    if receipt.status != "complete" or receipt.failure is not None:
        raise SupervisorRefusalError(
            "receipt does not represent accepted, failure-free evidence", code=FailureCode.SUPERVISOR_REFUSED
        )

    oracle = validation.recomputed.get("oracle", {})
    if oracle.get("all_match") is not True:
        raise SupervisorRefusalError("oracle parity was not independently established", code=FailureCode.SUPERVISOR_REFUSED)

    compatibility = validation.recomputed.get("compatibility", {})
    if compatibility.get("compatible") is not True:
        raise SupervisorRefusalError(
            "measurement pairs are incomplete or structurally malformed", code=FailureCode.SUPERVISOR_REFUSED
        )

    if artifact_root is not None and not validation.artifacts_verified:
        raise SupervisorRefusalError(
            "declared artifacts could not be independently verified", code=FailureCode.SUPERVISOR_REFUSED
        )

    return receipt_attestation(receipt, key)


__all__: Final = ["attest_receipt"]
