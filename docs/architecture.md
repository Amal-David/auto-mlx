# Architecture

Auto MLX is a deliberately narrow control plane. The current path is:

```text
local JSON/stdin
    -> strict parser (duplicate keys and floats rejected)
    -> typed immutable contract
    -> canonical JSON
    -> SHA-256 identity and dry-run result
```

The CLI never turns a document into a command. A `DeclarativeProvider` contains frozen scalar maps; `CandidateProposal` validates those maps against a `FrozenWorkload` and derives `candidate_id` from `provider_id`, `workload_hash`, and `config`. A caller cannot provide or override that ID.

Artifacts are identified by a relative POSIX path, byte size, and SHA-256. Verification uses descriptor-relative no-follow operations where available and fails closed when that safety primitive is unavailable. This is an integrity gate, not a sandbox.

## Current and future lanes

| Lane | Current status | Boundary |
| --- | --- | --- |
| G0 contracts | Implemented | Parse, validate, canonicalize, hash, and inspect local documents. |
| G1 evaluator | Library implemented; production/CLI gated | `Evaluator` owns plans, output/oracle evidence, and complete paired observations. `execute_plan` runs for real when (a) the host has the local sandbox primitives (macOS, `sandbox-exec` on PATH, descriptor-relative artifact access) and (b) a caller explicitly constructs and passes a concrete `IsolationProvider`/`IsolationAuthority` pair -- `auto_mlx.sandbox.LocalSandboxProvider`/`LocalSandboxAuthority` is the first such pair (a developer-grade, single-user local guard; see docs/threat-model.md). Without either condition, execution stays exactly as fail-closed (`SANDBOX_UNAVAILABLE`) as a host with no execution engine at all -- this is true unconditionally on non-macOS hosts, including every Linux CI runner. CLI orchestration remains deferred. |
| G2 receipts/promotion/dispatch | Libraries implemented; activation/CLI gated | Receipt validation/storage, promotion decisions, and exact-match dispatch with native fallback exist as libraries. Production activation remains gated on later evidence and activation proof; CLI orchestration remains deferred. G1 evidence-layer acceptance is a necessary input to, not a substitute for, this independent gate: `receipts.py`'s evaluator-bundle adapter still unconditionally reports `status="failed"` for every G1-derived receipt. |
| Compiler search/custom Metal | Deferred | Research only until parity, measurement, fallback, and rollback gates pass. |

The separation is intentional: a document can be valid without being runnable, a receipt can be valid without being promotable, and a local observation is not a public performance claim.

## Identity surfaces

Canonical JSON is compact, sorted-key UTF-8 JSON with no floating-point values. Workload and candidate identity therefore binds all declared parameters, artifacts, knobs, provider, and configuration. Generic document inspection reports the SHA-256 of the canonical value; it does not grant that document receipt or promotion semantics.

## JSON Schema documents

The Draft 2020-12 schema for every G0 contract lives at `src/auto_mlx/schemas/*.json` and ships as an importable `auto_mlx.schemas` package resource (`from auto_mlx.schemas import schema_names, schema_text`), so external tooling can validate Auto MLX documents without importing the Python contract classes. This is the single source of truth for the schemas; they previously lived at a repo-root `schemas/` directory, which no longer exists. The schemas describe the same wire shape the `auto_mlx.contracts` and `auto_mlx.receipts` classes enforce in Python; `tests/test_schema_parity.py` checks the two stay in agreement.

## Design lessons versus evidence

The architecture follows lessons from systems that separate candidate generation, measurement, and result application. Those are design inputs. This repository has measured no MLX speedup; a library being implemented is not evidence that production activation is safe. See [research-landscape.md](research-landscape.md) for source boundaries.
