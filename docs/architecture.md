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
| G1 evaluator | Library implemented; production/CLI gated | `Evaluator` owns subprocess plans, timing, output limits, source-oracle comparison, and complete paired observations. Production evaluation may fail closed until real isolation and supervisor proof exist; CLI orchestration remains deferred. |
| G2 receipts/promotion/dispatch | Libraries implemented; activation/CLI gated | Receipt validation/storage, promotion decisions, and exact-match dispatch with native fallback exist as libraries. Production activation remains gated on later evidence and activation proof; CLI orchestration remains deferred. |
| Compiler search/custom Metal | Deferred | Research only until parity, measurement, fallback, and rollback gates pass. |

The separation is intentional: a document can be valid without being runnable, a receipt can be valid without being promotable, and a local observation is not a public performance claim.

## Identity surfaces

Canonical JSON is compact, sorted-key UTF-8 JSON with no floating-point values. Workload and candidate identity therefore binds all declared parameters, artifacts, knobs, provider, and configuration. Generic document inspection reports the SHA-256 of the canonical value; it does not grant that document receipt or promotion semantics.

## Design lessons versus evidence

The architecture follows lessons from systems that separate candidate generation, measurement, and result application. Those are design inputs. This repository has measured no MLX speedup; a library being implemented is not evidence that production activation is safe. See [research-landscape.md](research-landscape.md) for source boundaries.
