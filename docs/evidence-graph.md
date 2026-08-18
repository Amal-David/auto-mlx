# The evidence graph and the pre-race advisor

This document describes the two Phase-1 pieces of Auto MLX's knowledge layer:
the **evidence-graph contract** (`auto_mlx.graph`, `schemas/evidence_graph.json`)
and the **pre-race advisor** (`auto_mlx.advisor`). It states plainly what is
implemented, what is enforced, and what remains unmeasured.

## Why a graph at all

Receipts are the unit of *measurement*; mechanisms are the unit of
*knowledge*. A receipt proves that one configuration, on one workload, on one
runtime, at one time, was faster, slower, or indistinguishable. The thing
worth keeping is one level up: "this mechanism, applied to models exhibiting
these architectural traits, produced these measured effects on this
hardware." That is what transfers across models, hardware, and (eventually)
contributors -- and it is what the evidence graph stores.

The graph is a **decision aid with citations, never measurement evidence**:

- **The graph never promotes.** Activation still requires a fresh, attested,
  independently recomputed receipt through the unchanged promotion lane.
  Nothing in `auto_mlx.graph` or `auto_mlx.advisor` can activate a candidate.
- **Every shortcut cites its receipt.** Any decision derived from stored
  knowledge names the receipt and summary it came from, and the citation is
  re-verified before it is honored. An uncitable shortcut is refused.
- **`inconclusive` never hardens into a prune.** Noise floors and thresholds
  move; only a decisive regression is prune-grade evidence.

## The contract (schema version 2)

One graph document: `{schema_version: 2, graph_id, generated_at, nodes,
edges}`, strict-JSON like every other Auto MLX contract (no floats, no
duplicate keys, no unknown fields, fail closed).

**Node kinds.** `model`, `trait`, `hardware`, `workload`, `mechanism`,
`applied_result`, `hypothesis`, `constraint`, `finding`,
`external_reference`, `frontier`. The transfer unit is the **trait** -- an
architectural property (`trait:affine-q4-group-64`, `trait:gated-delta-net`)
that models *exhibit* and mechanisms *apply to*. Transfer is predicted along
shared traits, never along model names.

**Reified hyperedges.** The relationship worth storing is 4-ary --
"mechanism × model × hardware × workload → effect" -- but n-ary edges make
weak contracts. Instead an `applied_result` node reifies the hyperedge: it
carries the structured effect and points outward through typed binary edges
(`instantiates` → mechanism, `applied_on` → model, `measured_on` → hardware,
`under_workload` → workload). Plain edges, hypergraph semantics.

**Effects are integers.** `effect.delta_bp_ci` is a `[lower, upper]` pair in
basis points (-2.57% is `-257`); a single unrepeated run is a point interval
with a `sample_note` recording the caveat. `verdict` is closed to
`improved`/`regressed`/`inconclusive`. `receipt_id`, when present, is a
64-hex content address. The optional `metric_direction`
(`higher_is_better`/`lower_is_better`) makes decisive verdicts mechanically
checkable: with a direction declared, the WHOLE interval must sit strictly
on the claimed side of zero -- `improved` under `higher_is_better` requires
`lower > 0`, so `[0, 100]` and any zero-touching or zero-straddling interval
are rejected. Touching zero does not support a sign; the honest verdict for
such an interval is `inconclusive`, which is never sign-constrained.

**Provenance is a ladder, never flattened.** `official_verified`,
`code_verified`, `local_measured`, `author_claim`, `contributor_claim`,
`replicated`, `research_inference`, `transfer_inference`. A claim keeps the
weakest provenance it has actually earned; merging never upgrades it.

**Typed relations are endpoint-checked.** `exhibits` (model→trait),
`applies_to` (mechanism→trait), `instantiates`/`applied_on`/`measured_on`/
`under_workload` (applied_result→...), `transfer_predicted`
(mechanism→model), and `confirms`/`refutes` (applied_result→mechanism or
hypothesis) reject wrong-kind endpoints. The inherited campaign vocabulary
(`supports`, `constrains`, `invalidates`, `supersedes`, `contradicts`, ...)
is intentionally unconstrained. `confirms`/`refutes` close the transfer
loop: only a reified measurement may confirm or refute a prediction, which
is how transfer reliability becomes a measured quantity instead of a hope.

Validate and inspect from the CLI:

```bash
auto-mlx validate graph --input examples/evidence-graph.json
auto-mlx inspect graph --input examples/evidence-graph.json
```

The JSON Schema twin ships as a package resource
(`auto_mlx/schemas/evidence_graph.json`) for external tooling; the Python
module remains the executable authority and additionally enforces global id
uniqueness, dangling-edge rejection, and the endpoint-kind table.

## The advisor: climbing known hills cheaply

`auto-mlx tune` re-races every legal candidate from the bottom rung on every
run, even when this exact `(workload_hash, runtime_identity)` pair already
holds attested terminal verdicts. `auto_mlx.advisor` closes that gap:

- A candidate whose configuration previously reached a **decisive
  regression** at the same identity is pruned before spending a single
  block -- only when the prior entrant was attested, its receipt still
  independently loads from the content-addressed store, and no contradictory
  `improved` verdict exists anywhere in stored history. The prune is
  recorded in the tuning summary's `prefilter.pruned` with `receipt_id` and
  `summary_id` citations, so `auto-mlx history` can walk the evidence chain.
- A prior **winner** is seeded first in racing order -- and still re-races
  in full. Seeding is ordering, not promotion.
- Prior **futile or at-cap-inconclusive** verdicts demote a candidate to the
  back of the order. They never prune: a decisive regression means the whole
  bootstrap CI sat below the negative min-effect bound (strictly slower at
  any threshold); an inconclusive result means only that *this* noise floor
  could not resolve it.
- **Contradictory history disables the shortcut** and is surfaced in the
  advice output. Conflicting evidence is a signal, never something to
  average away.
- **Identity drift finds no history at all.** History is keyed by the exact
  `(workload_hash, runtime_identity)` pair, so a changed runtime, MLX
  version, or workload inherits nothing -- structurally, not by policy.

`--no-advice` opts out entirely. The advice actions are a closed vocabulary
-- prune, seed, demote -- with no promote/activate member, so the "graph
never promotes" invariant is enforced by the type system, not by discipline.

## What remains unmeasured, honestly

- **No cross-model transfer is implemented yet.** Fingerprinting a model
  into a trait vector and ranking mechanisms by `applies_to` coverage and
  measured effect is Phase 2. The `transfer_predicted`/`confirms`/`refutes`
  vocabulary exists so those predictions land as first-class, refutable
  graph objects -- but as of today no transfer has been predicted, let alone
  confirmed.
- **On the checked-in toy workload, advice cannot prune.** `toy-matmul` has
  no real effect to find (the `tile` knob is inert by construction, and
  eager-vs-compiled is indistinguishable at this host's noise floor), so its
  history holds only inconclusive verdicts -- which, by design, only ever
  demote. Block savings from pruning require a workload with a genuine
  decisive effect; realistic LLM prefill/decode workloads remain future
  work, as the roadmap has always said. The prune path is exercised by
  deterministic unit fixtures (`tests/test_advisor.py`), not by a live
  measured regression.
- **The graph does not yet feed the advisor.** Phase 1 advice reads tuning
  summaries and receipts directly; the graph is validated, stored, and
  exchanged with the knowledge repository, but no query path from graph
  nodes to racing decisions exists yet. That wiring is Phase 2, and it will
  keep the same invariants: reorder, defer, prune-with-citation -- never
  decide.
