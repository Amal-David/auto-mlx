# Auto-MLX v2: trait-indexed evidence hypergraph + transfer + collective knowledge

Approved goal (2026-08-17): rebuild Auto-MLX around a trait-indexed evidence
hypergraph so optimization knowledge compounds — same-model re-runs are cheap,
architecturally similar models inherit calibrated priors, and contributor
evidence merges into one collective, receipt-backed knowledge base.

Decisions locked with the user:
- Execution stack: **mlx-lm (Python)**; mlx-swift is an evidence source only.
- Two repos: `auto-mlx` = tool (contracts/validator/racer/fingerprint);
  `mlx-porting-skill` = knowledge repo (graph shards, evidence packs, CI,
  skill rendering). An Opus sub-agent re-architects the knowledge repo in
  parallel.
- Founding knowledge: Qwen 3.8 campaign ledger at
  `/Users/amal/experiments.noindex/personal/qwen38-challenge/.auto-mlx/`
  (82-node hypergraph, PR-corpus summaries) + the skill's existing 712-node
  `knowledge_graph.json` and YAML stores.
- Invariants: graph never promotes; every shortcut cites a receipt;
  identity drift downgrades prune→reorder; `inconclusive` never prunes;
  everything fails closed; no floats (basis points for effects).

## Phase 1 — v2 graph contracts + racer integration (this repo)

- [ ] Design shared evidence-graph schema v2 (traits, mechanisms, reified
      applied_result, structured integer-bp effects, provenance tiers,
      freshness) — `src/auto_mlx/schemas/evidence-graph.schema.json`
- [ ] `src/auto_mlx/graph.py`: strict fail-closed contracts mirroring the
      JSON schema, plus structural rules (unique ids, no dangling edges,
      endpoint-kind constraints for typed relations, effect required on
      applied_result)
- [ ] CLI: `auto-mlx validate graph` / `auto-mlx inspect graph`
- [ ] Tests: contract round-trips, fail-closed cases, schema parity
- [ ] `src/auto_mlx/advisor.py`: pre-race advice from stored receipts +
      tuning summaries at exact (workload_hash, runtime_identity) —
      prune proven regressions (with receipt citation), seed proven winner,
      never prune on inconclusive, downgrade on identity mismatch
- [ ] Wire advisor into `auto-mlx tune` (opt-out flag) and surface
      advice + citations in the tuning summary
- [ ] Tests for advisor invariants (each maps to an acceptance criterion)
- [ ] Build verification: full pytest suite green
- [ ] Docs: README + docs/evidence-graph.md in the repo's candid register
- [ ] Git checkpoints at each verified step

## Phase 1b — knowledge repo re-architecture (sub-agent, parallel)

- [ ] Storage re-architecture in `mlx-porting-skill`: `graph/` canonical JSON
      shards conforming to schema v2 + compiled index + validation tooling
- [ ] Migrate the 712-node v1 graph + techniques/architectures/guidance YAML
      into v2 nodes (mechanism/trait/external_reference/applied_result)
- [ ] Port Qwen 3.8 campaign ledger + PR corpus into founding shards
      (scrubbed: no tokens, no credentials, no private submission details)
- [ ] Evidence-pack contribution format + validation script
- [ ] SKILL.md rendering layer connected to the graph

## Phase 2 — fingerprint + transfer (after Phase 1)

- [ ] `auto-mlx fingerprint`: HF config.json + weight index → trait vector
- [ ] Transfer prediction: mechanisms whose applies_to ⊆ model traits, ranked
      by effect × transfer track record × hardware proximity → hypothesis queue
- [ ] Confirm/refute loop writes back to graph; pick model #2 and prove one
      transfer end-to-end with a fresh receipt

## Phase 3 — community layer (after first confirmed transfer)

- [ ] Evidence packs CI, trust tiers (contributor_claim → replicated →
      maintainer_verified), merge semantics (conflicts become contradicts
      edges), per-Mac benchmark pages

## Review

(to be filled as work completes)
