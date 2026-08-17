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

## Phase 1 — v2 graph contracts + racer integration (this repo) — DONE 2026-08-17

- [x] Shared evidence-graph schema v2 — `src/auto_mlx/schemas/evidence_graph.json`
      (renamed to house convention; surrogate propertyNames guards added)
- [x] `src/auto_mlx/graph.py`: strict fail-closed contracts + structural
      rules (unique ids, no dangling edges, endpoint-kind constraints,
      effect required on and exclusive to applied_result)
- [x] CLI: `auto-mlx validate graph` / `auto-mlx inspect graph` +
      `examples/evidence-graph.json`
- [x] Tests: round-trips, fail-closed cases, schema parity, packaging
- [x] `src/auto_mlx/advisor.py` with the full invariant set
- [x] Wired into `auto-mlx tune` (`--no-advice` opt-out); cited prunes land
      in prefilter.pruned; advice block in CLI result
- [x] Advisor invariant tests (9, each pinned to an invariant)
- [x] Full pytest suite green: 383 passed, 0 failed, 38 skipped
      (skips = sandbox/MLX-gated tests under the homebrew py3.14 runner)
- [x] Docs: docs/evidence-graph.md + README section, candid register
- [x] Git checkpoints: 6825c8e (graph), 58fa8dd (advisor), a7143a7 (docs)
- [x] Live end-to-end on this M4 Pro (python3.13 + MLX 0.31.2, real
      sandbox + attestation): cold tune 6 blocks/2 futile entrants/no
      winner; warm tune consulted 1 stored summary and emitted 2 demote
      advice entries each citing receipt_id + summary_id; no prune and no
      block savings — correct, since toy-matmul yields only inconclusive
      verdicts and inconclusive never prunes. Prune-path block savings are
      proven by deterministic fixtures (test_advisor.py); a live pruning
      demo requires an effectful workload (Phase 2).

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
