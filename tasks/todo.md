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

## Phase 1b — knowledge repo re-architecture (sub-agent, parallel) — DONE 2026-08-18

Delivered on branch `graph-v2` of `/Users/amal/Downloads/mlx-porting-skill`
(commits 97704b8, 887ea11, 4183205, 9a5bae9), independently verified from
this session:

- [x] `mlx-model-porting/graph/`: schema copy, 16 shards (core/mechanisms/
      models/campaigns/migrated/ecosystem), stdlib-only validate_graph.py +
      compile_graph.py + validate_pack.py + render_graph_summary.py +
      migrate_v1.py
- [x] Compiled graph: 319 nodes / 575 edges (64 mechanisms, 37 traits,
      38 applied_results — 15 regressed / 5 improved / rest inconclusive —
      4 models, 3 hardware, 124 external references). v1 migration was
      deliberately selective (contributor-candidate noise dropped, allowed
      by brief).
- [x] Verified: their validator OK; compile byte-deterministic (identical
      sha256 across two runs); the compiled graph passes this repo's strict
      `auto-mlx validate graph`; scrub scan clean (no ykn_ tokens, no
      api keys, no /Users paths).
- [x] Founding negative knowledge present, e.g. the head-step-cost response
      surface (0.14 → −385bp, 0.18 → −132bp, 0.24 → −151bp), GH=2 grouped
      SDPA −257bp, compiled MTP front −408bp — each cited to public PRs.
- [x] Evidence-pack format + example pack + docs/evidence-packs.md + CI gates
- [x] MECHANISM_INDEX.md rendering generated from the compiled graph

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

Phase 1 + 1b complete, 2026-08-18.

- auto-mlx: 5 commits on main (6825c8e graph contract, 58fa8dd advisor,
  a7143a7 docs, 8ce8eca todo, 12573f9 contract tightening). Full suite
  389 passed / 0 failed / 38 skipped (sandbox+MLX gated under py3.14
  runner; live loop verified separately under python3.13 + MLX 0.31.2).
- Contract tightening adopted from the sub-agent's integration review:
  effect required-on/exclusive-to applied_result in the JSON Schema too;
  reification cardinality (exactly-one applied_on/measured_on/
  under_workload, ≥1 instantiates); id prefix↔kind table; evidence
  minItems 1; optional effect.metric_direction with verdict-sign
  consistency; caps mirrored into the JSON Schema. The knowledge repo's
  319-node compiled graph passes the stricter contract unchanged.
- Knowledge repo (mlx-porting-skill) branch graph-v2: 16 shards,
  319 nodes / 575 edges, packs format + CI, 581/581 tests in a clean
  worktree. NOTE for user: graph-v2 was cut from fix/audit-remediation
  (repo HEAD at the time), not main; merging graph-v2 brings that branch's
  work along. Sub-agent findings worth keeping: E014 byte-identical rerun
  spread ±130bp is the official-score noise floor and 18/38 campaign
  results are honestly inconclusive under it; two promoted mechanisms
  (+3/+9bp) are real promotions with unresolved effects; Huffman/2-bit
  head lanes have ZERO official evidence (all six submissions failed
  unscored) and are recorded as unresolved, not negative.
- Follow-up queued for knowledge repo: re-copy updated schema from
  auto-mlx into graph/schema/ (asked of sub-agent 2026-08-18).

Next: Phase 2 — auto-mlx fingerprint (HF config → trait vector),
transfer prediction from the founding graph, model #2 selection, first
confirmed transfer with a fresh receipt. mlx-lm 0.31.3 + MLX 0.31.2
already installed under python3.13 on this host.
