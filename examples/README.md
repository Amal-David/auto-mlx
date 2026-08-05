# Examples

These examples are data only. They do not download a model, execute a candidate, or require MLX. Every file validates against its schema-backed document kind:

```bash
PYTHONPATH=src python3 -m auto_mlx validate workload --input examples/workload.json
PYTHONPATH=src python3 -m auto_mlx inspect provider --input examples/provider.json
PYTHONPATH=src python3 -m auto_mlx validate artifact --input examples/artifact.json
PYTHONPATH=src python3 -m auto_mlx validate candidate --input examples/candidate.json --workload examples/workload.json
PYTHONPATH=src python3 -m auto_mlx validate knob --input examples/knob.json
PYTHONPATH=src python3 -m auto_mlx validate policy --input examples/policy.json
PYTHONPATH=src python3 -m auto_mlx validate runtime --input examples/runtime.json
```

The provider configurations are proposals for the declared knobs. A future evaluator would derive candidate IDs only after binding one configuration to the workload; no candidate ID is stored in the provider document.

The remaining examples are all consistent with `workload.json`'s `toy-matmul` workload: `knob.json` is the same `tile` knob the workload declares, and `candidate.json` binds the `declarative-example` provider to that workload with a `mode`/`tile` config drawn from `provider.json`'s second configuration -- its `workload_hash` and `candidate_id` are derived, not hand-written, so any change to `workload.json` must be followed by regenerating `candidate.json` (for example, via `CandidateProposal(...).to_dict()`) or `auto-mlx validate candidate` will reject it as a mismatch. `artifact.json`, `policy.json`, and `runtime.json` are standalone documents for their kinds and do not reference the workload.
