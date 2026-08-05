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

The provider configurations are proposals for the declared knobs. Binding one configuration to the workload (via `CandidateProposal`) derives a candidate ID; no candidate ID is stored in the provider document itself.

The remaining examples are all consistent with `workload.json`'s `toy-matmul` workload: `knob.json` is the same `tile` knob the workload declares, and `candidate.json` binds the `declarative-example` provider to that workload with a `mode`/`tile` config drawn from `provider.json`'s second configuration -- its `workload_hash` and `candidate_id` are derived, not hand-written, so any change to `workload.json` must be followed by regenerating `candidate.json` (for example, via `CandidateProposal(...).to_dict()`) or `auto-mlx validate candidate` will reject it as a mismatch. `artifact.json`, `policy.json`, and `runtime.json` are standalone documents for their kinds and do not reference the workload.

## Running `toy-matmul` for real

`workload.json`'s `mode`/`tile` knobs and `parameters` (`dtype: "float32"`, `shape: [1, 3072, 3072]`) are not just descriptive here -- they match, by hand-kept convention, the contract hardcoded into `auto_mlx.runners.reference_matmul`, a standalone, MLX-only-inside-its-functions script that actually builds two deterministic `3072x3072` float32 matrices and matmuls them, either eagerly or under `mx.compile`. `tile` is validated (must be an integer in `[16, 32]`) but intentionally left uncomputed -- see that module's docstring for why. `auto_mlx.runners.register_reference_matmul_runners(registry)` registers it into a `TrustedRunnerRegistry` as a baseline (pinned to `mode=eager` via an immutable `--force-mode=eager` argv flag, since an `Evaluator` binds one shared config to both its baseline and candidate execution plans) and a candidate runner (uses the proposal's own `mode`). `tests/test_runner_reference.py` exercises this end to end -- real sandboxed subprocess executions, not a mock -- on any host with MLX and the local `sandbox-exec` sandbox tier available. There is still no CLI wiring for any of this; that is a later wave.
