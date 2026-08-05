# Measurement contract

`auto-mlx evaluate` runs a real, receipt-backed local measurement on the local sandbox tier (macOS `sandbox-exec`; see `docs/threat-model.md`). Without the local sandbox execution primitives, evaluator execution stays unavailable: every external provider launch is rejected before invocation, and `evaluate` reports a stable `unavailable` diagnostic rather than a result. The evaluator retains raw rejected observations for offline diagnosis even then.

No general MLX speedup has been measured by this project. The one measurement that exists is the checked-in `toy-matmul` reference workload (`examples/workload.json`, driven by `auto_mlx.runners.reference_matmul`): on the Apple Silicon machine it was run on, the `mode="compiled"` candidate was not faster than the `mode="eager"` baseline, and `auto-mlx promote` correctly resolved that receipt to `native_fallback` (`gain_not_positive`). That is one honest, receipt-backed data point for one toy workload, not a general claim, and every requirement below still applies to any future workload's evidence.

## Required future record

A future evaluation receipt must identify, at minimum:

- the source repository, revision, model/artifact manifest, tokenizer or processor revision, and license;
- the frozen workload hash, candidate ID, provider ID, runtime identity, and evaluation-policy identity;
- Apple Silicon hardware, OS, Python, MLX, package, and compiler versions;
- the exact workload shape, dtype, prompt or fixture, seed, batch/context policy, and output limit;
- baseline and candidate warmups, every planned sample, timing boundary, compile/materialization state, and memory observation;
- exact-output or task-quality result, failure code, and whether the sample set is complete.

The baseline and candidate must be paired under the same workload and runtime policy. A reported mean or percentile without the full sample set and identity metadata is incomplete evidence. Compiler and custom-kernel experiments must compare against the native MLX fallback and retain a rollback condition.

## What counts as evidence

**Architecture lesson:** MLX's lazy evaluation and compilation model can change where work is materialized and may motivate a hypothesis. This is not a result for Auto MLX.

**Empirical evidence:** a reproducible receipt from a future supervised evaluator with the metadata and correctness gates above. Caller-created authorities, attestations, or production-eligibility flags are not authentication. Until then, wording such as “faster,” “supported,” or “promoted” is not allowed.
