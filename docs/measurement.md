# Measurement contract

`auto-mlx evaluate` runs a real, receipt-backed local measurement on the local sandbox tier (macOS `sandbox-exec`; see `docs/threat-model.md`). Without the local sandbox execution primitives, evaluator execution stays unavailable: every external provider launch is rejected before invocation, and `evaluate` reports a stable `unavailable` diagnostic rather than a result. The evaluator retains raw rejected observations for offline diagnosis even then.

No general MLX speedup has been measured by this project. The one measurement that exists is the checked-in `toy-matmul` reference workload (`examples/workload.json`, driven by `auto_mlx.runners.reference_matmul`): on the Apple Silicon machine it was run on, the `mode="compiled"` candidate was not faster than the `mode="eager"` baseline, and `auto-mlx promote` correctly resolved that receipt to `native_fallback` (`gain_not_positive`). That is one honest, receipt-backed data point for one toy workload, not a general claim, and every requirement below still applies to any future workload's evidence.

**Cross-process shader-cache confound (fixed).** The single compiled-vs-eager data point above predates a measurement-integrity remediation and was confounded in two independent, now-fixed ways:

1. `mx.compile`'s shader cache is a Metal/OS-level cache, not a Python-process-level one -- but every evaluator sample launches as a fresh subprocess. The first-ever compiled launch on a machine pays a real cold-start cost (~1151ms observed) versus ~140ms once the OS-level cache is warm; eager mode carries no equivalent cost. Because the evaluator always pins the baseline arm to eager, that cold tax landed asymmetrically on the compiled candidate arm alone across every sample -- a structural artifact of subprocess isolation, not a real baseline-vs-candidate difference. `auto_mlx.runners.reference_matmul` now performs one full, uncounted in-runner warmup execution of the measured computation before the timed run (see its module docstring), so both arms are measured warm; the evaluator records whether that warmup ran and whether the sample was the first launch of its config, per sample, as a non-evidentiary `warm_state` note.
2. The evidentiary timed span used to run from before process launch until after the isolation authority's out-of-band verification (three `sandbox-exec` probe subprocesses, ~60ms) had already executed, so that probe overhead was baked into every sample's measured duration. `execute_plan` now times only the runner subprocess's own launch-to-exit span (`ExecutionRecord.runner_elapsed_ns`) and defers verification until after that span closes; every receipt's gain math reads this narrower, evidentiary span, never the wider full-sample span (`parent_elapsed_ns`, retained only as a diagnostic).

The recorded data point predates both fixes and should be read as evidence that this class of confound exists, not as a reproducible verdict on compiled-vs-eager performance. No receipt collected under the current executor exists yet.

## Required future record

A future evaluation receipt must identify, at minimum:

- the source repository, revision, model/artifact manifest, tokenizer or processor revision, and license;
- the frozen workload hash, candidate ID, provider ID, runtime identity, and evaluation-policy identity;
- Apple Silicon hardware, OS, Python, MLX, package, and compiler versions;
- the exact workload shape, dtype, prompt or fixture, seed, batch/context policy, and output limit;
- baseline and candidate warmups, every planned sample, timing boundary, compile/materialization state, memory observation, and thermal-pressure state at measurement time (no sudo, no frequency lock is available on Apple Silicon; a throttled block is annotated `thermally_suspect`, never silently pooled -- see `auto_mlx.thermal`);
- exact-output or task-quality result, failure code, and whether the sample set is complete.

The baseline and candidate must be paired under the same workload and runtime policy. A reported mean or percentile without the full sample set and identity metadata is incomplete evidence. Compiler and custom-kernel experiments must compare against the native MLX fallback and retain a rollback condition.

## What counts as evidence

**Architecture lesson:** MLX's lazy evaluation and compilation model can change where work is materialized and may motivate a hypothesis. This is not a result for Auto MLX.

**Empirical evidence:** a reproducible receipt from a future supervised evaluator with the metadata and correctness gates above. Caller-created authorities, attestations, or production-eligibility flags are not authentication. Until then, wording such as “faster,” “supported,” or “promoted” is not allowed.
