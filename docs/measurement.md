# Measurement contract

`auto-mlx evaluate` runs a real, receipt-backed local measurement on the local sandbox tier (macOS `sandbox-exec`; see `docs/threat-model.md`). Without the local sandbox execution primitives, evaluator execution stays unavailable: every external provider launch is rejected before invocation, and `evaluate` reports a stable `unavailable` diagnostic rather than a result. The evaluator retains raw rejected observations for offline diagnosis even then.

No general MLX speedup has been measured by this project. The one measurement that exists is the checked-in `toy-matmul` reference workload (`examples/workload.json`, driven by `auto_mlx.runners.reference_matmul`): on the Apple Silicon machine it was run on, the `mode="compiled"` candidate was not faster than the `mode="eager"` baseline, and `auto-mlx promote` correctly resolved that receipt to `native_fallback` (`gain_not_positive` under the pre-Wave-B bare-sign gate). That is one honest, receipt-backed data point for one toy workload, not a general claim, and every requirement below still applies to any future workload's evidence.

**Cross-process shader-cache confound (fixed).** The single compiled-vs-eager data point above predates a measurement-integrity remediation and was confounded in two independent, now-fixed ways:

1. `mx.compile`'s shader cache is a Metal/OS-level cache, not a Python-process-level one -- but every evaluator sample launches as a fresh subprocess. The first-ever compiled launch on a machine pays a real cold-start cost (~1151ms observed) versus ~140ms once the OS-level cache is warm; eager mode carries no equivalent cost. Because the evaluator always pins the baseline arm to eager, that cold tax landed asymmetrically on the compiled candidate arm alone across every sample -- a structural artifact of subprocess isolation, not a real baseline-vs-candidate difference. `auto_mlx.runners.reference_matmul` now performs one full, uncounted in-runner warmup execution of the measured computation before the timed run (see its module docstring), so both arms are measured warm; the evaluator records whether that warmup ran and whether the sample was the first launch of its config, per sample, as a non-evidentiary `warm_state` note.
2. The evidentiary timed span used to run from before process launch until after the isolation authority's out-of-band verification (three `sandbox-exec` probe subprocesses, ~60ms) had already executed, so that probe overhead was baked into every sample's measured duration. `execute_plan` now times only the runner subprocess's own launch-to-exit span (`ExecutionRecord.runner_elapsed_ns`) and defers verification until after that span closes; every receipt's gain math reads this narrower, evidentiary span, never the wider full-sample span (`parent_elapsed_ns`, retained only as a diagnostic).

The recorded data point predates both fixes and should be read as evidence that this class of confound exists, not as a reproducible verdict on compiled-vs-eager performance.

## Wave B: statistically sound accept/reject decisions

A bare sign check on a sum difference (`baseline_sum_ns - candidate_sum_ns > 0`) is not a decision Apple Silicon's measurement noise supports: there is no frequency lock or core pinning available to a local, unprivileged evaluator. `auto_mlx.statistics` replaces it with:

- **In-runner K-repetition.** After its one uncounted warmup, the runner performs `policy.k_repetitions` (default 50) additional eval-fenced timed iterations, reporting the per-iteration array on a non-oracle stderr channel. The parent-observed `runner_elapsed_ns` remains the sole evidentiary timing anchor: a reported array is trusted only if its sum does not exceed `runner_elapsed_ns` (plus a 2ms clock-skew tolerance) and no individual iteration is implausibly small (below 1 microsecond -- no real Python/MLX call completes faster). A report that fails either check degrades the sample to K=1 semantics (the parent span alone); a per-sample point estimate is then min-of-K when trusted, min-of-K's absence otherwise.
- **BCa bootstrap CI.** Per-block paired differences (baseline point estimate minus candidate point estimate) feed a bias-corrected-and-accelerated bootstrap (pure `random`/`statistics.NormalDist`, no third-party dependency), producing a 95%-base confidence interval, Bonferroni-adjusted for the number of sequential peeks taken (`policy.max_measurement_runs - policy.measurement_runs + 1`, fixed before the first block is measured).
- **Sequential sampling.** Starts at `policy.measurement_runs` blocks; while the verdict stays `inconclusive`, extends one block at a time up to `policy.max_measurement_runs`, stopping early the moment a verdict becomes decisive.
- **Three-way verdict.** `improved` / `regressed` / `inconclusive`, gated on a minimum-effect threshold (`policy.min_effect_bps`, default 200 = 2% of the baseline point estimate) -- the whole CI must clear the threshold, not just the point estimate. `inconclusive` is a first-class outcome in receipts, promotion (`auto_mlx.promotion`), and dispatch (`auto_mlx.dispatch`); it never rounds to a win or a loss, and a receipt whose statistics are missing or malformed fails closed to not-promotable.
- **A/A calibration.** `auto-mlx evaluate --calibrate` forces `policy.calibration=True` and runs the baseline runner on both arms, measuring the workload's real noise floor instead of asserting a threshold. Calibration receipts are valid evidence but are never promotable.

Two real `--calibrate` runs against `toy-matmul` on the Apple Silicon machine this was validated on (`--samples 3`, reduced `k_repetitions`/`bootstrap_resamples` from the production defaults to keep the run fast) measured a 95%-ish CI noise floor of roughly ±2-7ms around a ~23-25ms baseline point estimate -- both correctly resolved to `inconclusive` (no false positive), but the measured floor is noticeably wider than the 2% default `min_effect_ns` (~0.5ms) computes to on this small, short-running toy workload. This does not mean the statistics are wrong -- the CI-must-fully-clear-the-threshold rule is exactly what kept an A/A run from reporting a spurious `improved`/`regressed` verdict -- but it is a concrete data point that `min_effect_bps=200` may be underpowered for very short workloads at reduced K/block-count settings, consistent with the original proposal's own note that toy and LLM-decode workloads may want different thresholds. Re-calibrate with production `k_repetitions`/`max_measurement_runs` before trusting 2% as workload-agnostic.

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
