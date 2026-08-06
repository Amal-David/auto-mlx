# How Auto MLX decides whether a candidate is faster

This page explains the measurement and search design: what problem it solves, what we
found when we looked, which ideas came from which prior system, and what is still
missing. It is a design document. Every number in it that describes *this* project was
measured locally on the host named beside it; every reference to another system is an
architecture lesson, not evidence about MLX or Apple Silicon (see
[the research landscape](research-landscape.md) for that boundary rule).

## The problem

An earlier revision of this tool compared one candidate against one baseline and decided
by the sign of a difference of summed wall-clock times. Three consecutive runs of the
identical comparison on the same machine produced:

| Run | Reported delta | Would have concluded |
| --- | --- | --- |
| 1 | -181 ms | candidate slower |
| 2 | +480 ms | candidate faster |
| 3 | +218 ms | candidate faster |

Same code, same host, same configuration, three different answers, including a sign
change. Any promotion decision built on that is a coin flip wearing a lab coat.

## What we found

Three separate causes, two of them defects in this project.

**1. The stopwatch was measuring the wrong thing.** The evidentiary timed span ran from
before the runner subprocess launched to after it exited — and the isolation authority's
three sandbox verification probes ran *inside* that window. Every sample carried ~72–77 ms
of verification and staging cost. On a workload whose actual compute is ~19 ms, the tool
was measuring roughly four parts overhead to one part signal.

Fixed by timing only the runner subprocess lifetime and running verification after it
exits. The authority's evidence is unchanged: it recovers the sandbox profile from the
process's own immutable arguments and launches its own probe subprocesses under that same
profile, so probing after exit produces byte-identical evidence to probing during.
Both spans are recorded — the runner span is evidentiary, the full span is diagnostic.

**2. A cold start that landed on only one arm.** `mx.compile` populates a Metal shader
cache that persists *across processes*. A first-ever compiled subprocess took ~1151 ms
against ~140 ms warm. Because the baseline arm is pinned to `mode=eager` and only the
candidate arm ever compiles, that one-time cost was charged entirely to the candidate.
This is a structural confound, not jitter: it biases in a fixed direction.

Fixed by having the runner perform one uncounted in-process warmup execution before the
measured one, so compilation and cache population are paid before the stopwatch starts.
Receipts record whether that warmup marker was observed and whether the sample was the
first launch of its configuration in the run.

**3. The platform genuinely cannot be pinned.** macOS exposes no equivalent of
`taskset`/`sched_setaffinity` for P-core versus E-core placement, and no public API to
lock CPU or GPU clocks the way `nvidia-smi -lgc` can. `thread_policy_set`'s affinity
policy is documented by Apple as a cache-locality *hint*, not an assignment. Quality-of-
service classes and `taskpolicy` influence scheduling priority, and `pmset -g therm`
reports thermal pressure without root, but nothing available to userspace makes two runs
of the same code land in the same clock state.

That is not a bug to fix — it is the operating envelope. It means statistics have to do
the work that hardware pinning does on other platforms.

## The measurement design

**One sandboxed launch yields many timings.** Because per-launch cost (interpreter start,
MLX import, sandbox setup) dwarfs the computation, the runner performs K eval-fenced
iterations in-process after its uncounted warmup and reports each iteration's duration.
Default K is 50. This turns roughly 250 ms of mostly-overhead into a ~19 ms measurement of
the thing we actually care about.

**Self-reported timings are never evidentiary.** The runner's iteration array is data the
runner produced about itself, so it is cross-checked against the parent's own wall-clock
span for that subprocess. Implausible timings — a reported sum exceeding the parent span,
or absurdly small — raise a typed `forged_timing` rejection mirroring the existing
forged-oracle pattern, and the sample degrades to parent-span-only semantics rather than
being quietly believed. The full raw iteration array is stored in the receipt so the
supervisor can recompute any statistic independently.

**Per-sample estimate is the minimum of K.** Timing noise on a busy machine is one-sided:
interference can only make a run slower, never faster than the hardware permits. The
minimum is therefore the least-contaminated estimator of steady-state cost. This follows
pyperf, hyperfine and Criterion convention; TVM is a deliberate counterexample that
averages its repeats, which is why the raw array is retained rather than only the reduction.

**Pairing and order balance.** Baseline and candidate are measured in alternating ABBA /
BAAB blocks so that monotonic drift — thermal ramp, background load ramp — falls on both
arms rather than accumulating on whichever ran second.

**Thermal state is recorded, not assumed.** Each block reads `pmset -g therm` (no root
required) before measuring, retries once with a bounded pause if throttled, and annotates
the block as thermally suspect rather than silently pooling it with clean data. A parse
failure or missing tool yields `unknown`, never a crash.

## The decision rule

A verdict comes from a bias-corrected accelerated (BCa) bootstrap confidence interval over
the paired per-block differences, not from the sign of a total.

- **Three verdicts, not two.** `improved` when the interval's lower bound clears the
  min-effect threshold, `regressed` when its upper bound falls below the negative
  threshold, and **`inconclusive`** when the interval straddles. That third verdict is the
  one the old design could not express: `gain_not_positive` conflated "measurably slower"
  with "we cannot tell," and those warrant different responses.
- **Reproducible by construction.** The bootstrap seed, resample count, interval bounds
  and verdict are all recorded in the receipt, so the supervisor recomputes the identical
  interval from the stored raw samples during attestation and refuses on mismatch.
- **Sequential extension.** Sampling starts at the policy's block count and extends
  block-by-block while the verdict is inconclusive, up to a cap, stopping early once the
  interval resolves. Because this peeks repeatedly, the interval is widened as peeks
  accumulate.
- **A minimum effect size.** An effect smaller than the threshold (default 2%) is never
  promoted even if the interval excludes zero, because a statistically detectable
  difference that small is not worth a dispatch change.

**The threshold is measured, not asserted.** `evaluate --calibrate` runs an A/A comparison
— candidate configuration identical to baseline, so the true difference is zero by
construction — through the exact same machinery, and produces a receipt explicitly marked
as calibration and refused by promotion. On an Apple M4 Pro with MLX 0.32.0 it returns
`inconclusive` with an interval of `[-30082, +87302]` ns against a 19.22 ms baseline,
roughly ±0.45%. The 2% default therefore sits about four times above the measured floor.

That same A/A run is the clearest demonstration of why any of this matters. On identical
configurations, where the honest answer is "no difference":

| Gate | Verdict on identical configurations |
| --- | --- |
| Sign of the summed difference | `improved: true` — a 150 ms "win" |
| Bootstrap confidence interval | `inconclusive` |

## The search design

`auto-mlx tune` races a declarative provider's knob grid. Candidates remain inert scalar
data throughout; a candidate never selects a command, a runner, or code.

- **The baseline is a permanent, unremovable entrant.** A race concludes "keep baseline"
  unless a candidate reaches a decisive `improved` verdict. There is no configuration in
  which search can remove the fallback floor.
- **Elimination requires evidence.** A candidate leaves the race only as
  `eliminated_futile` — its recomputed interval's upper bound is already below the
  min-effect threshold, so no additional measurement could make it a winner — or as
  `decisive_regressed`. A single unlucky delta never eliminates anything.
- **Each rung is self-contained.** A racing rung is an ordinary evaluation at
  `measurement_runs == max_measurement_runs`, so every rung's receipt stops exactly at its
  own cap and the "inconclusive is only legitimate at the cap" rule holds unchanged.
- **Every measured candidate gets a full attested receipt.** There is no shortcut evidence
  lane for search results.
- **Budgets are honored strictly**, and the summary reports whether the budget ran out
  with candidates still unresolved rather than presenting a partial race as complete.
- **Results are reusable but not portable.** A content-addressed tuning summary records
  ranked entrants, intervals, elimination reasons, budget accounting, and both the
  workload hash and runtime identity. `auto-mlx history` reads them back, and a later run
  seeds its ordering from a prior winner only when *both* identities match.

## Where each idea came from

Prior art below informed the architecture. None of it is evidence about MLX or Apple
Silicon; every mechanism still had to be re-measured locally.

| Source | What we took | Status |
| --- | --- | --- |
| Triton `do_bench` / autotuner | Fill a time budget with an adaptively derived iteration count instead of fixing counts; select on median-first quantiles rather than a mean | Adapted |
| irace (F-Race) | Eliminate on a statistical test applied repeatedly as evidence accrues, not on one comparison | Adapted |
| SMAC intensification | Dethrone an incumbent only at matched evaluation depth, comparing on shared conditions | Adapted |
| XLA autotuner | Discard wrong-output candidates before they ever enter the timing comparison; key any cached tuning result on device *and* toolchain identity, and treat invalidation as the operator's responsibility | Adopted |
| Halide (Adams2019) | Always keep a floor entrant so search can never do worse than not searching; funnel many candidates through cheap filtering into few real measurements | Adopted (floor), partially (funnel) |
| MetaSchedule | Separate workload identity from tuning records in storage so history is queryable per workload; verify legality before spending a measurement | Adopted |
| Kalibera & Jones (ISMM 2013) | Decide *how many* repetitions are enough from the observed variance, rather than a fixed count; build an interval on the ratio rather than testing a raw difference | Adapted (sequential extension). Sourcing for their specific repetition counts was secondary; treated as directional |
| Mytkowicz et al. (ASPLOS 2009) | Incidental environment differences can flip which configuration looks faster; balance ordering and control setup | Adopted (ABBA/BAAB, sanitized environment) |
| mlx-serve | Warmup as an explicit, named, recorded phase; position-balanced paired A/B discipline; an instant kill switch back to the known-good path for every optimization | Adopted |
| TVM AutoTVM / Ansor | `number`/`repeat`/`min_repeat_ms` protocol shape; cost-model-guided search over a large space | Protocol adapted; cost model deferred (our knob space is small enough to race exhaustively) |
| TorchInductor | — | **Rejected.** Its candidates are generated code, and its numerical correctness check is disabled by default. Both are incompatible with a model where candidates are inert data and an independent oracle is mandatory |
| AutoTVM / Ansor template & sketch generation | — | **Rejected.** Procedurally generated schedule structure is program synthesis, not a closed declarative knob space |

Two observations worth recording. First, neither Triton nor TorchInductor re-races a
winner or applies any significance test — they take a single benchmarking pass and an
argmin — so the noise-robustness ideas had to come from the racing and benchmarking-
statistics literature rather than from compiler autotuners. Second, of TVM's three
generations, AutoTVM and Ansor have been removed from the project's main branch and only
MetaSchedule survives; its distinguishing traits are explicit pipeline stages, replayable
traces, and a database that separates workload identity from measurements.

## What has actually been measured

On an Apple M4 Pro, MLX 0.32.0, against the checked-in `toy-matmul` reference workload:

- `mode="compiled"` versus `mode="eager"` is **statistically indistinguishable** — a 95%
  interval of roughly `[-52µs, +18µs]` on a ~19.2 ms per-iteration baseline. Earlier
  revisions of this tool reported the same comparison as a loss; that reading did not
  survive the two measurement fixes above.
- An A/A calibration measures the noise floor at roughly ±0.45% of baseline.
- A bounded race over the checked-in provider grid finished in ~47 s and produced **no
  winner**: both entrants were eliminated as futile and the baseline was kept.

That last result is correct rather than disappointing. The example workload's `tile` knob
is deliberately inert — real tiling would change float accumulation order and break the
byte-exact digest parity the oracle depends on — so the grid is effectively two
configurations wide, and the one axis that does vary has no measurable effect. A knob
space containing no real effect has no optimum, and the tool says so instead of
manufacturing one.

No general MLX speedup has been measured by this project.

## Known gaps

- **Per-sample cost is still dominated by process startup.** K-repetition amortizes it
  across iterations, but every sample still pays a full interpreter and MLX import. A
  persistent measurement worker that loads once and receives configurations as messages —
  the structure mlx-serve uses for inference — would remove it. MLX binds a GPU stream to
  the thread that first touches it, so such a worker must own that thread for its lifetime.
- **The only workload is a toy.** The knobs where real wins are expected — KV-cache
  quantization, speculative decoding depth, prefix caching, batching — only show effects on
  realistic prefill/decode workloads, and several only under concurrency or multi-turn
  reuse. Those are not implemented.
- **No cost model.** Justified while the knob space is small enough to race exhaustively;
  the tuning-summary schema records knob vectors alongside outcomes so a ranker can be
  trained later without a migration.
- **`EvaluationPolicy.racing` is currently unused.** It permits an early stop on futility
  and is tested and gated, but the rung-based search made it unnecessary. It is retained as
  an opt-in mechanism for a future incremental-extension search; see
  [the CLI reference](cli.md) for details.
