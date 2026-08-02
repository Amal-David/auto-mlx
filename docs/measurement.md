# Measurement contract

There is no local MLX benchmark in this repository. No MLX speedup has been measured. The G1 evaluator library exists, but production evaluation may fail closed until real isolation and supervisor proof are available; its CLI orchestration is deferred.

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

**Empirical evidence:** a reproducible receipt from the future evaluator with the metadata and correctness gates above. Until then, wording such as “faster,” “supported,” or “promoted” is not allowed.
