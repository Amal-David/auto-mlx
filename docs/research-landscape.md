# Research landscape and provenance

This page records architecture lessons separately from empirical evidence. All
non-MLX entries are held `research-candidate` prior art. The MLX/Apple entries
are framework/reference evidence informing G0, G1, and G3 boundaries, not local
results. Official documentation is recorded with a visible version when one is
shown; otherwise use `version not shown; accessed 2026-08-02`. None of these
sources establishes an Auto MLX speedup, MLX support, Apple Silicon
performance, safe execution, or promotion authority.

The staged boundary is:

- **G0:** declarative contracts, strict identities, artifact integrity, and
  default-deny input handling.
- **G1:** evaluator-owned isolated execution, an independent exact oracle, and
  complete paired baseline/candidate measurement.
- **G2:** immutable receipts, independent verification, authenticated
  promotion/rollback, and exact-key dispatch with native fallback.
- **G3:** compiler search and custom Metal; deferred until G0-G2 evidence gates
  pass on a pinned MLX/Apple workload.

| Primary source family | Architecture lesson | Hard evidence / transfer boundary |
| --- | --- | --- |
| [SIA paper](https://arxiv.org/html/2605.27276v2), [repository architecture](https://github.com/hexo-ai/sia/blob/main/docs/architecture.md), [evaluation guide](https://github.com/hexo-ai/sia/blob/main/EVALUATION_GUIDE.md), [security model](https://github.com/hexo-ai/sia/blob/main/SECURITY.md) | Separate Meta-Agent, Task-Specific Agent, and Feedback-Agent roles; preserve generation lineage and bounded execution-analysis-improvement feedback. | Copy the role separation and lineage only. Generated code, direct host execution, task-owned evaluators, network/secret access, and weight updates are outside the G0 trust boundary and cannot be promotion evidence. Reported results use SIA agents, tasks, H100/CUDA, and task verifiers; they are not MLX evidence. |
| [Self-Refine paper](https://arxiv.org/abs/2303.17651), [repository](https://github.com/madaan/self-refine) | A bounded generate-feedback-refine loop can use actionable diagnostics to propose the next declarative candidate. | Gains depend on task prompts, model error recognition, and task-specific feedback; refinement can regress. Feedback is advisory, not an oracle or promotion decision, and the reported language-task results do not transfer to MLX. |
| [Reflexion paper](https://arxiv.org/abs/2303.11366), [repository](https://github.com/noahshinn024/reflexion) | Keep Actor, Evaluator, and Self-Reflection distinct; retain bounded episodic history across trials. | Exact-match graders, heuristics, LLM evaluators, and generated tests are task-specific and can produce false positives. History may guide proposal search but cannot replace an immutable receipt, independent oracle, or MLX measurement. |
| [CRITIC paper](https://arxiv.org/abs/2305.11738), [Microsoft implementation](https://github.com/microsoft/ProphetNet/tree/master/CRITIC) | Self-correction should be grounded in external tools rather than the proposing model's self-assessment. | Tool feedback can be incomplete or misinterpreted and does not prove isolation, benchmark validity, or MLX performance. Auto MLX therefore keeps source-oracle comparison and timing outside the candidate generator. |
| [Darwin Gödel Machine paper](https://arxiv.org/abs/2505.22954), [repository](https://github.com/jennyzzt/dgm) | Maintain an archive/tree of candidate lineage, including non-monotonic paths, parent identities, and evaluated branches. | The repository executes model-generated code and its results use SWE-bench/Polyglot environments. An archive is diagnostic search state, not correctness authority, receipt authority, or deployment authority. |
| [SICA paper](https://arxiv.org/abs/2504.15228), [repository](https://github.com/MaximeRobeyns/self_improving_coding_agent) | Evaluate the current candidate, generate a successor, and score quality/cost/time with explicit oversight and retained events. | Docker, shell execution, hosted models, and a SWE-bench subset do not establish a safe G0 boundary or MLX result. Observability is not isolation, and a multi-objective score is not promotion evidence. |
| [AlphaEvolve paper](https://arxiv.org/abs/2506.13131), [DeepMind announcement](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) | Separate proposal generation, evaluator pools, evolutionary archive state, and parent selection; retain evaluated candidates rather than only the winner. | The linked sources provide no public implementation or independently reproducible evaluator infrastructure. Its architecture is prior art only; no reported result transfers to MLX or Apple Silicon without a fresh local receipt. |
| [FunSearch paper](https://www.nature.com/articles/s41586-023-06924-6), [repository](https://github.com/google-deepmind/funsearch) | Use correctness-filtered archives, parallel/concurrent evaluation, islands, and diversity preservation to keep search from collapsing on one candidate. | Correctness is defined by the supplied evaluator and inputs, not a general proof. The public repository omits the full LLM service and production sandbox; its results are not MLX evidence. |
| [AlphaDev paper](https://www.nature.com/articles/s41586-023-06004-9), [archived released pseudocode/tests](https://github.com/google-deepmind/alphadev) | Gate expensive performance measurement behind exact output correctness and distinguish proxy scores from target-device latency. | Fixed-size assembly programs, CPU-specific behavior, and small test domains do not transfer to MLX graphs, Metal kernels, or Apple Silicon. Proxy metrics never promote a candidate. |
| [AutoTVM paper](https://papers.neurips.cc/paper/7599-learning-to-optimize-tensor-programs.pdf), [pinned canonical TVM AutoTVM module](https://apache.googlesource.com/tvm/+/21095b98bb6e2d337b811a2bf306c7c031e746c5/python/tvm/autotvm/__init__.py) | Separate template-defined search spaces, cost models, device measurement, and durable history; use models to reduce expensive trials, not replace them. | Templates, features, and transfer are TVM- and target-specific. The pinned module is a source snapshot, not Auto MLX measurement or MLX evidence. Historical measurements may seed search only after workload, hardware, runtime, and identity compatibility are proven locally. |
| [Ansor paper](https://www.usenix.org/system/files/osdi20-zheng.pdf), [TVM overview](https://tvm.apache.org/2021/03/03/intro-auto-scheduler.html), [repository](https://github.com/apache/tvm) | Use hierarchical candidate generation, evolutionary/cost-model ranking, and task scheduling to spend budget on end-to-end bottlenecks. | The system assumes a TVM compiler stack and primarily static dense workloads on non-MLX targets. Its numbers do not transfer to MLX/Metal; only measured candidates can update local evidence. |
| [Tensor Program Optimization with Probabilistic Programs](https://arxiv.org/abs/2205.13603) (introduces MetaSchedule), [TVM RFC](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0005-meta-schedule-autotensorir.md), [documentation](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html) | Make space generation, search, cost model, builder, runner, scheduler, and database explicit; persist replayable traces and target identity. | TVM/TensorIR traces and cost models are not MLX artifacts. Reuse requires exact structural/workload/runtime compatibility and fresh correctness plus paired measurements. |
| [Triton autotuner API](https://triton-lang.org/main/python-api/generated/triton.autotune.html), [repository](https://github.com/triton-lang/triton) | Represent bounded configurations with workload keys, warmup/repetition rules, pruning, mutation reset/restore, and cache identity. | The cited Triton material is centered on Linux NVIDIA/AMD GPU backends; it is not MLX/Metal evidence. Its protocols are transferable; kernels, configurations, and timings are not. |
| [TileLang paper](https://arxiv.org/abs/2504.17577), [repository](https://github.com/tile-ai/tilelang), [merged Metal backend work](https://github.com/tile-ai/tilelang/pull/799) | Separate kernel dataflow from layout, tiling, thread binding, and pipeline choices; bind cache keys to source, version, configs, compile arguments, and profiling arguments. | The merged work is Torch/MPS-backed Metal work, not MLX evidence. Gaps remain around MLX benchmarking, add-kernel launch, a proper Metal/metallib backend, and macOS arm64 CI. Treat it as a future-provider lead only. |
| [egg paper](https://doi.org/10.1145/3434304), [repository](https://github.com/egraphs-good/egg) | Equality saturation separates equivalence-preserving exploration from extraction; e-class analyses can carry shape/semantic facts, while runner budgets bound graph growth. | User rewrites are not automatically sound, and extraction cost is not a correctness proof. Any future use must record rewrite-set identity, semantic domain, saturation limits, and an independent oracle result. |
| [TENSAT paper](https://proceedings.mlsys.org/paper_files/paper/2021/file/cc427d934a7f6c0663e5923f49eba531-Paper.pdf) | Tensor graph search needs shape/layout analysis, multi-pattern rewrites, and an explicit extraction policy distinct from semantic validation. | The reported measurements use NVIDIA T4/CUDA/cuDNN. The paper also discusses non-local costs from concurrent execution and cost-model/runtime mismatch. These hardware, backend, and modeling boundaries mean the result is not MLX evidence. |
| [Optuna paper](https://arxiv.org/abs/1907.10902), [repository](https://github.com/optuna/optuna), [storage/pruner docs](https://optuna.readthedocs.io/en/stable/reference/storages.html) | Separate configuration sampling, intermediate-result pruning, and resumable study storage; use pruning to allocate finite trials. | Optuna optimizes reported objectives. A pruned trial is incomplete evidence, storage is not an immutable receipt, and sampler state cannot establish exact correctness or activation. |
| [SMAC3 paper](https://www.jmlr.org/papers/v23/21-0888.html), [repository](https://github.com/automl/SMAC3), [documentation](https://automl.github.io/SMAC3/latest/3_getting_started/) | Make configuration space, scenario, target function, intensification, repeated seeds/instances, and run history explicit. | The target function returns a metric; racing and incumbent selection do not establish exact-output correctness, paired native comparison, provenance, or promotion. |
| [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html), [compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html), [custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html), [extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html), [Apple threadgroup guidance](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes) | Treat lazy materialization, first-call compilation, steady-state execution, shapes, dtypes, grid/threadgroup state, and native fallback as explicit measurement dimensions. | Official behavior and examples are documentation evidence, not an Auto MLX benchmark. A custom kernel remains G3 research until parity, fallback, pinned Apple/MLX metadata, complete paired samples, rollback, and an immutable receipt pass. |
| [in-toto paper](https://www.usenix.org/system/files/sec19-torres-arias.pdf), [in-toto specification](https://in-toto.io/docs/specs/), [SLSA v1.2 provenance](https://slsa.dev/spec/v1.2/build-provenance), [TUF specification](https://theupdateframework.github.io/specification/latest/), [reproducible-builds paper](https://arxiv.org/abs/2104.06020) | in-toto binds authorized actors, materials, and products; SLSA structures build provenance; TUF protects signed metadata with thresholds, freshness, versioning, and rollback. Auto MLX could compose these ideas around evaluator receipts and promotion pointers, but those require an Auto MLX schema. | These standards authenticate a declared process, not correctness or performance. SLSA build provenance does not model paired benchmarks, and TUF metadata does not judge speed. The evaluator-receipt and promotion-pointer schema, verifier, and freshness policy remain Auto MLX design work. |

## SIA structural reference

SIA is the clearest structural reference for Auto MLX, but not an execution or
promotion model to copy.

**Copy:** role separation between proposal/meta, task execution, and feedback;
generation-aware lineage; bounded feedback loops; and retention of full
trajectory/rationale artifacts for diagnosis.

**Reject:** generated code, direct host execution, task-owned evaluators,
network/secret access, and weight updates as promotion evidence. Auto MLX also
rejects candidate-controlled thresholds or callbacks under its G0
trust-boundary rule; that is an Auto MLX rule, not a claim about SIA behavior.
Auto MLX candidates remain declarative data;
the evaluator owns execution and the oracle; receipts are immutable; and only
independent promotion may affect dispatch. SIA's task-, verifier-, model-, and
H100/CUDA-specific results are not MLX or Apple Silicon evidence.

## Recommended architecture synthesis

The transferable control-flow pattern is:

```text
declarative proposal/search
    -> evaluator-owned isolated execution
    -> exact independent oracle
    -> complete ABBA/BAAB paired measurement
    -> immutable canonical receipt
    -> independently authenticated promotion
    -> exact-key dispatch / native fallback
```

Search models, feedback traces, archives, cost models, HPO state, e-graphs,
and signed provenance can inform the control plane, but none is itself a
correctness or performance claim. G0 remains offline and declarative; G1 must
prove the evaluator boundary and complete measurement; G2 must independently
verify receipts and promotion/rollback; G3 remains deferred compiler/custom-
Metal research until those gates pass.

## Source handling rule

Primary papers, project repositories, official MLX/Apple documentation, and
formal provenance specifications can provide architecture lessons. A number
becomes empirical evidence only when the exact experiment, environment,
baseline, sample set, correctness result, and artifact identity are
independently reproducible. Research collection must never auto-promote a
recommendation into a contract, runbook, receipt, or dispatch decision.
