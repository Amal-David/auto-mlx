# Serving-engine survey: adopted contracts and implementation boundaries

Reviewed 2026-10-05. The companion `mlx-porting-skill` change owns the canonical
24-project / 14-method survey in `mlx-model-porting/assets/inference_engines.json`
and its generated `INFERENCE_ENGINE_SURVEY.md`. This document identifies what
Auto MLX actually implements from that research, rather than claiming to have
ported every engine feature.

## What changed in this implementation

### Declared comparison contexts

`auto-mlx compare-serving --baseline baseline.json --candidate candidate.json`
checks whether two declared benchmark cells can be compared. It requires the same
model bytes, tokenizer/processor, template/adapter, workload, sampling, quality
contract, hardware, measurement protocol, cache state/precision, load state,
concurrency, token limits and metric meaning. Engine names and commit revisions
may differ: those are the implementations being compared. Both sides must declare
that the shared quality gate passed.

This command does not run benchmarks, authenticate the declarations, compute a
speedup or promote an engine. Its result always includes `promotion_allowed=false`
and `speedup=null`. An incomparable but well-formed pair is a successful inspection
with `comparable=false`; malformed contracts produce the normal nonzero CLI error.
An upstream README number cannot be turned into a promoted receipt through this
command. The packaged JSON Schema and strict Python loader share a closed shape.

```bash
auto-mlx compare-serving \
  --baseline examples/serving-baseline.json \
  --candidate examples/serving-candidate.json
```

The checked-in examples are **synthetic declarations only**, not real measurements.
Their differing cache regimes deliberately make them incomparable. Replace all
identity digests with identities of the actual artifacts/contracts before use.
`physical_prefill_tokens_per_second` and `logical_prefill_tokens_per_second` are
different metrics: a cache restore cannot masquerade as a faster prefill kernel.
Likewise, process RSS is not Metal allocated bytes, and aggregate throughput is
not single-request decode throughput.

### State-cache identity prerequisite

`CacheNamespace` binds the target model, tokenizer, processor, template, adapter,
draft, cache layout, tenant namespace, ordered media identity and cache ABI into
one canonical identifier. Every dimension is explicit; changes invalidate the
identifier. The layout digest must cover cache precision, recurrent topology,
draft capture/window policy and any other state-affecting option. An absent adapter/processor/draft must be represented by an explicit
canonical absence digest rather than reusing some other field's value.

This is a **contract for future cache implementations**, not a KV cache. It does
not authorize a tenant, discover the user's tenant ID, copy state, implement
copy-on-write, persist tensors, restore recurrent/MTP history or prove parity.
The current inference preview still has no prefix or SSD cache.

### Truthful runtime capability reporting

The local server reports its implemented capability surface on `/health` and
completion metadata: one active compute request, no continuous batching, no
prefix/SSD cache, no weight offload, no speculation, raw-text completions, greedy
sampling, no tools and no multimodal route. Configured CPU/GPU is reported
separately from execution validation. Metadata does not certify a usable GPU.

Concurrent HTTP connection slots are not model batch width. The existing bounded
HTTP admission path now returns an explicit `503` JSON response and `Retry-After`
rather than silently resetting an over-budget connection. Compute admission
remains non-queued and single-request; a busy worker is not described as a batch.

### Parent-boundary phase metrics

Both ordinary and streaming completions retain worker prefill/generation/memory
metrics and add parent-observed `request_first_content_ns` and
`request_end_to_end_ns`. The exact boundary is declared in each response:
handler entry **after HTTP headers have been parsed**, through generation complete,
**before the final response write**. This is not client-observed network latency.
It includes request-body parsing, profile lookup and any lazy worker startup.
First content is the first nonempty text delta: an empty delta, heartbeat, ready
message or final EOS does not count. An empty completion has a null first-content
metric. Never compare this boundary to a different engine's client RTT or
prefill-only timestamp without a shared measurement protocol.

## Primary-source lessons behind the contracts

| Source snapshot | Lesson adopted here | Still not implemented here |
|---|---|---|
| [oMLX cache](https://github.com/jundot/omlx/blob/a435a3736071286fdb2b74f486b3505f23f3c194/omlx/cache/prefix_cache.py) | Cache topology and accepted-token boundary are part of identity/qualification. | Per-layer leases, recurrent checkpoints, hot/cold persistence. |
| [oMLX packaging](https://github.com/jundot/omlx/blob/a435a3736071286fdb2b74f486b3505f23f3c194/packaging/README.md) | A runnable environment includes interpreter, frameworks and native libraries. | Relocatable environment packaging or a fix for the existing evaluator staging defect. |
| [vllm-mlx scheduler](https://github.com/waybarrios/vllm-mlx/blob/80e7fdec7e8641c8f81b01ece2d7cc256348b76f/vllm_mlx/scheduler.py) | HTTP concurrency and actual model batching are distinct; cancellation must reach compute. | Continuous batching and interleaved token-budget prefill. |
| [dflash-mlx](https://github.com/bstnxbt/dflash-mlx/blob/60803233af4589e18588b9bacbb03880801c828a/README.md) | Physical, restored and apparent prefill counts must not be conflated. | DFlash verification, draft registry and recurrent tape-replay rollback. |
| [Swift runtime capabilities](https://github.com/osaurus-ai/vmlx-swift/blob/2b05d39a6f43c1bc8789112fc92baa426d84c833/Libraries/MLXLMCommon/ModelRuntimeCapabilitySnapshot.swift) | Unknown capability and parsed metadata are not runtime/model proof. | A Swift engine or new modality implementation. |
| [rMLX KV formats](https://github.com/Pushkinist/rMLX/blob/e5d967a5c065c7a602d9e171c7d41ac006b1adcc/docs/KV_QUANT.md) | Actual storage, including full-precision shadows, must be measured. | Any KV quantization codec. |
| [MTPLX](https://github.com/youssofal/MTPLX/blob/9882703f3105363ddc37eca9f97aa09a1d387112/README.md) | Keep the ordinary decoder as baseline; match sampling and quality across paths. | Native MTP or stochastic residual-correction implementation. |
| [mlx-serve acceptance modes](https://github.com/ddalcu/mlx-serve/blob/c84a5a6321c7bad40c50b2bc50f7a5f4510413ab/src/mtp_acceptance.zig) | Alternative acceptance modes must not inherit exact-mode quality claims. | Zig runtime, alternate acceptance rules or custom Metal kernels. |
| [Splash](https://github.com/incoai/splash/blob/35c828c34bf4e3718dc8064c061132db9e018385/README.md) | Model-specialized verification and packaged kernels are qualification candidates. | Its native runtime, DFlash2 models or precompiled kernel package. |

These are original contracts and summaries. No upstream implementation or model
weights are copied or vendored. A future code reuse must review the relevant
license/NOTICE and exact component scope independently.

## Next work, in dependency order

1. Resolve the existing isolated GPU execution and complete-runtime staging
   blockers. Probe the actual target workload; never widen the sandbox merely
   to obtain a passing run or label CPU inference as GPU validation.
2. Add workload-level reference/task quality and comparable parent-observed
   baseline/candidate measurements through the existing receipt/activation gate.
3. Add a small, precisely qualified hot prefix cache with immutable leases and
   topology-aware state; then bounded admission, fair chunked prefill and batching.
4. Add SSD persistence only after restore, corruption, privacy and memory gates.
5. Experiment with matched MTP/DFlash and shape-specific kernels only on a verified
   numerical baseline. Active-expert SSD and multi-Mac placement are separate
   capacity research, not ordinary cache toggles.

The draft native preview's two prior blockers remain open: sandboxed GPU compiler
cache writes and copied-interpreter relocation/ambient dependency resolution.
This change does not claim to fix them, complete native evaluation/promotion,
benchmark a third-party engine, or demonstrate an inference speedup.
