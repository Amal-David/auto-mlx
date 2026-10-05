# Native MLX-LM inference preview

Status: **experimental; draft integration, not a production inference engine**.
The CPU native-generation path is implemented. Metal execution and the complete
native evaluate/promote/dispatch loop still have validation blockers described
below. Do not infer a GPU speedup or source-framework parity from this preview.

## What this adds

The optional adapter provides content-addressed model bundles, a fixed MLX-LM
runner, deterministic raw-text generation, a resident subprocess, and a bounded
localhost completions endpoint. The ordinary Auto MLX installation remains
stdlib-only. Importing the CLI, bundle reader, HTTP server, or adapter does not
import MLX or execute model code.

A bundle binds the exact local model, tokenizer, metadata, source declaration,
license declaration, immutable revision, explicit CPU/GPU device, context limit,
verification prompts, Python/OS/architecture, chip, memory and backend package
versions. Model inventory and package fingerprints are independently checked
before execution. Source and license strings are operator declarations, not an
automatic license audit or proof that a remote revision produced these bytes.

Only flat, ordinary local files are accepted. Symlinks, code files, unknown
entries, custom tokenizer classes, remote-code mappings, external metadata paths,
unknown contract fields and changed bytes are rejected. Native support is
intentionally restricted to `qwen2`, `qwen3`, and `llama` model types; this is not
an arbitrary-checkpoint converter. The native loader must still accept the exact
configuration and weights. Existing quantized checkpoints can be inspected; this
preview does not quantize weights itself or prove quantization quality.

## Three different verification claims

| Result | What it actually proves |
| --- | --- |
| `bundle prepare` / `bundle verify` | The local artifact inventory matches its content-addressed declaration. No model inference ran. |
| `bundle verify --execute` | Frozen native greedy fixtures repeated with identical token IDs and text on the named device/runtime. This is not Torch/source parity. |
| Existing evaluator receipts | A candidate must pass the evaluator-owned output oracle and statistical/attestation gates before promotion. Native end-to-end promotion remains a validation gate, not an advertised completed result. |

The runner is always eager (`mx.disable_compile()`), uses greedy sampling, and
exposes only the bounded `prefill_step_size` knob (128 through 4096). Its baseline
is fixed at 2048. Candidates cannot choose scripts, interpreter paths, model
paths, network options, tokenizers, samplers, or their own oracle.

## Install

From this checkout on Apple Silicon, install the optional dependencies in a
virtual environment. They are not installed by inspection or generation commands.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install '.[inference]'
```

The direct backend versions are pinned in the extra. The bundle records the
resolved package versions as well; modifying that environment requires a new
bundle and evaluation. This is not a fully hash-locked transitive deployment.

## Prepare a native checkpoint

Obtain and review a checkpoint separately. None of the commands below downloads
weights. This example records the immutable revision used for the local
Qwen2.5-0.5B-Instruct-4bit smoke test, not a floating `main` reference.

```bash
MODEL=/absolute/path/to/qwen2.5-0.5b-instruct-4bit
PY=.venv/bin/python

$PY -m auto_mlx bundle prepare \
  --model "$MODEL" \
  --source-id mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --revision a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
  --license apache-2.0 \
  --fixtures examples/native-fixtures.json \
  --context-limit 4096 --device cpu \
  --output native-bundle.json

$PY -m auto_mlx bundle inspect --bundle native-bundle.json
$PY -m auto_mlx bundle verify \
  --bundle native-bundle.json --model-root "$MODEL" \
  --execute --output native-verification.json
```

`--device cpu` is explicit above because that is the working validation lane.
The default device is GPU. A failed GPU request never silently becomes CPU.
Outputs are create-only: existing files are not overwritten. Use a new name for
an intentionally changed bundle or report. Descriptor-safe CLI paths must not
have symlink ancestors; on macOS prefer the real resolved path or a relative
path anchored at the checkout.

## Generate and serve

Generation and serving are dry-run unless `--execute` is supplied.

```bash
$PY -m auto_mlx generate \
  --bundle native-bundle.json --model-root "$MODEL" \
  --prompt 'The capital of France is' --max-tokens 8 --execute

$PY -m auto_mlx generate \
  --bundle native-bundle.json --model-root "$MODEL" \
  --prompt 'The capital of France is' --max-tokens 8 --stream --execute

$PY -m auto_mlx serve \
  --bundle native-bundle.json --model-root "$MODEL" \
  --port 8000 --execute
```

The generation stream is newline-delimited JSON. Serving emits a `server_ready`
JSON event with its bundle ID and bound port. It exposes `GET /health`,
`GET /v1/models`, and `POST /v1/completions`. Use the bundle ID from `/v1/models`
as `model`. A request has this shape:

```json
{"model":"<bundle-id>","prompt":"The capital of France is","max_tokens":8,"stream":true,"temperature":0}
```

This is a **raw-text completions subset**, not a full OpenAI API implementation.
There is no chat-template application, chat-completions route, tools, logprobs,
attachments, nonzero temperature, batching or remote binding. Temperature must
be omitted or integer zero; the canonical JSON protocol rejects floats.

The server only binds `127.0.0.1`. It rejects browser-origin requests, unexpected
Host headers, duplicate/malformed JSON, unsupported fields, transfer-encoded
bodies and oversized requests. It admits one generation at a time, with at most
four connection threads. It is a local developer server, not a hardened
multi-tenant HTTP gateway, and has no authentication suitable for exposure.

The worker holds the model between successful requests, but creates fresh
generation state for every prompt. A timeout, interrupted iterator, or observed
client disconnect terminates the worker process group and releases its model;
a later request can reload the verified snapshot. There is no cross-request
prefix/KV reuse. Shutdown closes the worker and private staged files.

Generation reports model-load duration separately from time to first token and
generation duration. GPU peak allocation and CPU process peak resident memory
are different metrics, identified by `memory_metric`; do not compare them as
though they were the same measurement. CPU peak resident memory is the worker's
lifetime high-water mark, not a resettable per-request allocation count.

## Connect a bundle to the existing evaluator

```bash
$PY -m auto_mlx bundle workload --bundle native-bundle.json \
  --output native-workload.json \
  --provider-output native-provider.json \
  --candidate-output native-candidate.json

$PY -m auto_mlx evaluate \
  --workload native-workload.json --candidate native-candidate.json \
  --policy examples/native-policy.json --artifact-root "$MODEL" \
  --store ./native-store --key-dir ./native-keys
```

The generated provider enumerates prefill sizes 128, 512 and 2048. The example
prompts are short smoke fixtures, not a representative optimization benchmark:
all may fit in one prefill chunk. Supply longer, task-representative prompts to
measure a real chunking tradeoff. Do not advertise a win from the example grid.

`evaluate`, `tune`, and `dispatch --execute` resolve a fixed, artifact-bound
runner for `mlx-lm-text-v1`. The same baseline output digest is the evaluator's
oracle; measurements exclude initial model loading and include each entire
fixture-generation loop. Promotion remains an explicit separate operation and
can correctly return no activation.

`generate` and `serve` can accept `--candidate`, its `--policy`, `--store`, and
`--key-dir`. They consult the existing independent dispatch decision for every
request. A candidate is eligible only for an **exact frozen fixture request**;
other prompts/context lengths use native settings. Missing, stale, invalid or
rolled-back activation also falls back to native settings. Advice is not
permission to apply a measured configuration to arbitrary prompts.

## Current blockers and limits

Two failures were reproduced during this integration and are deliberately not
hidden behind a fallback or broader host permissions:

* On the tested macOS host, sandboxed GPU generation requests a Metal compiler
  cache write that the existing sandbox denies. Eager execution also encounters
  the denial. CPU validation does not close this GPU gate.
* The existing executor stages a copy of the interpreter. A relocatable `uv`
  Python can then fail to locate its adjacent dynamic library. A staged Homebrew
  interpreter can discover ambient global packages outside the intended virtual
  environment; on this host that imported an unrelated Torch/dill stack and
  caused an additional denied `/dev/null` access. The direct resident-worker
  path uses the original virtual-environment interpreter and does not share
  that staging behavior. The native evaluator's full receipt/promotion loop
  remains unverified until interpreter/dependency isolation is resolved.

No sandbox cache exception is shipped. The native worker's file-descriptor cap
is 4096 because the pinned MLX-LM version sets that limit during import; network
and arbitrary external writes remain denied by the existing local sandbox.

Not implemented: source-framework oracle import, conversion orchestration,
quality metrics beyond exact native output, memory-budget admission, automatic
quantization, continuous batching, multi-tenant serving, custom kernels, or a
general promise that arbitrary checkpoints will load. These are follow-on gates,
not implied by bundle validity.

## Tests

The normal suite is offline and does not need MLX or model weights:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

The explicit integration lane never downloads a model:

```bash
AUTO_MLX_NATIVE_MODEL="$MODEL" AUTO_MLX_NATIVE_DEVICE=cpu \
  $PY -m unittest discover -s tests -p test_native_model_integration.py -v
```

It exercises real repeated generation, context overflow, cancellation/reload,
non-streaming completions and server-sent events. Setting the device to `gpu`
tests the Metal gate rather than substituting a CPU result.

## Serving observability update

[Engine-survey adoption](engine-survey-adoption.md) adds explicit capabilities,
bounded HTTP overload errors and parent handler-boundary latency metrics to both
JSON and streaming completions. These are not new batch/cache/speculation features
or fixes to the execution blockers described above.
