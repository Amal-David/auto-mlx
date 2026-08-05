# CLI reference

The CLI is standard-library Python. `validate` and `inspect` never import MLX. `evaluate` and `dispatch --execute`
launch a checked-in trusted-runner subprocess that imports MLX inside itself, under the local sandbox tier
(`auto_mlx.sandbox`) -- see [the local evidence-gated loop](#the-local-evidence-gated-loop) below and
[the threat model](threat-model.md) for exactly what that tier does and does not guarantee.

## Available commands

```text
auto-mlx validate KIND [PATH | --input FILE] [--workload FILE] [--artifact-root DIR] [--output FILE]
auto-mlx inspect KIND [PATH | --input FILE] [--workload FILE] [--artifact-root DIR]
auto-mlx evaluate --workload FILE --candidate FILE [--policy FILE] [--runtime FILE] --artifact-root DIR
                   [--store DIR] [--key-dir DIR] [--samples N] [--warmup-runs N] [--timeout-seconds N] [--max-output-bytes N]
auto-mlx promote --receipt RECEIPT_ID [--store DIR] [--key-dir DIR] [--artifact-root DIR]
auto-mlx dispatch --workload FILE --candidate FILE [--policy FILE] [--runtime FILE] [--artifact-root DIR]
                   [--store DIR] [--key-dir DIR] [--execute]
auto-mlx rollback [--store DIR] [--key-dir DIR]
auto-mlx keys ensure [--key-dir DIR]
```

`KIND` is one of `artifact`, `candidate`, `knob`, `policy`, `provider`, `receipt`, `runtime`, `workload`, or `document`. A positional path is accepted in place of `--input`; exactly one spelling is required. Irrelevant options and ambiguous long-option abbreviations are rejected as usage errors.

The exact option matrix for `validate`/`inspect` is:

| Command | Kinds | `--workload` | `--artifact-root` | `--output` |
| --- | --- | --- | --- | --- |
| `validate` | every `KIND` | candidate only | workload, candidate, receipt | yes |
| `inspect` | every `KIND` | candidate only | workload, candidate, receipt | no |

For example, candidate inspection uses both context options explicitly when artifact bytes are part of the contract:

```text
auto-mlx inspect candidate --input candidate.json --workload workload.json --artifact-root artifacts/
```

`validate` returns the canonical document and its SHA-256. `inspect` returns only identity fields: workload/candidate/runtime/provider/receipt IDs where the contract defines them, or `document_id` for a generic document. Receipt inspection parses the checked-in receipt contract and independently recomputes its stored fields.

`--artifact-root` verifies every artifact declared by a workload against local bytes. Every path component for file
inputs, artifact roots, and outputs is walked from a stable anchor (the opened cwd for relative paths or the opened
filesystem anchor for absolute paths) with no-follow directory opens and `fstat`; symlink ancestors, missing or
non-directory parents, and special final inputs are rejected. The final input is opened once and read only from that
descriptor. FIFO, socket, directory, device, and symlink final inputs fail closed; `-` remains the stdin spelling.
The no-follow descriptor-relative stat is only an early classification aid: a regular-file decision is still confirmed
by the final `open`/`fstat` pair, and platform socket/special-file open errors are reported as regular-file failures.
This same component walk rejects symlink ancestors such as `/tmp` or `/var` when those names are symlinks on the host;
they are not resolved as a fallback.

`--output` opens the parent directory once, creates an unpredictable private staging name with `O_CREAT|O_EXCL|O_NOFOLLOW`
and `dir_fd`, writes all bytes, applies mode `0600`, fsyncs the staged descriptor, and verifies that the staging name
still names that descriptor. It publishes with `os.link` using the same source/destination directory descriptor,
`follow_symlinks=False`, and create-only semantics. It immediately verifies the published device/inode, fsyncs the
same directory, removes the private staging name, and fsyncs the directory again. The final name is therefore either
absent or a complete payload; an existing destination is never overwritten. Raw output arguments ending in `/` (or
the platform separator) are rejected before `Path` normalization.

Path-based input/output requires the POSIX-style `openat`/`O_NOFOLLOW`/`O_DIRECTORY`/`dir_fd`/`fstat`/`linkat`/`unlinkat`
capability set, plus `fchmod` and `fsync` for publication. Unsupported hosts return exit code 5 rather than falling
back to path-following access or `os.replace`. Ordinary failures before the hard link leave no final name. If a
post-link identity, write, fsync, cleanup, or close failure occurs, the CLI attempts an identity-checked rollback and
reports whether the destination or private staging name could not be removed or whether durability is uncertain. A
process crash may leave only an unpredictable private staging name, never a partial final name.

Each JSON file or stdin document is bounded to 4 MiB (`4,194,304` bytes) before parsing or accumulation. Larger input returns exit code 3 with the stable `input_too_large` diagnostic and no JSON result on stdout.

Filesystem and output failures return exit code 5 with an `io_error` diagnostic. File inputs must be regular files; `-` remains the supported stdin spelling. Successful commands write one canonical JSON object to stdout and no diagnostics. Broken stdout pipes are handled as output failures without a traceback.

## The local evidence-gated loop

`evaluate`, `promote`, `dispatch`, `rollback`, and `keys ensure` wire the evaluator, receipt, local supervisor
attestation, promotion, and dispatch libraries into a real command-line loop. They run only through the **local
sandbox tier** (`auto_mlx.sandbox`): macOS `sandbox-exec` (Seatbelt), independently probed by its own authority
before any evidence is trusted. This is a developer-grade guard for evaluating your own candidate on your own
Mac, not a hardened multi-tenant production sandbox -- see `docs/threat-model.md`. `evaluate` and
`dispatch --execute` require the local sandbox execution primitives (macOS, `sandbox-exec` on `PATH`,
descriptor-relative artifact access); without them they return exit code 4 with a JSON `unavailable` diagnostic
(`error.details.status == "unavailable"`, `surface: local_sandbox`), identical in kind to every other fail-closed
boundary in this CLI. `promote`, plain `dispatch` (without `--execute`), `rollback`, and `keys ensure` never need
the sandbox and always run.

### Workload → runner binding

Which trusted runner actually executes a workload is a CLI-owned, closed registry keyed by the workload's `name`
-- a candidate's config never selects a runner or command. Today the registry has exactly one entry: workloads
named `toy-matmul` (matching `examples/workload.json`'s contract -- `parameters.dtype == "float32"`,
`parameters.shape == [1, 3072, 3072]`, `mode`/`tile` knobs) run through `auto_mlx.runners.reference_matmul` via
`auto_mlx.runners.register_reference_matmul_runners`. Any other workload name returns a typed `provider_error`
diagnostic (exit code 3) rather than falling back to a generic runner.

### `evaluate`

```text
auto-mlx evaluate --workload FILE --candidate FILE [--policy FILE] [--runtime FILE] --artifact-root DIR
                   [--store DIR] [--key-dir DIR] [--samples N] [--warmup-runs N] [--timeout-seconds N] [--max-output-bytes N]
```

Loads and validates `--workload`/`--candidate` through the same contract layer `validate` uses. `--policy`
defaults to `EvaluationPolicy()` defaults when omitted; `--samples` (`measurement_runs`), `--warmup-runs`,
`--timeout-seconds`, and `--max-output-bytes` override individual fields on top of whatever `--policy` loaded.
`--runtime`, when given, must independently validate as a `runtime` document *and* match this host's actual
current runtime identity (`RuntimeIdentity.current()`) -- it can never be used to impersonate a different host;
omitted, the current runtime is used directly. `--artifact-root` is required: it is both the workload
artifact-verification root (as in `validate`/`inspect`) and the evaluator's own artifact root.

With the local sandbox primitives present, `evaluate`: resolves the workload's trusted runner(s); runs one real
baseline execution to derive the source-of-truth exact-output oracle (never a hardcoded literal); runs a full
`Evaluator.evaluate()` pass (warmups plus paired ABBA/BAAB measurement blocks) under `LocalSandboxProvider`/
`LocalSandboxAuthority`; builds a `Receipt` from the resulting evidence bundle; stores it in the receipt/decision
store (`--store`, see below); auto-ensures a local attestation key exists (`--key-dir`, generating one on first
use); and attempts local supervisor attestation (`auto_mlx.supervisor.attest_receipt`). The receipt is stored
regardless of its evidence status (`"complete"` or `"failed"`) -- `evaluate`'s job is to record real evidence, not
to pre-judge promotion. Output (canonical JSON, exit 0):

```json
{"ok":true,"command":"evaluate","receipt_id":"<sha256>","status":"complete","attested":true,
 "store":"<path>","candidate_id":"<sha256>","workload_hash":"<sha256>","workload_name":"toy-matmul",
 "baseline_runner_id":"reference-matmul-baseline","candidate_runner_id":"reference-matmul-candidate",
 "isolation_tier":"local-sandbox-exec","gain":{"baseline_sum_ns":...,"candidate_sum_ns":...,"delta_ns":...,
 "improved":false,"numerator":...,"denominator":...}}
```

If the local supervisor refuses attestation (incomplete or non-"complete" evidence), `attested` is `false` and an
`attestation_refusal` field carries the reason; `evaluate` still exits 0, since it successfully ran and recorded
real evidence. `gain.improved`/`gain.delta_ns` are recomputed by the receipt itself, never asserted by the CLI.
Two `evaluate` runs with identical inputs are content-addressed-stable: each produces its own immutable,
independently retrievable receipt (real measurement timing makes their `receipt_id`s differ; nothing is
overwritten or corrupted).

### `promote`

```text
auto-mlx promote --receipt RECEIPT_ID [--store DIR] [--key-dir DIR] [--artifact-root DIR]
```

Loads the stored receipt (`--receipt`), independently re-verifies and re-attests its evidence chain through the
local supervisor (never trusting any prior attestation), decides activation (`make_promotion_decision`), and
persists the resulting decision and pointer (`activate`). `--artifact-root` defaults to the current working
directory when omitted (used for activation-time artifact verification; the `toy-matmul` workload declares no
artifacts, so any existing directory works). Never needs the local sandbox. A missing or tampered receipt, or
invalid/missing key material at `--key-dir`, is a contract-level failure (exit code 3) -- `promote` never silently
mints a fresh key for a receipt it did not itself evaluate (see `auto_mlx.keys.load_attestation_key`, strict, not
`ensure_attestation_key`). Output (canonical JSON, exit 0):

```json
{"ok":true,"command":"promote","receipt_id":"<sha256>","action":"native_fallback","reason":"gain_not_positive",
 "decision_id":"<sha256>","attested":true,"current_decision_id":"native_fallback","store":"<path>",
 "gain":{...}}
```

`action` is `"activate"` only for a receipt whose independently recomputed gain is positive and well-formed;
otherwise it is `"native_fallback"` with a specific `reason` (for example `gain_not_positive`,
`supervisor_attestation_required`, or an `activation_rejected:<code>` from a failed activation-time check). This
is never a hardcoded outcome -- it is the receipt's own honest, recomputed evidence.

### `dispatch`

```text
auto-mlx dispatch --workload FILE --candidate FILE [--policy FILE] [--runtime FILE] [--artifact-root DIR]
                   [--store DIR] [--key-dir DIR] [--execute]
```

Loads and validates the evaluation context exactly like `evaluate` (`--artifact-root` defaults to the current
working directory). Matches the current activation decision's pointer against that exact context and reports
`candidate` or `native_fallback` -- never a candidate for a stale, mismatched, unattested, or tampered decision.
Missing or invalid key material at `--key-dir` degrades to `native_fallback` (dispatch is a safety boundary: it
never crashes on a missing key). Output (canonical JSON, exit 0):

```json
{"ok":true,"command":"dispatch","store":"<path>",
 "dispatch":{"schema":"auto_mlx.dispatch.v1","mode":"native_fallback","reason":"native_fallback_pointer",
 "candidate_id":null,"receipt_id":null,"decision_id":null,"created_at_ns":...,"dispatch_id":"<sha256>"}}
```

With `--execute` (requires the local sandbox primitives; `unavailable`/exit 4 otherwise), `dispatch` actually runs
the selected side -- the promoted candidate's runner in `candidate` mode, or the baseline runner (always
`mode=eager`, representing native behavior) in `native_fallback` mode -- and reports its output digest and
duration:

```json
{"...": "as above", "execution":{"mode":"native_fallback","runner_id":"reference-matmul-baseline",
 "digest":"<sha256 of stdout>","duration_ns":...}}
```

A `--execute` run that does not succeed (non-zero exit, crash, timeout, sandbox denial) is a contract-level
failure (exit code 3), never a silently reported success.

### `rollback`

```text
auto-mlx rollback [--store DIR] [--key-dir DIR]
```

Writes an immutable rollback decision and points dispatch at native code, regardless of the prior decision. Never
needs the local sandbox or a valid key; always succeeds against a reachable store. Output:

```json
{"ok":true,"command":"rollback","action":"native_fallback","reason":"operator_rollback",
 "decision_id":"<sha256>","current_decision_id":"native_fallback","store":"<path>"}
```

### `keys ensure`

```text
auto-mlx keys ensure [--key-dir DIR]
```

Creates the local attestation key if missing (`0700` directory, `0600` file), or reuses the existing one. Never
prints key bytes -- only the resolved directory/file path and a truncated SHA-256 fingerprint of the key (not
reversible to the key itself):

```json
{"ok":true,"command":"keys","subcommand":"ensure","key_dir":"<path>","key_path":"<path>/attestation.key",
 "fingerprint_sha256_16":"<16 hex chars>"}
```

### `--store` / `--key-dir` precedence

Both follow the same explicit-arg > environment-variable > default precedence, already implemented by
`auto_mlx.store_config`/`auto_mlx.keys` and simply exposed by these flags:

| Flag | Env var | Default |
| --- | --- | --- |
| `--store` | `AUTO_MLX_STORE` | `./auto-mlx-store` |
| `--key-dir` | `AUTO_MLX_KEY_DIR` | `~/.auto-mlx/keys` |

The receipt/decision store root and the attestation key directory are validated disjoint (neither may nest
inside the other); a conflicting pair fails closed with a `store_config_invalid` diagnostic.

No command in this CLI ever prints raw attestation key bytes, on success or failure.

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | Success |
| 2 | Usage or argument error |
| 3 | Contract or integrity failure (including a missing/tampered receipt, invalid key material, an unrecognized workload, or a failed `--execute` run) |
| 4 | The command requires the local sandbox execution primitives, which are unavailable on this host (`evaluate`, `dispatch --execute`) |
| 5 | Filesystem, input-file, or output failure |
| 70 | Unexpected internal failure |
