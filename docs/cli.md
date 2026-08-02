# CLI reference

The CLI is standard-library Python and does not import MLX.

## Available commands

```text
auto-mlx validate KIND [PATH | --input FILE] [--workload FILE] [--artifact-root DIR] [--output FILE]
auto-mlx inspect KIND [PATH | --input FILE] [--workload FILE] [--artifact-root DIR]
auto-mlx evaluate
auto-mlx promote
auto-mlx dispatch
```

`KIND` is one of `artifact`, `candidate`, `knob`, `policy`, `provider`, `receipt`, `runtime`, `workload`, or `document`. A positional path is accepted in place of `--input`; exactly one spelling is required. Irrelevant options and ambiguous long-option abbreviations are rejected as usage errors.

The exact option matrix is:

| Command | Kinds | `--workload` | `--artifact-root` | `--output` |
| --- | --- | --- | --- | --- |
| `validate` | every `KIND` | candidate only | workload, candidate, receipt | yes |
| `inspect` | every `KIND` | candidate only | workload, candidate, receipt | no |
| `evaluate`, `promote`, `dispatch` | no document kind | no | no | no |

For example, candidate inspection uses both context options explicitly when artifact bytes are part of the contract:

```text
auto-mlx inspect candidate --input candidate.json --workload workload.json --artifact-root artifacts/
```

`validate` returns the canonical document and its SHA-256. `inspect` returns only identity fields: workload/candidate/runtime/provider/receipt IDs where the contract defines them, or `document_id` for a generic document. Receipt inspection parses the checked-in receipt contract and independently recomputes its stored fields. Evaluator, receipt, promotion, and dispatch libraries exist, but production evaluation/activation remains fail-closed without the required proof.

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

## Deferred commands

`evaluate`, `promote`, and `dispatch` are intentionally recognized and return exit code 4 with a JSON `unavailable` diagnostic on stderr. This defers only CLI orchestration, not the underlying libraries. The diagnostic marks `evaluate` as G1 and `promote`/`dispatch` as G2, with `surface: cli_orchestration`.

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | Success |
| 2 | Usage or argument error |
| 3 | Contract or integrity failure |
| 4 | Recognized but deferred command |
| 5 | Filesystem, input-file, or output failure |
| 70 | Unexpected internal failure |
