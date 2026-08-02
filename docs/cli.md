# CLI reference

The CLI is standard-library Python and does not import MLX.

## Available commands

```text
auto-mlx validate KIND --input FILE [--workload FILE] [--artifact-root DIR] [--output FILE]
auto-mlx inspect KIND --input FILE [--workload FILE] [--artifact-root DIR]
```

`KIND` is one of `artifact`, `candidate`, `knob`, `policy`, `provider`, `receipt`, `runtime`, `workload`, or `document`. A positional path is accepted in place of `--input`. Candidate validation needs the separate workload document because candidate identity is recomputed from that workload.

`validate` returns the canonical document and its SHA-256. `inspect` returns only identity fields: workload/candidate/runtime/provider/receipt IDs where the contract defines them, or `document_id` for a generic document. Receipt inspection parses the checked-in receipt contract and independently recomputes its stored fields. Evaluator, receipt, promotion, and dispatch libraries exist, but production evaluation/activation remains fail-closed without the required proof.

`--artifact-root` verifies every artifact declared by a workload against local bytes. `--output` creates a canonical document with exclusive creation and never overwrites an existing path.

Each JSON file or stdin document is bounded to 4 MiB (`4,194,304` bytes) before parsing or accumulation. Larger input returns exit code 3 with the stable `input_too_large` diagnostic and no JSON result on stdout.

## Deferred commands

`evaluate`, `promote`, and `dispatch` are intentionally recognized and return exit code 4 with a JSON `unavailable` diagnostic on stderr. This defers only CLI orchestration, not the underlying libraries. The diagnostic marks `evaluate` as G1 and `promote`/`dispatch` as G2, with `surface: cli_orchestration`.

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | Success |
| 2 | Usage or argument error |
| 3 | Contract or integrity failure |
| 4 | Recognized but deferred command |
| 5 | Input/output failure |
| 70 | Unexpected internal failure |
