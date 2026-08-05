# Auto MLX

Auto MLX is a small, standalone G0 contract layer for evidence-gated MLX tuning. It records declarative workloads and bounded configuration spaces without allowing a candidate to choose commands, code, evaluator logic, or its own identity.

## Implemented G0 capabilities

- Strict JSON parsing: duplicate keys, floating-point values, non-finite numbers, unknown fields, and malformed contract shapes fail closed.
- Canonical UTF-8 JSON and SHA-256 identities for workloads, candidates, providers, runtime descriptions, and generic documents.
- Immutable workload, artifact, knob, provider, policy, runtime, and candidate proposal contracts.
- Descriptor-relative, no-follow artifact verification with size and SHA-256 checks where the host exposes the required primitives.
- A standard-library-only CLI that validates documents and inspects IDs. It imports no MLX.
- Descriptor-stable, create-only output publication: `--output` stages privately, hard-links atomically, and refuses to overwrite an existing path.

## Install and validate

The package has no runtime dependencies. For a normal local installation from this directory:

```bash
python3 -m pip install .
auto-mlx validate workload --input examples/workload.json
auto-mlx inspect workload --input examples/workload.json
```

Editable installation is intended for development only:

```bash
python3 -m pip install -e .
```

Both successful commands emit one JSON object on stdout. Failures emit a JSON diagnostic on stderr and return a stable nonzero exit code. The same CLI is available as `python3 -m auto_mlx`.

The checked-in example is intentionally declarative and offline:

```bash
PYTHONPATH=src python3 -m auto_mlx validate provider --input examples/provider.json
PYTHONPATH=src python3 -m auto_mlx validate workload examples/workload.json --output ./auto-mlx-workload.json
```

`--output` uses a descriptor-relative, no-follow walk, so it only accepts a path anchored at the working directory or one with no symlinked ancestor component; an absolute path through a symlinked prefix (for example a `mktemp -d` result on macOS, where `/var` is a symlink) fails closed. `auto-mlx-workload.json` is listed in `.gitignore` so running this command from a checkout does not dirty `git status`.

Output creation uses a private unpredictable staging name and a descriptor-relative no-follow hard link. The final
destination is absent or a complete `0600` payload; it is never visible while being written and existing output paths
are never replaced. Ordinary pre-publication failures leave no final name. Post-publication failures attempt an
identity-checked rollback and explicitly report any remaining destination, private staging name, or uncertain durability.
Hosts without the required POSIX descriptor and hard-link primitives fail closed.

## The local evidence-gated loop

`auto-mlx evaluate`, `promote`, `dispatch`, `rollback`, and `keys ensure` wire the evaluator, receipt, local supervisor attestation, promotion, and dispatch libraries into a real command-line loop, gated on the **local sandbox tier** (`auto_mlx.sandbox`: macOS `sandbox-exec`, one Mac, one operator). `evaluate` runs a candidate against its workload's baseline under that sandbox, stores an immutable receipt, and attempts local supervisor attestation. `promote` independently re-verifies and re-attests a stored receipt and decides activation. `dispatch` resolves (and, with `--execute`, actually runs) the currently active candidate or the native fallback. `rollback` forces dispatch back to native. See [the CLI reference](docs/cli.md) for the full command surface.

This is explicitly the **local** tier only -- a single-user, developer-grade guard, not a hardened multi-tenant production sandbox (see [the threat model](docs/threat-model.md)). Without the local sandbox execution primitives (non-macOS, no `sandbox-exec` on `PATH`), `evaluate` and `dispatch --execute` fail closed with a stable `unavailable` diagnostic and exit code 4, identical in kind to the rest of this CLI's fail-closed boundaries. A future, hardened production activation gate (G3) is separate, later work and remains deferred.

Measurements now happen locally and are receipt-backed, not asserted. Running `evaluate` against this repository's checked-in `toy-matmul` reference workload (`examples/workload.json`, driven by `auto_mlx.runners.reference_matmul`) on Apple Silicon measured the `mode="compiled"` candidate as **not faster** than the `mode="eager"` baseline -- `promote` correctly resolves that receipt to `native_fallback` with reason `gain_not_positive`. That is the honest, current, receipt-backed result for that one toy workload on the machine it was run on; it is not a general MLX speedup claim, and no other workload has been measured. Compiler search and custom Metal kernels (G3) remain deferred. External MLX documentation and research are design inputs, not local benchmark evidence.

Read [the architecture](docs/architecture.md), [the CLI reference](docs/cli.md), [the threat model](docs/threat-model.md), [the measurement contract](docs/measurement.md), [evidence and promotion](docs/evidence-and-promotion.md), and [the research landscape](docs/research-landscape.md) before proposing a new lane.

## License

MIT. See [LICENSE](LICENSE).
