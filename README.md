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

Output creation uses a private unpredictable staging name and a descriptor-relative no-follow hard link. The final
destination is absent or a complete `0600` payload; it is never visible while being written and existing output paths
are never replaced. Ordinary pre-publication failures leave no final name. Post-publication failures attempt an
identity-checked rollback and explicitly report any remaining destination, private staging name, or uncertain durability.
Hosts without the required POSIX descriptor and hard-link primitives fail closed.

## Explicit non-capabilities

The evaluator, receipt, promotion, and dispatch libraries are implemented and exported from the package root. The evaluator retains raw observations but G0 rejects every external provider launch before invocation and reports `SANDBOX_UNAVAILABLE` until a checked-in supervisor and authority exist. The `evaluate`, `promote`, and `dispatch` command names remain explicit deferred CLI orchestration: they return `unavailable` and never imply success. Promotion and dispatch remain later G2 evidence/activation work.

No MLX speedup has been measured by this project. CLI orchestration, compiler search, and custom Metal kernels remain deferred. External MLX documentation and research are design inputs, not local benchmark evidence.

Read [the architecture](docs/architecture.md), [the threat model](docs/threat-model.md), [the measurement contract](docs/measurement.md), [evidence and promotion](docs/evidence-and-promotion.md), and [the research landscape](docs/research-landscape.md) before proposing a new lane.

## License

MIT. See [LICENSE](LICENSE).
