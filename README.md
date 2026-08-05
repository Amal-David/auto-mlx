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

Measurements happen locally and are receipt-backed, not asserted, and decisions come from a confidence interval rather than the sign of a single difference. Every sample is the minimum of K in-process iterations timed inside one sandboxed runner launch; verdicts come from a bootstrap CI over paired baseline-vs-candidate blocks, extended sequentially until the interval resolves or a cap is reached. There are three verdicts, and **`inconclusive` is a first-class one** -- the tool distinguishes "measurably slower" from "not resolvable at this noise floor" instead of collapsing both into a loss.

Running that machinery against this repository's checked-in `toy-matmul` reference workload (`examples/workload.json`, driven by `auto_mlx.runners.reference_matmul`) on an Apple M4 Pro (MLX 0.32.0) found the `mode="compiled"` candidate **statistically indistinguishable** from the `mode="eager"` baseline: a 95% CI of roughly `[-52us, +18us]` against a ~19.2 ms per-iteration baseline, straddling zero and well inside the 2% min-effect threshold. An earlier, noisier revision of this tool reported that same comparison as a loss; that reading did not survive the measurement fixes (verification-probe time counted inside the timed span, and a cross-process Metal shader-cache cold start landing asymmetrically on the candidate arm). The honest current result is *no detectable difference on this workload at this host's noise floor* -- neither a win nor a loss.

An A/A calibration run (`evaluate --calibrate`, candidate configuration identical to baseline) measures that noise floor directly and produces a deliberately unpromotable receipt. On this host it returns `inconclusive` with a CI roughly +/-0.45% of baseline, which is what justifies the 2% default threshold empirically rather than by assertion. No other workload has been measured. Compiler search and custom Metal kernels (G3) remain deferred. External MLX documentation and research are design inputs, not local benchmark evidence.

`auto-mlx tune` races a declarative provider's knob grid against the baseline under that same statistical gate, eliminating candidates only on evidence and never removing the baseline floor; `auto-mlx history` reads back the stored, content-addressed tuning summaries. On the checked-in example grid the race correctly finds **no winner** -- the example's `tile` knob is inert by construction, so the grid is effectively two configurations wide and contains no real effect to find. See [the CLI reference](docs/cli.md) for the full search surface and that result in detail.

Read [the architecture](docs/architecture.md), [the CLI reference](docs/cli.md), [the threat model](docs/threat-model.md), [the measurement contract](docs/measurement.md), [evidence and promotion](docs/evidence-and-promotion.md), and [the research landscape](docs/research-landscape.md) before proposing a new lane.

## License

MIT. See [LICENSE](LICENSE).
