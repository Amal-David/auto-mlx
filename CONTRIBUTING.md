# Contributing to Auto MLX

Auto MLX is an executable engineering standard: correctness, provenance, and honest capability boundaries are part of the product.

## Before changing a contract

Read the relevant class, schema, tests, and [architecture](docs/architecture.md). Preserve canonical serialization, evaluator-owned IDs, exact-field rejection, immutable values, and the no-code provider boundary. A schema or contract change must include a focused offline test and an explanation of how old documents behave.

## Development loop

Use Python 3.11 or newer and no network-dependent test fixture:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Run the CLI against the files in `examples/` after changing command behavior. Keep JSON results on stdout and diagnostics on stderr. Do not overwrite files by default.

## Evidence rules

Architecture lessons are not empirical results. Any future evaluator must record the pinned workload, source oracle, candidate and runtime identities, hardware/software versions, warmups, all planned samples, output limits, correctness result, and failure state. A benchmark number without that metadata is not publishable evidence. No contributor may describe the current repository as having measured an MLX speedup.

Do not add remote model execution, arbitrary code generation, shell execution, secret handling, automatic activation, or unsupported sandbox claims. The evaluator, receipt, promotion, and dispatch libraries must fail closed without their required isolation, supervisor, evidence, and activation gates; CLI orchestration, compiler search, and custom Metal work need an explicit design and review gate before wiring or implementation.

## Documentation and review

Keep `SKILL.md`-style guidance compact and put architecture detail in `docs/`. Every research-derived recommendation must name its primary source, label the claim as an architecture lesson or empirical evidence, and state the missing validation gate. CI runs two jobs: an offline unittest matrix with no network access, and a package-certification job that installs `build`/`setuptools`/`wheel` (and `jsonschema` for the live schema-validator assertions) from PyPI to build and smoke-test the sdist and wheel.
