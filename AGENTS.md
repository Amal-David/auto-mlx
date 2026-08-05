# Agent guidance

This directory is a standalone, standard-library Python 3.11+ project. Work from the existing contract classes and JSON schemas; do not widen their public invariants casually.

- Keep the G0 CLI offline, dry-run-first, and importable without MLX.
- Keep JSON output on stdout and diagnostics on stderr with stable nonzero exits.
- Treat candidate inputs as declarative data only. Never add shell, remote code, network, secret, or arbitrary callback surfaces.
- Preserve canonical identity, evaluator-derived IDs, safe artifact checks, and exact-field rejection.
- Evaluator, receipt, promotion, and dispatch libraries exist and are wired into the CLI (`evaluate`, `promote`, `dispatch`, `rollback`, `keys ensure`) on the local sandbox tier (`auto_mlx.sandbox`, macOS `sandbox-exec`, one Mac, one operator). `evaluate` and `dispatch --execute` fail closed with `unavailable`/exit 4 without the local sandbox primitives. A hardened, multi-tenant, or fleet-wide production evaluation/activation gate does not exist; compiler search and custom Metal work (G3) remain deferred.
- Every optimization proposal needs a correctness check, benchmark metadata, and a rollback condition.
- Run the complete offline unittest suite before handing work back.
