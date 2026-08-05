# Security policy

Auto MLX is a contract and identity layer with evaluator, receipt, promotion, and dispatch libraries, wired into a CLI loop (`evaluate`/`promote`/`dispatch`/`rollback`/`keys ensure`) gated on a local, single-operator sandbox tier. It is not a hardened, multi-tenant OS sandbox, network client, or code-generation service -- see `docs/threat-model.md` for the exact boundary the local tier draws.

## G0/G1/G2 security boundary

The current CLI accepts local JSON files or stdin, plus (for `evaluate`/`dispatch --execute`) real sandboxed subprocess execution of an evaluator-owned, checked-in trusted runner. It does not fetch URLs, install dependencies, import remote modules, invoke an unsandboxed shell, or execute arbitrary candidate code; `validate`/`inspect` never import MLX. Candidate and provider documents contain only bounded scalar configuration values; they cannot supply a command, source text, timer, threshold, evaluator, callback, or self-selected `candidate_id` -- the workload-to-runner binding is CLI-owned (a closed registry keyed by workload name), never candidate-selected. The evaluator's real execution path requires the local sandbox execution primitives and an evaluator-owned isolation provider/authority; without them it fails closed with `SANDBOX_UNAVAILABLE` (CLI: `unavailable`, exit 4). Promotion and dispatch activation require independent recomputation, local supervisor attestation (`auto_mlx.supervisor`), and a positive gain; anything else resolves to native fallback.

Artifact checks reject unsafe relative paths, symlinks, non-regular files, size mismatches, and digest mismatches where safe descriptor operations are available. The local sandbox tier (macOS `sandbox-exec`) is a developer-grade guard for evaluating your own candidate on your own Mac -- it is not a hardened operating-system sandbox and makes no multi-tenant or untrusted-code isolation claim.

The project must not:

- run remote model code, `trust_remote_code`, arbitrary candidate code, shell commands, or install hooks;
- make network calls or accept secrets, API keys, credentials, or environment-derived evaluator policy;
- let a candidate self-promote, choose its identity, choose its own execution command, hide a failed measurement, or rewrite a receipt;
- present an unproven, hardened production evaluation/activation path (beyond the local, single-operator tier) as a successful operation, or leak attestation key material in any command output;
- turn a local observation into a general public speedup claim without independent, receipt-backed evidence.

## Reporting

For a suspected vulnerability, do not attach secrets or proprietary model artifacts to a public issue. Report the smallest reproducible input, the command, the Python version, and the observed diagnostic through the repository's private security channel when one is available. Until a private channel is configured, open a minimal issue without sensitive data and mark it as security-sensitive.

Security fixes must include an offline regression test and preserve the stable failure-code vocabulary where practical.
