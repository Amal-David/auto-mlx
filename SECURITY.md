# Security policy

Auto MLX is deliberately a contract and identity layer with evaluator, receipt, promotion, and dispatch libraries. It is not a CLI orchestrator, OS sandbox, network client, or code-generation service.

## G0 security boundary

The current CLI accepts local JSON files or stdin. It does not fetch URLs, install dependencies, import remote modules, invoke a shell, execute candidate code, or import MLX. Candidate and provider documents contain only bounded scalar configuration values; they cannot supply a command, source text, timer, threshold, evaluator, callback, or self-selected `candidate_id`. The evaluator library may fail closed unless a real isolation and supervisor capability is proven; promotion and dispatch activation may fail closed until their evidence and activation gates pass.

Artifact checks reject unsafe relative paths, symlinks, non-regular files, size mismatches, and digest mismatches where safe descriptor operations are available. This is input integrity checking, not an operating-system sandbox. Auto MLX makes no OS sandbox or isolation claim.

The project must not:

- run remote model code, `trust_remote_code`, arbitrary candidate code, shell commands, or install hooks;
- make network calls or accept secrets, API keys, credentials, or environment-derived evaluator policy;
- let a candidate self-promote, choose its identity, hide a failed measurement, or rewrite a receipt;
- present deferred CLI orchestration or an unproven production evaluation/activation path as a successful operation;
- turn a local observation into a public speedup claim without independent evidence.

## Reporting

For a suspected vulnerability, do not attach secrets or proprietary model artifacts to a public issue. Report the smallest reproducible input, the command, the Python version, and the observed diagnostic through the repository's private security channel when one is available. Until a private channel is configured, open a minimal issue without sensitive data and mark it as security-sensitive.

Security fixes must include an offline regression test and preserve the stable failure-code vocabulary where practical.
