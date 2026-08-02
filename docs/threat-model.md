# Threat model

## Assets

- Contract identity: workload, artifact, provider, runtime, and candidate hashes.
- Candidate configuration and the exact declared workload.
- Future measurement receipts and the distinction between observation and promotion.
- Local model/artifact bytes supplied for integrity checking.

## Adversaries and failures

| Threat | G0 control | Remaining boundary |
| --- | --- | --- |
| Candidate selects its own ID or changes config after hashing | IDs are derived; values are frozen; exact fields are checked | A future runner must bind the same identities to every receipt. |
| Path traversal, symlink substitution, or artifact mutation | Relative POSIX paths, no-follow descriptor traversal, size and digest checks | The host must provide the required no-follow primitives. |
| JSON ambiguity | Duplicate keys, floats, non-finite numbers, and unknown fields fail closed | Schemas and Python classes must stay aligned. |
| Arbitrary code or shell injection | Providers contain scalar data only; CLI never invokes a command | The evaluator library requires a real isolation/supervisor proof before production use. |
| Fake promotion or hidden regression | Promotion and dispatch libraries validate identity and fall back closed; no speed claim is emitted | G2 activation must independently recompute receipts and require correctness and evidence. |
| Secret or network exfiltration | No network client, secret input, remote code, or dependency installer in G0 | Future tooling must retain this default-deny posture. |
| Overstated isolation | Documentation explicitly says G0 is not an OS sandbox | A sandbox claim requires a proved host-level isolation mechanism. |

## Security invariant

Untrusted input may be parsed as data and rejected as a contract. It must never become Python source, a shell command, a URL request, an environment-controlled threshold, or a successful-looking evaluation result.
