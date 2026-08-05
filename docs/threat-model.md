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

## Local sandbox tier (macOS, `sandbox-exec`)

`auto_mlx.sandbox.LocalSandboxProvider`/`LocalSandboxAuthority` are the first concrete `IsolationProvider`/`IsolationAuthority` pair, and they draw a deliberately narrow boundary that must not be overstated:

- **This is a developer-grade, single-user local guard, not a hardened multi-tenant security boundary.** It is appropriate for evaluating your own candidate configuration on your own Mac. It is not appropriate for running untrusted code from strangers, for isolating mutually-distrusting tenants, or for any claim of production-grade sandboxing. `sandbox-exec` (Seatbelt) is Apple-deprecated-but-present tooling; it is not a kernel-enforced container boundary comparable to gVisor, Firecracker, or a VM.
- **What it enforces**: outbound network denial (`(deny network*)`), filesystem writes scoped to the execution's own working directory (`(allow file-write* (subpath ...))`; reads stay broad for interpreter/framework imports), a fresh session/process group, and conservative CPU/file-size/open-file resource limits (`RLIMIT_CPU`/`RLIMIT_FSIZE`/`RLIMIT_NOFILE`, defense-in-depth alongside `execute_plan`'s own wall-clock timeout and process-group cleanup). Child processes spawned by a sandboxed process inherit the same restrictions (Seatbelt profiles are not escaped by forking).
- **What it does not claim**: no defense against kernel or Seatbelt-bypass vulnerabilities, no defense against a malicious binary that is itself trusted (`TrustedRunner` bytes are pinned and verified, but nothing here re-audits their behavior), no cross-process memory isolation beyond what the OS already provides between any two processes, and no guarantee that the exact same profile enforces identically across macOS versions.
- **Verification stays independent of the provider's self-report.** `LocalSandboxAuthority` never trusts the claim `LocalSandboxProvider.enforce()` returns. It recovers the *exact* profile text the process was launched with from the read-only `Popen.args` attribute (not a provider self-report), confirms the claim's attestation digest actually binds that text, and then runs its own probe subprocesses under the identical profile: an outbound connect attempt, a write outside the scoped subpath, and a write inside it. Only a positive result on all three mints `VerifiedIsolation`; any anomaly (an unexpected allow, a probe crash or timeout, a missing `sandbox-exec` binary) is a typed, fail-closed error, never a claim.
- **Fail-closed by host.** `execute_plan` gates on host primitive availability (macOS, `sandbox-exec` on `PATH`, descriptor-relative `O_NOFOLLOW`/`dir_fd` support) before it will run anything for real, regardless of which provider a caller passes. Hosts without these primitives (including every Linux CI runner) see the exact same `SANDBOX_UNAVAILABLE` behavior as a host with no execution engine at all. Opting into real execution additionally requires a caller to explicitly construct and pass a concrete provider/authority pair; the module defaults remain fail-closed.
- **`VerifiedIsolation.production_eligible` remains permanently `False`** regardless of tier or evidence quality (see `IsolationAuthority`/`VerifiedIsolation` in `executor.py`) -- a Python object field is not authentication, and this local tier does not change that. G1 evidence-layer acceptance (`ObservationBundle.accepted`/`promotion_eligible`) is a necessary input to, but not the same thing as, G2 production activation, which remains a separate, independently-gated lane (see docs/evidence-and-promotion.md).
