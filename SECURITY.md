# Security Policy

## Supported Versions

OpenVoiceCS-Bench is pre-1.0. Only the `main` branch and the most recent
release tag receive fixes.

| Version | Supported |
| --- | --- |
| `main` | Yes |
| `v0.1.x` | Yes |
| Older tags | No |

## Reporting A Vulnerability

**Do not open a public issue for a security problem.**

Report privately through GitHub:

1. Go to the [Security advisories page](https://github.com/0daycloud/openvoicecs-bench/security/advisories/new).
2. Describe the issue, affected files or commands, and reproduction steps.
3. Include impact: what an attacker gains, and whether it needs local access,
   a crafted submission, or a hosted evaluator deployment.

Expect an acknowledgement within 7 days and a status update within 30 days.
There is no bug bounty.

## What Counts As A Vulnerability Here

This is a benchmark harness, not a hosted service, so the threat model is
narrower than a typical application. In scope:

- **Arbitrary code execution from untrusted input.** `submit` and
  `init-submission` import a Python module by path and call it. Anything that
  turns a *scenario file*, *report*, *manifest*, or *judge annotation* into
  code execution is a vulnerability.
- **Credential leakage.** Any path where a provider API key reaches a report,
  log, run manifest, release bundle, or error message.
- **Sealed-split disclosure.** Any way to recover sealed-test scenario IDs,
  prompts, or expected states from published commitments, judged reports,
  release bundles, or the sealed-evaluator queue. Split commitments are meant
  to publish counts and hashes only.
- **Integrity bypass.** Any way to make `verify-release`,
  `validate-release-bundle`, `validate-submission-intake`, or the SHA-256
  pinning in the manifests accept mismatched or tampered artifacts.
- **Path traversal or overwrite** through a scenario, manifest, or output-path
  argument.

## Explicitly Out Of Scope

- **Benchmark gaming.** Overfitting to public-dev scenarios, prompt-tuning
  against the oracle, or scoring high without solving the task is a benchmark
  *validity* problem, not a security problem. Open a normal issue; see
  `docs/known-limitations.md`.
- **Model behavior.** Unsafe, biased, or non-compliant output from a scored
  model is the finding the benchmark exists to produce.
- **Scenario realism.** Adversarial-compliance scenarios deliberately contain
  social-engineering attempts, fake authentication challenges, and prompt
  injection. That content is intentional.
- **Dependency CVEs with no exploit path** through this code.
- Running `scripts/serve_openvoicecs_realtime.py` on a public interface. It is a local
  development server with no authentication, by design; it binds `127.0.0.1`.

## Handling Credentials

Provider API keys are read from the environment or a gitignored `.env` (see
`.env.example`). They are never written to reports or manifests. If you find a
key in any committed artifact, treat it as a vulnerability and report it
privately, then rotate the key.

## Data Sensitivity

All benchmark data is synthetic. It contains no real customer records, no real
personal data, and no recordings of real people. If you believe any scenario
contains data traceable to a real person, report it privately and it will be
treated as a release-blocking erratum under `GOVERNANCE.md`.
