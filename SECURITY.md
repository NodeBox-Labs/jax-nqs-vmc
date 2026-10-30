# Security Policy

## Supported Versions

Only the latest release of `nqs-vmc` is actively supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.4.x   | :white_check_mark: |
| < 0.4   | :x:                |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

If you discover a security issue (for example, unsafe deserialization in the checkpoint loader, or a dependency with a known CVE), please report it privately by emailing the maintainers at:

**security@nodebox-labs.org**

Please include:
- A description of the vulnerability and its potential impact.
- Steps to reproduce the issue.
- The version of `nqs-vmc` and the versions of `jax`, `flax`, `optax`, `numpy`, and `scipy` you are using.
- Any suggested mitigation.

We will acknowledge your report within 72 hours and aim to provide a fix within 30 days. We will credit you in the release notes for the patched version, unless you prefer to remain anonymous.

## Scope

This policy covers:
- The Python code in `src/nqs_vmc/`.
- The CLI in `src/nqs_vmc/cli.py`.
- The GitHub Actions workflows in `.github/workflows/`.

It does **not** cover vulnerabilities in third-party dependencies (`jax`, `flax`, `optax`, `scipy`, `numpy`, `pyyaml`, `orbax-checkpoint`). Those should be reported to the respective upstream projects.