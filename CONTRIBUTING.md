# Contributing

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

Thank you for contributing to MRGE Research Edition.

## Before opening a pull request

1. Start from main and keep changes focused on one public surface.
2. Do not submit supplier SDKs, real device handles, credentials, absolute paths, restricted weights, unapproved personal data or private deployment secrets.
3. Put the correct SPDX identifier on new source files. Contracts/Replay use Apache-2.0; engine, tools and documentation use AGPL-3.0-only unless explicitly mapped otherwise.
4. Update schema, Replay samples, tests and documentation together for every public cross-module field change.
5. Keep candidate, shadow, qualified, authority_ready and production claims separate.
6. Run the local checks below.

~~~powershell
python -m pip install -e . pytest
python -m pytest -q
python tools/check_markdown_links.py
python tools/check_governance.py
python tools/r6_release_check.py
~~~

## Historical archive

Do not modify historical/ as part of a normal feature pull request. Archive corrections require a clear provenance, privacy, license or security rationale and maintainer review.

## Contributor License Agreement

This project retains a CLA to keep the public AGPL distribution and possible separate commercial licensing path legally clear. By checking the CLA confirmation in the pull request template, you confirm that you have read and agree to [CLA.md](CLA.md), and that you have the right to submit the contribution.

## Pull request description

State the change scope, verification commands, affected contracts, evidence paths if applicable, and any unresolved boundary. Do not describe a local smoke or replay result as field qualification or production readiness.
