<!-- SPDX-License-Identifier: AGPL-3.0-only -->

## What changed

Describe the public surface, contracts and documentation changed.

## Verification

- [ ] python -m pytest -q
- [ ] python tools/check_markdown_links.py
- [ ] python tools/check_governance.py
- [ ] python tools/r6_release_check.py

## Boundary checklist

- [ ] No supplier SDK, credential, restricted model weight, private endpoint or unapproved personal data is included.
- [ ] Candidate, shadow, qualification, Authority and production claims remain distinct.
- [ ] Historical files are unchanged, or an archival correction rationale is included.
- [ ] I have read and agree to CLA.md and have the right to submit this contribution.
