# Historical data policy

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

## Purpose

Historical JSON, JSONL, configuration and evidence records are included to make the engineering history inspectable and reproducible. They are not a benchmark certification, a biometric dataset, a production training corpus or a field-performance claim.

## Use boundary

- Use synthetic samples for normal development and CI.
- Treat historical records as frozen research references.
- Do not use repository material for identity resolution, biometric profiling, face recognition training, commercial model training, surveillance, or any purpose incompatible with the original collection and publication authority.
- Do not combine it with external personal data or publish derived personal identifiers.
- Do not assume that public visibility is an independent data license. When a downstream use needs a data grant, contact the maintainer before use.

## Privacy and security

The maintainer declares release authority only for material deliberately imported into the archive. Credentials, supplier artifacts, model weights, unauthorized personal material and files with unclear provenance are excluded. If you find a privacy, credential or authorization problem, follow SECURITY.md and do not open a public issue containing the data.

## Traceability

Each release must retain the archive release manifest, exclusion list and data catalog. Historical data must not be silently edited; corrections require a provenance note.
