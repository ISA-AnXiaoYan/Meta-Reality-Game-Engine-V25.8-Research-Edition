# Versioning

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

MRGE uses separate version axes so that a historical archive is never confused with a supported package.

| Axis | Current value | Meaning |
| --- | --- | --- |
| Repository identity | Meta-Reality-Game-Engine-Research-Edition | Stable public repository name |
| Default branch | main | Current public development line |
| Research package | 0.2.0.dev0 | Version of the supported Python research package |
| Historical coverage | V25.8–V26.6 | Frozen reference material present in historical/ |
| Architecture roadmap | V1.0 | Target design, not a release or qualification statement |

Tags and GitHub Releases are immutable public checkpoints. A release tag must include release notes, source provenance, SBOM scope, known limitations and validation results. Branch names must not be used as the only version signal.
