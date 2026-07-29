# SPDX-License-Identifier: AGPL-3.0-only
"""Check release identity and required governance surfaces."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "GOVERNANCE.md",
    "MAINTAINERS.md",
    "ROADMAP.md",
    "LICENSES/LICENSE-MAP.md",
    "docs/GETTING_STARTED.md",
    "docs/DEVELOPMENT.md",
    "docs/data/DATA_POLICY.md",
    "docs/governance/VERSIONING.md",
    "docs/governance/RELEASE_POLICY.md",
    "historical/SUPPORT_STATUS.md",
    "historical/DATA_CATALOG.md",
    "historical/LICENSE_AUDIT.json",
    ".github/CODEOWNERS",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/bug.yml",
    ".github/ISSUE_TEMPLATE/research-proposal.yml",
]


def main() -> int:
    errors: list[str] = []
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            errors.append(f"missing required governance file: {name}")

    package = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = package["project"]["version"]
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    manifest = json.loads((ROOT / "RESEARCH_RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    sbom = json.loads((ROOT / "SBOM.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    if f'version: "{version.removesuffix(".dev0")}"' not in citation:
        errors.append("CITATION.cff version does not match package release line")
    if manifest.get("research_package_version") != version:
        errors.append("release manifest research_package_version does not match pyproject")
    if sbom["metadata"]["component"]["version"] != version:
        errors.append("SBOM version does not match pyproject")
    for marker in (version, "默认开发分支为", "历史研究数据的公开可见性"):
        if marker not in readme:
            errors.append(f"README missing governance marker: {marker}")

    if errors:
        print("\n".join(errors))
        return 1
    print("governance metadata: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
