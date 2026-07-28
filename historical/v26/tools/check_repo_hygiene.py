#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REQUIRED_IGNORE_PROBES = {
    "sync_ipc/probe.json": "runtime IPC",
    "recordings/probe.raw": "recorded datasets",
    "raw_records/probe.raw": "sensor RAW",
    "desktop_runs/probe/status.json": "desktop run output",
    "status_logs/probe/status.json": "status logs",
    "cpu_hotspots_probe/perf.data": "CPU hotspot output",
    "probe.tar.gz": "run archives",
    "probe.pt": "model weights",
    "probe.engine": "TensorRT engines",
    "runtime/probe.json": "generated runtime root",
    "configs/local/probe.json": "machine-local config",
    ".ssh/private_key": "credentials",
}

FORBIDDEN_TRACKED_PATTERNS = (
    "sync_ipc/**",
    "recordings/**",
    "raw_records/**",
    "desktop_runs/**",
    "status_logs/**",
    "cpu_hotspots*/**",
    "runtime/**",
    "configs/local/**",
    ".ssh/**",
    "*.tar.gz",
    "*.tgz",
    "*.pt",
    "*.pth",
    "*.engine",
    "*.onnx",
    "*.raw",
    "*.avi",
    "*.mp4",
    "perf.data*",
    "core.*",
)

ABSOLUTE_HOME_MARKER = "/home/" + "ysxq/"


def _run(root: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        list(args),
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(args)}\n{proc.stderr}")
    return proc.stdout


def _repo_root(raw: str) -> Path:
    start = Path(raw).expanduser().resolve()
    output = _run(start, "git", "rev-parse", "--show-toplevel")
    return Path(output.strip()).resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _repo_files(root: Path) -> Tuple[List[str], List[str], List[str]]:
    tracked = [line.strip().replace("\\", "/") for line in _run(root, "git", "ls-files").splitlines() if line.strip()]
    untracked = [
        line.strip().replace("\\", "/")
        for line in _run(root, "git", "ls-files", "--others", "--exclude-standard").splitlines()
        if line.strip()
    ]
    return tracked, untracked, sorted(set(tracked + untracked))


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_groups(groups: Sequence[Sequence[str]]) -> set[Tuple[str, ...]]:
    return {tuple(sorted(str(item).replace("\\", "/") for item in group)) for group in groups}


def _check_ignore(root: Path, probe: str) -> bool:
    proc = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", probe],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def evaluate(root: Path) -> Tuple[List[str], List[str], Dict[str, Any]]:
    errors: List[str] = []
    warnings: List[str] = []
    summary: Dict[str, Any] = {}

    entrypoints = _load_json(root / "governance" / "entrypoints.json")
    ownership = _load_json(root / "governance" / "file_ownership.json")
    baseline = _load_json(root / "governance" / "hygiene_baseline.json")
    tracked, untracked, files = _repo_files(root)
    summary.update(tracked_files=len(tracked), untracked_files=len(untracked), evaluated_files=len(files))

    for probe, purpose in REQUIRED_IGNORE_PROBES.items():
        if not _check_ignore(root, probe):
            errors.append(f"gitignore_missing: {probe} ({purpose})")

    forbidden = [path for path in files if _matches(path, FORBIDDEN_TRACKED_PATTERNS)]
    if forbidden:
        errors.extend(f"forbidden_tracked_artifact: {path}" for path in forbidden)

    limits = baseline.get("limits", {}) if isinstance(baseline.get("limits"), dict) else {}
    root_files = [path for path in files if "/" not in path]
    root_python = [path for path in root_files if path.lower().endswith(".py")]
    root_markdown = [path for path in root_files if path.lower().endswith(".md")]
    root_metrics = {
        "root_tracked_files": len(root_files),
        "root_python_files": len(root_python),
        "root_markdown_files": len(root_markdown),
    }
    summary.update(root_metrics)
    for key, actual in root_metrics.items():
        limit = int(limits.get(key, actual))
        if actual > limit:
            errors.append(f"root_limit_exceeded: {key} actual={actual} limit={limit}")

    allowed_zero = {str(item).replace("\\", "/") for item in baseline.get("allowed_zero_byte_files", [])}
    zero_files = []
    for rel in files:
        path = root / rel
        if path.is_file() and path.stat().st_size == 0:
            zero_files.append(rel)
            if rel not in allowed_zero:
                errors.append(f"new_zero_byte_file: {rel}")
    summary["zero_byte_files"] = len(zero_files)
    stale_zero_allowances = sorted(allowed_zero - set(zero_files))
    if stale_zero_allowances:
        warnings.append(f"remove_stale_zero_allowances: count={len(stale_zero_allowances)}")

    min_bytes = int(limits.get("duplicate_min_bytes", 1024))
    by_hash: Dict[str, List[str]] = defaultdict(list)
    for rel in files:
        path = root / rel
        if path.is_file() and path.stat().st_size >= min_bytes:
            by_hash[_sha256(path)].append(rel)
    actual_duplicate_groups = {
        tuple(sorted(group)) for group in by_hash.values() if len(group) > 1
    }
    allowed_duplicate_groups = _normalized_groups(baseline.get("allowed_duplicate_groups", []))
    for group in sorted(actual_duplicate_groups - allowed_duplicate_groups):
        errors.append(f"new_duplicate_group: {' | '.join(group)}")
    stale_duplicate_allowances = allowed_duplicate_groups - actual_duplicate_groups
    if stale_duplicate_allowances:
        warnings.append(f"remove_stale_duplicate_allowances: count={len(stale_duplicate_allowances)}")
    summary["duplicate_groups"] = len(actual_duplicate_groups)

    hard_path_budget = {
        str(path).replace("\\", "/"): int(count)
        for path, count in (baseline.get("absolute_home_path_budget", {}) or {}).items()
    }
    hard_path_counts: Dict[str, int] = {}
    for rel in files:
        path = root / rel
        if not path.is_file() or path.suffix.lower() not in {".py", ".sh", ".json", ".ps1"}:
            continue
        try:
            count = path.read_text(encoding="utf-8").count(ABSOLUTE_HOME_MARKER)
        except UnicodeError:
            continue
        if count:
            hard_path_counts[rel] = count
            budget = int(hard_path_budget.get(rel, 0))
            if count > budget:
                errors.append(f"absolute_home_path_budget_exceeded: {rel} actual={count} budget={budget}")
    summary["absolute_home_paths"] = sum(hard_path_counts.values())
    stale_hard_path_allowances = sorted(set(hard_path_budget) - set(hard_path_counts))
    if stale_hard_path_allowances:
        warnings.append(f"remove_stale_hard_path_allowances: count={len(stale_hard_path_allowances)}")

    seen_ids = set()
    canonical_paths = set()
    for item in entrypoints.get("entrypoints", []):
        if not isinstance(item, dict):
            errors.append("entrypoint_not_object")
            continue
        entry_id = str(item.get("id") or "")
        canonical = str(item.get("path") or "").replace("\\", "/")
        if not entry_id or entry_id in seen_ids:
            errors.append(f"entrypoint_id_invalid_or_duplicate: {entry_id or '<missing>'}")
        seen_ids.add(entry_id)
        if not canonical or canonical in canonical_paths:
            errors.append(f"canonical_entrypoint_invalid_or_duplicate: {canonical or '<missing>'}")
        canonical_paths.add(canonical)
        if canonical and not (root / canonical).is_file():
            errors.append(f"canonical_entrypoint_missing: {entry_id} -> {canonical}")
        for alias in item.get("aliases", []):
            alias_path = str(alias).replace("\\", "/")
            if alias_path and not (root / alias_path).exists():
                warnings.append(f"legacy_alias_missing: {entry_id} -> {alias_path}")
    summary["canonical_entrypoints"] = len(canonical_paths)

    document_index = root / "PROJECT_DOCUMENT_INDEX_CN.md"
    indexed_paths = []
    if document_index.is_file():
        indexed_paths = re.findall(r"`([^`]+\.(?:md|json))`", document_index.read_text(encoding="utf-8"), flags=re.IGNORECASE)
        for indexed in indexed_paths:
            normalized = indexed.replace("\\", "/")
            if "*" in normalized:
                if not list(root.glob(normalized)):
                    errors.append(f"document_index_glob_missing: {normalized}")
            elif not (root / normalized).is_file():
                errors.append(f"document_index_path_missing: {normalized}")
    else:
        errors.append("document_index_missing: PROJECT_DOCUMENT_INDEX_CN.md")
    summary["document_index_paths"] = len(indexed_paths)

    rules = ownership.get("rules", [])
    fallback_files = []
    unowned_files = []
    for rel in files:
        match = next(
            (rule for rule in rules if isinstance(rule, dict) and fnmatch.fnmatch(rel, str(rule.get("pattern") or ""))),
            None,
        )
        if match is None:
            unowned_files.append(rel)
        elif str(match.get("owner") or "") == "fallback_flat":
            fallback_files.append(rel)
    errors.extend(f"file_without_owner: {path}" for path in unowned_files)
    summary["fallback_flat_files"] = len(fallback_files)
    if fallback_files:
        warnings.append(f"fallback_flat_migration_debt: count={len(fallback_files)}")

    return errors, warnings, summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Enforce repository file-governance guardrails.")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--json", action="store_true", default=False)
    args = ap.parse_args(argv)
    root = _repo_root(args.project_root)
    errors, warnings, summary = evaluate(root)
    payload = {"ok": not errors, "summary": summary, "warnings": warnings, "errors": errors}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("repo_hygiene=" + ("ok" if not errors else "failed"))
        for key, value in sorted(summary.items()):
            print(f"{key}={value}")
        for warning in warnings:
            print(f"WARN {warning}")
        for error in errors:
            print(f"ERROR {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
