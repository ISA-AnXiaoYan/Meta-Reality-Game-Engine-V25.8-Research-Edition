param(
    [string]$Python = "",
    [switch]$StrictGit
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "===== $Title ====="
}

function Invoke-Checked {
    param([string]$Exe, [string[]]$ArgList)
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe failed with exit code $LASTEXITCODE"
    }
}

function Resolve-Python {
    if ($Python -ne "") {
        return $Python
    }
    $candidates = @()
    if ($env:CODEX_PYTHON) {
        $candidates += $env:CODEX_PYTHON
    }
    $candidates += "C:\Users\axyis\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return $py.Source
    }
    throw "No Python runtime found. Pass -Python or set CODEX_PYTHON."
}

Write-Section "repo"
$realRoot = (& git -C $RepoRoot rev-parse --show-toplevel).Trim()
Write-Host "repo_root=$realRoot"
$realRootNorm = (($realRoot -replace "\\", "/").TrimEnd("/"))
$repoRootNorm = (($RepoRoot -replace "\\", "/").TrimEnd("/"))
if ($realRootNorm -ne $repoRootNorm) {
    throw "PSScriptRoot repo mismatch: expected=$RepoRoot actual=$realRoot"
}
$branch = (& git -C $RepoRoot branch --show-current).Trim()
$commit = (& git -C $RepoRoot rev-parse --short HEAD).Trim()
Write-Host "branch=$branch commit=$commit"

Write-Section "python"
$PyExe = Resolve-Python
Write-Host "python=$PyExe"
Invoke-Checked -Exe $PyExe -ArgList @("--version")

Write-Section "py_compile"
$compileFiles = @(
    "launch_fusion_system.py",
    "pyopengl_cuda_preview_renderer.py",
    "manual_hit_service.py",
    "game_event_bridge_service.py",
    "global_bev_fusion_renderer.py",
    "global_person_id_server.py",
    "person_id_dataset_recorder.py",
    "remote_ops/check_fullsys_run.py",
    "remote_ops/v25_replay_control_panel.py",
    "remote_ops/start_system_from_panel.py",
    "tools/evaluate_fullsys_node.py",
    "tools/analyze_hit_pipeline.py",
    "tools/compare_hit_judge_experiments.py",
    "tools/compare_global_person_id_filters.py"
)
$existingCompileFiles = @()
foreach ($rel in $compileFiles) {
    if (Test-Path -LiteralPath (Join-Path $RepoRoot $rel)) {
        $existingCompileFiles += $rel
    } else {
        Write-Host "skip_missing=$rel"
    }
}
if ($existingCompileFiles.Count -gt 0) {
    $compileScript = @'
from pathlib import Path
import sys
ok = True
for name in sys.argv[1:]:
    path = Path(name)
    try:
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
        print('compile_ok', name)
    except Exception as exc:
        ok = False
        print('compile_fail', name, type(exc).__name__, exc)
if not ok:
    raise SystemExit(1)
'@
    Invoke-Checked -Exe $PyExe -ArgList (@("-c", $compileScript) + $existingCompileFiles)
}
Write-Host "py_compile=ok"

Write-Section "json"
$jsonFiles = @(
    "launch_profiles/627_event_overlay_bev.json",
    "ids_8cam_fusion_config.json",
    "ids_8cam_fusion_config_627_runtime.json",
    "experiments/hit_judge/bbox48_soft.json",
    "experiments/hit_judge/bbox96_soft.json",
    "experiments/hit_judge/bbox48_shadow.json",
    "experiments/global_id/alpha_beta_countgate.json",
    "experiments/global_id/kalman_shadow_countgate.json",
    "experiments/global_id/handoff_strict_anchor.json",
    "governance/entrypoints.json",
    "governance/file_ownership.json",
    "governance/hygiene_baseline.json",
    "docs/versions/VERSION_MANIFEST_20260702_EVENT_OVERLAY_BEV_DILATE_CONTROL.json",
    "archive/config_snapshots/ids_8cam_fusion_config.before_disable_bgr_preview.json"
)
foreach ($rel in $jsonFiles) {
    $path = Join-Path $RepoRoot $rel
    if (Test-Path -LiteralPath $path) {
        $jsonScript = "import json, pathlib, sys; json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))"
        Invoke-Checked -Exe $PyExe -ArgList @("-c", $jsonScript, $path)
        Write-Host "json_ok=$rel"
    } else {
        Write-Host "json_skip_missing=$rel"
    }
}

Write-Section "repo hygiene"
Invoke-Checked -Exe $PyExe -ArgList @("tools/check_repo_hygiene.py", "--project-root", $RepoRoot)

Write-Section "git status"
$status = & git -C $RepoRoot status --short --untracked-files=all
if ($status) {
    $status | ForEach-Object { Write-Host $_ }
} else {
    Write-Host "clean"
}
$untracked = & git -C $RepoRoot ls-files --others --exclude-standard
if ($untracked) {
    Write-Host "untracked_count=$($untracked.Count)"
    if ($StrictGit) {
        throw "StrictGit enabled and untracked files are present."
    }
}

Write-Section "remote ops"
$remoteFiles = @(
    "remote_ops/check_fullsys_run.py",
    "remote_ops/deploy_branch_snapshot.sh",
    "remote_ops/run_fullsys_test.sh",
    "remote_ops/start_latest_system.sh",
    "remote_ops/stop_fullsys_test.sh",
    "remote_ops/stop_latest_system.sh",
    "remote_ops/validate_remote_env.sh",
    "remote_ops/wait_latest_system_ready.sh",
    "remote_ops/open_startup_control_panel.sh",
    "remote_ops/start_system_from_panel.py",
    "tools/install_desktop_shortcuts.ps1",
    "tools/one_click_start_latest.ps1",
    "tools/one_click_stop_latest.ps1",
    "tools/remote_test.ps1"
)
foreach ($rel in $remoteFiles) {
    $path = Join-Path $RepoRoot $rel
    if (Test-Path -LiteralPath $path) {
        Write-Host "exists=$rel"
    } else {
        throw "missing remote tool file: $rel"
    }
}

Write-Section "result"
Write-Host "dev_check=ok"
