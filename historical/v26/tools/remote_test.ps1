param(
    [Parameter(Position = 0)]
    [ValidateSet("check", "sync-ops", "deploy", "deploy-worktree", "validate", "install-desktop-shortcuts", "open-startup-panel", "restart-startup-panel", "check-startup-panel", "build-native-brokers", "list-datasets", "check-dataset-manifest", "v25-replay-control", "check-event-raw-exttrig", "smoke", "smoke-live", "smoke-event-replay", "smoke-mvs-broker", "smoke-mvs-replay", "smoke-full-replay", "stress", "longrun", "start-latest", "stop-latest", "stop", "stop-smb", "summary", "assess", "fetch", "monitor", "hot-update")]
    [string]$Action = "smoke",

    [string]$HostName = "ysxq@192.168.31.115",
    [string]$RemoteProject = "~/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627",
    [string]$SmbProject = "\\192.168.31.115\test622yolo_gpu_627",
    [string]$Branch = "",
    [string]$Profile = "launch_profiles/627_event_overlay_bev.json",
    [int]$Duration = 0,
    [string]$Label = "",
    [string]$RunDir = "",
    [string]$LocalOutDir = "",
    [switch]$DeployWorktree,
    [switch]$Wait,
    [switch]$FetchAfter,
    [switch]$SummaryAfter,
    [switch]$FullSummary,
    [string]$PersonDataset = "",
    [string]$DatasetLabel = "",
    [switch]$EvidenceReplayDataset,
    [switch]$FullRecording,
    [int]$ControlDelay = 90,
    [int]$DatasetStopMargin = 15,
    [int]$StopWait = 90,
    [int]$ReadyWait = 120,
    [switch]$NoReadyWait,
    [switch]$LongRun,
    [string]$ExperimentId = "",
    [string]$ExperimentStage = "",
    [string]$ExperimentNote = "",
    [string]$AssessmentNode = "",
    [string]$HitProfile = "",
    [string]$GlobalIdProfile = "",
    [ValidateSet("strong_gate", "strong_reasoning", "both_shadow_compare", "")]
    [string]$AuthorityMode = "",
    [ValidateSet("off", "shadow", "soft", "hard", "")]
    [string]$HumanNoiseMode = "",
    [string]$BboxStartExpandPx = "",
    [ValidateSet("soft", "hard", "off", "")]
    [string]$BboxStartPolicy = "",
    [ValidateSet("official", "shadow", "both", "")]
    [string]$GlobalPersonTarget = "",
    [ValidateSet("alpha_beta", "kalman_cv", "")]
    [string]$GlobalPersonFilterBackend = "",
    [ValidateSet("pending_only", "shadow_only", "final_allowed", "")]
    [string]$ProjectedMaskFinalMode = "",
    [ValidateSet("true", "false", "")]
    [string]$ProjectedMaskRequireTerminalOrTurn = "",
    [string]$SameTrackEpisodeDedupMs = "",
    [string]$ProjectExtendPx = "",
    [string]$ProjectMaxLastDistPx = "",
    [string]$DatasetDir = "",
    [int]$DatasetLimit = 20,
    [int]$MaxFrames = 80,
    [switch]$UpdateManifest,
    [switch]$PrintResolvedManifest,
    [double]$ExpectedPeriodUs = 25000.0,
    [double]$PeriodToleranceUs = 2500.0,
    [string]$EventReplayRawFiles = "",
    [ValidateSet("live_hal", "file_replay", "")]
    [string]$EventSourceMode = "",
    [ValidateSet("live_camera", "file_replay", "")]
    [string]$MvsSourceMode = "",
    [ValidateSet("realtime", "fast", "")]
    [string]$ReplayTiming = "",
    [ValidateSet("live", "event_replay", "mvs_replay", "full_replay", "")]
    [string]$ReplayControlMode = "",
    [switch]$AllowMissingExtTrigger
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RemoteOps = Join-Path $RepoRoot "remote_ops"
$SshOptions = @("-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2")
$ScpOptions = $SshOptions
if ($LocalOutDir -eq "") {
    $standardOutDir = "C:\dev\mrg_remote_tests"
    if (Test-Path -LiteralPath "C:\dev") {
        $LocalOutDir = $standardOutDir
    } else {
        $LocalOutDir = Join-Path (Split-Path -Parent $RepoRoot) "remote_tests"
    }
}

function Invoke-Checked {
    param([string]$Exe, [string[]]$ArgList)
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Capture {
    param([string]$Exe, [string[]]$ArgList)
    $out = & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe failed with exit code $LASTEXITCODE"
    }
    return $out
}

function Quote-Sh {
    param([string]$Value)
    return "'" + ($Value -replace "'", "'\''") + "'"
}

function New-RemoteScriptText {
    param([string]$Command, [switch]$NoCd)
    $cdBlock = if ($NoCd) { "" } else { 'cd "$REMOTE_PROJECT"' }
    $template = @'
#!/usr/bin/env bash
set -euo pipefail
trap 'rm -f "$0"' EXIT
REMOTE_PROJECT_RAW=__REMOTE_PROJECT__
case "$REMOTE_PROJECT_RAW" in
  "~") REMOTE_PROJECT="$HOME" ;;
  "~/"*) REMOTE_PROJECT="$HOME/${REMOTE_PROJECT_RAW#\~/}" ;;
  *) REMOTE_PROJECT="$REMOTE_PROJECT_RAW" ;;
esac
export PROJECT="$REMOTE_PROJECT"
__CD_BLOCK__
__COMMAND__
'@
    return $template.Replace("__REMOTE_PROJECT__", (Quote-Sh $RemoteProject)).Replace("__CD_BLOCK__", $cdBlock).Replace("__COMMAND__", $Command)
}

function Write-TempRemoteScript {
    param([string]$Command, [switch]$NoCd)
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss_ffff"
    $name = "mrg_remote_test_${stamp}_$PID.sh"
    $local = Join-Path $env:TEMP $name
    $remote = "/tmp/$name"
    $text = New-RemoteScriptText -Command $Command -NoCd:$NoCd
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($local, ($text -replace "`r`n", "`n"), $utf8NoBom)
    Invoke-Checked -Exe "scp" -ArgList @($ScpOptions + @($local, "$HostName`:$remote"))
    Remove-Item -LiteralPath $local -Force
    return $remote
}

function Invoke-RemoteScript {
    param([string]$Command, [switch]$NoCd)
    $remote = Write-TempRemoteScript -Command $Command -NoCd:$NoCd
    Invoke-Checked -Exe "ssh" -ArgList @($SshOptions + @($HostName, "bash $remote"))
}

function Invoke-RemoteScriptCapture {
    param([string]$Command, [switch]$NoCd)
    $remote = Write-TempRemoteScript -Command $Command -NoCd:$NoCd
    return Invoke-Capture -Exe "ssh" -ArgList @($SshOptions + @($HostName, "bash $remote"))
}

function Invoke-Remote {
    param([string]$Command)
    Invoke-RemoteScript -Command $Command
}

function Invoke-RemoteRaw {
    param([string]$Command)
    Invoke-RemoteScript -Command $Command -NoCd
}

function Invoke-RemoteCapture {
    param([string]$Command)
    return Invoke-RemoteScriptCapture -Command $Command
}

function Get-SafeLabel {
    param([string]$Value, [string]$Default)
    $raw = if ($Value -ne "") { $Value } else { $Default }
    $safe = $raw -replace "[^A-Za-z0-9_.-]+", "_"
    if ($safe -eq "") { return $Default }
    return $safe
}

function Write-JsonNoBom {
    param([string]$Path, [object]$Value)
    $dir = Split-Path -Parent $Path
    if ($dir -ne "" -and !(Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
    $json = $Value | ConvertTo-Json -Depth 32 -Compress
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($Path, $json, $utf8NoBom)
}

function Write-TextNoBom {
    param([string]$Path, [string]$Text)
    $dir = Split-Path -Parent $Path
    if ($dir -ne "" -and !(Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($Path, $Text, $utf8NoBom)
}

function Resolve-LocalPython {
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
    throw "No local Python runtime found. Set CODEX_PYTHON or install python."
}

function Read-JsonHashtable {
    param([string]$Path)
    $out = [ordered]@{}
    if (!(Test-Path -LiteralPath $Path)) {
        return $out
    }
    try {
        $obj = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        foreach ($prop in $obj.PSObject.Properties) {
            $out[$prop.Name] = $prop.Value
        }
    } catch {
        $out["control_read_error"] = $_.Exception.Message
    }
    return $out
}

function Convert-RemoteRunToSmbPath {
    param([string]$RemoteRun)
    $raw = ($RemoteRun -replace "`r", "" -replace "`n", "").Trim()
    if ($raw -eq "") { return "" }
    $slash = $raw -replace "/", "\"
    $idx = $slash.IndexOf("\sync_ipc\")
    if ($idx -ge 0) {
        $rel = $slash.Substring($idx + 1)
        return (Join-Path $SmbProject $rel)
    }
    if ($slash.StartsWith("sync_ipc\")) {
        return (Join-Path $SmbProject $slash)
    }
    return (Join-Path $SmbProject $slash)
}

function Get-LatestRunViaSmb {
    param([string]$RunLabel)
    $sync = Join-Path $SmbProject "sync_ipc"
    if (!(Test-Path -LiteralPath $sync)) {
        throw "SMB sync path not reachable: $sync"
    }
    $latestFile = ""
    if ($RunLabel -ne "__latest__" -and $RunLabel -ne "") {
        $candidate = Join-Path $sync "latest_${RunLabel}_run_dir.txt"
        if (Test-Path -LiteralPath $candidate) {
            $latestFile = $candidate
        }
    }
    if ($latestFile -eq "") {
        $latestFile = Get-ChildItem -LiteralPath $sync -Filter "latest_*_run_dir.txt" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1 -ExpandProperty FullName
    }
    if ($latestFile -eq "") {
        throw "No latest_*_run_dir.txt found under $sync"
    }
    $remoteRun = (Get-Content -LiteralPath $latestFile -Raw).Trim()
    return [ordered]@{
        LatestFile = $latestFile
        RemoteRun = $remoteRun
        RunPath = Convert-RemoteRunToSmbPath $remoteRun
    }
}

function Request-StopViaSmb {
    param([string]$RunLabel, [int]$WaitSeconds = 90, [string]$Reason = "operator_stop_smb_fallback")
    $info = Get-LatestRunViaSmb -RunLabel $RunLabel
    $sync = Join-Path $SmbProject "sync_ipc"
    $runPath = [string]$info.RunPath
    $nowUs = [int64]([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() * 1000)
    $payload = [ordered]@{
        schema_version = 1
        source = "tools_remote_test_smb_stop"
        label = $RunLabel
        reason = $Reason
        created_wall_us = $nowUs
        remote_run = $info.RemoteRun
    }
    Write-JsonNoBom -Path (Join-Path $sync "remote_stop_request.json") -Value $payload
    if ($runPath -ne "" -and (Test-Path -LiteralPath $runPath)) {
        Write-JsonNoBom -Path (Join-Path $runPath "stop_requested.json") -Value $payload
    }

    $hitPath = Join-Path $sync "hit_judge_control.json"
    $hit = Read-JsonHashtable -Path $hitPath
    $seq = 0
    if ($hit.Contains("seq") -and $null -ne $hit["seq"]) {
        $seq = [int]$hit["seq"]
    }
    $hit["schema_version"] = 1
    $hit["seq"] = $seq + 1
    $hit["source"] = "tools_remote_test_smb_stop_safe_off"
    $hit["updated_wall_us"] = $nowUs
    $hit["hit_detection_enabled"] = $false
    $hit["hit_candidate_output_enabled"] = $false
    Write-JsonNoBom -Path $hitPath -Value $hit

    Write-Host "[remote_test][SMB] stop requested for run=$($info.RemoteRun)"
    Write-Host "[remote_test][SMB] wrote $(Join-Path $sync 'remote_stop_request.json')"
    if ($runPath -ne "" -and (Test-Path -LiteralPath $runPath)) {
        Write-Host "[remote_test][SMB] wrote $(Join-Path $runPath 'stop_requested.json')"
    }
    Write-Host "[remote_test][SMB] hit_judge_control safe-off written as UTF-8 without BOM"

    $archivePath = if ($runPath -ne "") { "$runPath.tar.gz" } else { "" }
    $deadline = (Get-Date).AddSeconds($WaitSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($runPath -ne "" -and (Test-Path -LiteralPath (Join-Path $runPath "launcher_rc.txt")) -and $archivePath -ne "" -and (Test-Path -LiteralPath $archivePath)) {
            break
        }
        Start-Sleep -Seconds 1
    }
    if ($runPath -ne "" -and (Test-Path -LiteralPath $runPath)) {
        Write-Host "[remote_test][SMB] run=$runPath"
        $rcPath = Join-Path $runPath "launcher_rc.txt"
        if (Test-Path -LiteralPath $rcPath) {
            Write-Host "[remote_test][SMB] launcher_rc=$(Get-Content -LiteralPath $rcPath -Raw)"
        } else {
            Write-Host "[remote_test][SMB][WARN] launcher_rc.txt not present yet"
        }
        if ($archivePath -ne "" -and (Test-Path -LiteralPath $archivePath)) {
            Get-ChildItem -LiteralPath $archivePath | Select-Object FullName,Length,LastWriteTime
        } else {
            Write-Host "[remote_test][SMB][WARN] archive not present yet: $archivePath"
        }
    }
}

function Sync-Ops {
    Invoke-RemoteRaw 'mkdir -p "$REMOTE_PROJECT/remote_ops" "$REMOTE_PROJECT/tools"'
    $files = @(
        "build_native_brokers.sh",
        "check_v25_dataset_manifest.py",
        "v25_replay_control_panel.py",
        "list_v25_datasets.py",
        "mvs_file_frame_broker.py",
        "open_startup_control_panel.sh",
        "start_system_from_panel.py",
        "check_fullsys_run.py",
        "deploy_branch_snapshot.sh",
        "preflight_clean_runtime.sh",
        "run_fullsys_test.sh",
        "start_latest_system.sh",
        "stop_fullsys_test.sh",
        "stop_latest_system.sh",
        "validate_remote_env.sh",
        "wait_latest_system_ready.sh"
    )
    foreach ($name in $files) {
        $src = Join-Path $RemoteOps $name
        Invoke-Checked -Exe "scp" -ArgList @($ScpOptions + @($src, "$HostName`:$RemoteProject/remote_ops/$name"))
    }
    $toolFiles = @(
        "check_event_raw_ext_trigger.py"
    )
    foreach ($name in $toolFiles) {
        $src = Join-Path (Join-Path $RepoRoot "tools") $name
        if (Test-Path -LiteralPath $src) {
            Invoke-Checked -Exe "scp" -ArgList @($ScpOptions + @($src, "$HostName`:$RemoteProject/tools/$name"))
        }
    }
    Invoke-RemoteRaw 'chmod +x "$REMOTE_PROJECT"/remote_ops/*.sh "$REMOTE_PROJECT"/remote_ops/*.py "$REMOTE_PROJECT"/tools/*.py'
}

function Get-CurrentBranch {
    if ($Branch -ne "") { return $Branch }
    $b = (& git -C $RepoRoot branch --show-current).Trim()
    if ($b -eq "") { return "HEAD" }
    return $b
}

function Get-CurrentCommit {
    return (& git -C $RepoRoot rev-parse HEAD).Trim()
}

function Deploy-Snapshot {
    Sync-Ops
    $branchName = Get-CurrentBranch
    $commit = Get-CurrentCommit
    $tmp = Join-Path $env:TEMP ("remote_ops_" + $branchName.Replace("/", "_") + "_" + $commit.Substring(0, 8) + ".tar")
    if (Test-Path -LiteralPath $tmp) {
        Remove-Item -LiteralPath $tmp -Force
    }
    Invoke-Checked -Exe "git" -ArgList @("-C", $RepoRoot, "archive", "--format=tar", "-o", $tmp, $branchName)
    $remoteArchive = "/tmp/" + [IO.Path]::GetFileName($tmp)
    Invoke-Checked -Exe "scp" -ArgList @($ScpOptions + @($tmp, "$HostName`:$remoteArchive"))
    Invoke-Remote "bash remote_ops/deploy_branch_snapshot.sh $(Quote-Sh $remoteArchive) $(Quote-Sh $branchName) $(Quote-Sh $commit)"
}

function Get-WorktreeOverlayFiles {
    $tracked = & git -C $RepoRoot diff --name-only HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "git diff failed"
    }
    $untracked = & git -C $RepoRoot ls-files --others --exclude-standard
    if ($LASTEXITCODE -ne 0) {
        throw "git ls-files failed"
    }
    $set = New-Object "System.Collections.Generic.HashSet[string]"
    foreach ($item in @($tracked + $untracked)) {
        $rel = [string]$item
        if ($rel -eq "") { continue }
        $full = Join-Path $RepoRoot $rel
        if (Test-Path -LiteralPath $full -PathType Leaf) {
            [void]$set.Add(($rel -replace "\\", "/"))
        }
    }
    return @($set | Sort-Object)
}

function Deploy-WorktreeOverlay {
    Sync-Ops
    $files = Get-WorktreeOverlayFiles
    if ($files.Count -eq 0) {
        Write-Host "deploy-worktree: no modified or untracked files to upload"
        return
    }
    $branchName = Get-CurrentBranch
    $commit = Get-CurrentCommit
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stage = Join-Path $env:TEMP ("mrg_worktree_overlay_" + $stamp)
    $tmp = Join-Path $env:TEMP ("mrg_worktree_overlay_" + $stamp + ".tar")
    if (Test-Path -LiteralPath $stage) {
        Remove-Item -LiteralPath $stage -Recurse -Force
    }
    if (Test-Path -LiteralPath $tmp) {
        Remove-Item -LiteralPath $tmp -Force
    }
    New-Item -ItemType Directory -Path $stage | Out-Null
    foreach ($rel in $files) {
        $src = Join-Path $RepoRoot ($rel -replace "/", "\")
        $dst = Join-Path $stage ($rel -replace "/", "\")
        $dstDir = Split-Path -Parent $dst
        if (!(Test-Path -LiteralPath $dstDir)) {
            New-Item -ItemType Directory -Path $dstDir | Out-Null
        }
        Copy-Item -LiteralPath $src -Destination $dst -Force
        if ($rel.ToLowerInvariant().EndsWith(".sh")) {
            $text = [IO.File]::ReadAllText($dst)
            $text = $text -replace "`r`n", "`n"
            $text = $text -replace "`r", "`n"
            $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
            [IO.File]::WriteAllText($dst, $text, $utf8NoBom)
        }
    }
    $listPath = Join-Path $stage "__deploy_files.txt"
    $files | Set-Content -LiteralPath $listPath -Encoding ASCII
    Invoke-Checked -Exe "tar" -ArgList @("-cf", $tmp, "-C", $stage, "-T", $listPath)
    $remoteArchive = "/tmp/" + [IO.Path]::GetFileName($tmp)
    Invoke-Checked -Exe "scp" -ArgList @($ScpOptions + @($tmp, "$HostName`:$remoteArchive"))
    $deployName = "${branchName}+worktree"
    Invoke-Remote "bash remote_ops/deploy_branch_snapshot.sh $(Quote-Sh $remoteArchive) $(Quote-Sh $deployName) $(Quote-Sh $commit)"
    Write-Host "deploy-worktree: uploaded $($files.Count) files"
    foreach ($rel in $files) {
        Write-Host "  $rel"
    }
}

function Build-LaunchExtraArgs {
    param([string]$Mode, [string]$ResolvedEventReplayRawFiles)
    if ($DatasetDir -ne "" -and ($Mode -eq "smoke-event-replay" -or $Mode -eq "smoke-mvs-replay" -or $Mode -eq "smoke-full-replay")) {
        return Resolve-RemoteLaunchExtraArgs -Mode $Mode
    }
    $parts = @()
    $effectiveEventMode = $EventSourceMode
    if (($Mode -eq "smoke-event-replay" -or $Mode -eq "smoke-full-replay") -and $effectiveEventMode -eq "") {
        $effectiveEventMode = "file_replay"
    }
    if ($Mode -eq "smoke-live" -and $effectiveEventMode -eq "") {
        $effectiveEventMode = "live_hal"
    }
    if ($effectiveEventMode -ne "") {
        $parts += "--event-source-mode"
        $parts += $effectiveEventMode
    }
    if ($ResolvedEventReplayRawFiles -ne "") {
        $parts += "--event-file-fifo-broker-raw-files"
        $parts += $ResolvedEventReplayRawFiles
    }
    if ($ReplayTiming -ne "") {
        $parts += "--event-file-fifo-broker-replay-timing"
        $parts += $ReplayTiming
    }
    if ($AllowMissingExtTrigger) {
        $parts += "--event-file-fifo-broker-allow-missing-ext-trigger"
    }
    $effectiveMvsMode = $MvsSourceMode
    if (($Mode -eq "smoke-mvs-replay" -or $Mode -eq "smoke-full-replay") -and $effectiveMvsMode -eq "") {
        $effectiveMvsMode = "file_replay"
    }
    if ($effectiveMvsMode -ne "") {
        $parts += "--mvs-source-mode"
        $parts += $effectiveMvsMode
    }
    if ($effectiveMvsMode -eq "file_replay") {
        if ($DatasetDir -eq "") {
            throw "MVS file replay requires -DatasetDir."
        }
        $parts += "--mvs-file-frame-broker-dataset-dir"
        $parts += $DatasetDir
        $timing = if ($ReplayTiming -ne "") { $ReplayTiming } else { "realtime" }
        $parts += "--mvs-file-frame-broker-replay-timing"
        $parts += $timing
        $parts += "--mvs-file-frame-broker-loop"
    }
    return ($parts -join " ")
}

function ReplayMode-ToManifestMode {
    param([string]$Mode)
    if ($Mode -eq "smoke-event-replay") { return "event_replay" }
    if ($Mode -eq "smoke-mvs-replay") { return "mvs_replay" }
    if ($Mode -eq "smoke-full-replay") { return "full_replay" }
    if ($EventSourceMode -eq "file_replay" -and $MvsSourceMode -eq "file_replay") { return "full_replay" }
    if ($EventSourceMode -eq "file_replay") { return "event_replay" }
    if ($MvsSourceMode -eq "file_replay") { return "mvs_replay" }
    return "live"
}

function Resolve-RemoteLaunchExtraArgs {
    param([string]$Mode)
    if ($DatasetDir -eq "") {
        return ""
    }
    Sync-Ops
    $manifestMode = ReplayMode-ToManifestMode -Mode $Mode
    $timing = if ($ReplayTiming -ne "") { $ReplayTiming } else { "realtime" }
    $cmd = "~/miniconda3/envs/ids_recv/bin/python3 remote_ops/check_v25_dataset_manifest.py --dataset-dir $(Quote-Sh $DatasetDir) --mode $manifestMode --require-usable --print-launch-extra-args --replay-timing $timing"
    if ($AllowMissingExtTrigger) {
        $cmd += " --allow-missing-ext-trigger"
    }
    return (Invoke-RemoteCapture $cmd).Trim()
}

function Resolve-RemoteEventRawMapping {
    if ($EventReplayRawFiles -ne "") {
        return $EventReplayRawFiles
    }
    if ($DatasetDir -eq "") {
        return ""
    }
    Sync-Ops
    $cmd = "~/miniconda3/envs/ids_recv/bin/python3 remote_ops/check_v25_dataset_manifest.py --dataset-dir $(Quote-Sh $DatasetDir) --mode event_replay --print-event-raw-mapping"
    return (Invoke-RemoteCapture $cmd).Trim()
}

function Build-NativeBrokers {
    Sync-Ops
    if ($DeployWorktree) {
        Deploy-WorktreeOverlay
    }
    Invoke-Remote "bash remote_ops/build_native_brokers.sh"
}

function Smoke-MvsBrokerStandalone {
    if ($DatasetDir -eq "") {
        throw "Provide -DatasetDir for smoke-mvs-broker."
    }
    Sync-Ops
    $frames = [Math]::Max(1, $MaxFrames)
    $timing = if ($ReplayTiming -ne "") { $ReplayTiming } else { "fast" }
    $cmd = "~/miniconda3/envs/ids_recv/bin/python3 remote_ops/mvs_file_frame_broker.py --dataset-dir $(Quote-Sh $DatasetDir) --replay-timing $timing --max-frames $frames"
    Invoke-Remote $cmd
    Invoke-Remote "cat sync_ipc/mvs_file_frame_broker_status.json"
}

function Check-DatasetManifest {
    if ($DatasetDir -eq "") {
        throw "Provide -DatasetDir for check-dataset-manifest."
    }
    Sync-Ops
    $mode = "full_replay"
    if ($EventSourceMode -eq "file_replay" -and $MvsSourceMode -ne "file_replay") {
        $mode = "event_replay"
    } elseif ($MvsSourceMode -eq "file_replay" -and $EventSourceMode -ne "file_replay") {
        $mode = "mvs_replay"
    }
    $cmd = "~/miniconda3/envs/ids_recv/bin/python3 remote_ops/check_v25_dataset_manifest.py --dataset-dir $(Quote-Sh $DatasetDir) --mode $mode --write-resolved-manifest"
    if ($DatasetLabel -ne "") {
        $cmd += " --dataset-label $(Quote-Sh $DatasetLabel) --scenario-name $(Quote-Sh $DatasetLabel)"
    }
    if ($ExperimentNote -ne "") {
        $cmd += " --operator-notes $(Quote-Sh $ExperimentNote)"
    }
    if ($PrintResolvedManifest) {
        $cmd += " --print-resolved-manifest"
    }
    Invoke-Remote $cmd
}

function Write-V25ReplayControl {
    Sync-Ops
    $mode = $ReplayControlMode
    if ($mode -eq "") {
        if ($EventSourceMode -eq "file_replay" -and $MvsSourceMode -eq "file_replay") {
            $mode = "full_replay"
        } elseif ($EventSourceMode -eq "file_replay") {
            $mode = "event_replay"
        } elseif ($MvsSourceMode -eq "file_replay") {
            $mode = "mvs_replay"
        } elseif ($DatasetDir -ne "") {
            $mode = "full_replay"
        } else {
            $mode = "live"
        }
    }
    $timing = if ($ReplayTiming -ne "") { $ReplayTiming } else { "realtime" }
    $cmd = "~/miniconda3/envs/ids_recv/bin/python3 remote_ops/v25_replay_control_panel.py --project-root . --profile $(Quote-Sh $Profile) --mode $mode --replay-timing $timing --write --print-json"
    if ($DatasetDir -ne "") {
        $cmd += " --dataset-dir $(Quote-Sh $DatasetDir)"
    }
    if ($AllowMissingExtTrigger) {
        $cmd += " --allow-missing-ext-trigger"
    }
    Invoke-Remote $cmd
}

function Check-EventRawExtTrigger {
    if ($DatasetDir -eq "") {
        throw "Provide -DatasetDir for check-event-raw-exttrig."
    }
    Sync-Ops
    $cmd = "if [ ! -x native_event_bridge/build/event_file_fifo_broker ]; then bash remote_ops/build_native_brokers.sh; fi"
    $cmd += "; ~/miniconda3/envs/ids_recv/bin/python3 tools/check_event_raw_ext_trigger.py $(Quote-Sh $DatasetDir) --project-root . --expected-period-us $ExpectedPeriodUs --period-tolerance-us $PeriodToleranceUs --output-json sync_ipc/event_raw_ext_trigger_check_latest.json"
    if ($UpdateManifest) {
        $cmd += " --update-manifest"
    }
    if ($AllowMissingExtTrigger) {
        $cmd += " --allow-missing-ext-trigger"
    }
    Invoke-Remote $cmd
}

function List-Datasets {
    Sync-Ops
    Invoke-Remote "~/miniconda3/envs/ids_recv/bin/python3 remote_ops/list_v25_datasets.py --project . --limit $DatasetLimit"
}

function Build-RunEnvPrefix {
    param([string]$RunLabel, [string]$Mode, [string]$ResolvedEventReplayRawFiles)
    $parts = @("PROFILE=$(Quote-Sh $Profile)", "CONTROL_DELAY_S=$ControlDelay")
    $launchExtraArgs = Build-LaunchExtraArgs -Mode $Mode -ResolvedEventReplayRawFiles $ResolvedEventReplayRawFiles
    if ($launchExtraArgs -ne "") {
        $parts += "LAUNCH_EXTRA_ARGS=$(Quote-Sh $launchExtraArgs)"
    }
    if ($DatasetDir -ne "") {
        $parts += "V25_DATASET_DIR=$(Quote-Sh $DatasetDir)"
    }
    $effectiveMvsMode = $MvsSourceMode
    if (($Mode -eq "smoke-mvs-replay" -or $Mode -eq "smoke-full-replay") -and $effectiveMvsMode -eq "") {
        $effectiveMvsMode = "file_replay"
    }
    if ($effectiveMvsMode -ne "") {
        $parts += "MVS_SOURCE_MODE_REQUESTED=$(Quote-Sh $effectiveMvsMode)"
    }
    $effectivePersonDataset = $PersonDataset
    $effectiveEvidenceReplayDataset = [bool]$EvidenceReplayDataset
    if ($FullRecording) {
        $parts += "FULL_RECORDING_ENABLE=1"
        if ($effectivePersonDataset -eq "") {
            $effectivePersonDataset = "mixed"
        }
        $effectiveEvidenceReplayDataset = $true
    }
    if ($AuthorityMode -ne "") {
        $parts += "FINAL_HIT_AUTHORITY=$(Quote-Sh $AuthorityMode)"
        if ($AuthorityMode -eq "strong_reasoning") {
            $parts += "STRONG_REASONING_SHADOW_ONLY=false"
        } elseif ($AuthorityMode -eq "strong_gate") {
            $parts += "STRONG_REASONING_SHADOW_ONLY=true"
        }
    }
    if ($HumanNoiseMode -ne "") { $parts += "HUMAN_NOISE_FILTER_MODE=$(Quote-Sh $HumanNoiseMode)" }
    if ($BboxStartExpandPx -ne "") { $parts += "HUMAN_NOISE_TRACK_START_BBOX_EXPAND_PX=$(Quote-Sh $BboxStartExpandPx)" }
    if ($BboxStartPolicy -ne "") { $parts += "HUMAN_NOISE_TRACK_START_BBOX_POLICY=$(Quote-Sh $BboxStartPolicy)" }
    if ($ProjectedMaskFinalMode -ne "") { $parts += "PROJECTED_MASK_FINAL_MODE=$(Quote-Sh $ProjectedMaskFinalMode)" }
    if ($ProjectedMaskRequireTerminalOrTurn -ne "") { $parts += "PROJECTED_MASK_REQUIRE_TERMINAL_OR_TURN=$(Quote-Sh $ProjectedMaskRequireTerminalOrTurn)" }
    if ($SameTrackEpisodeDedupMs -ne "") { $parts += "SAME_TRACK_EPISODE_DEDUP_MS=$(Quote-Sh $SameTrackEpisodeDedupMs)" }
    if ($ProjectExtendPx -ne "") { $parts += "STRONG_REASONING_TRACK_PROJECT_EXTEND_PX=$(Quote-Sh $ProjectExtendPx)" }
    if ($ProjectMaxLastDistPx -ne "") { $parts += "STRONG_REASONING_TRACK_PROJECT_MAX_LAST_DIST_PX=$(Quote-Sh $ProjectMaxLastDistPx)" }
    if ($effectivePersonDataset -ne "") {
        $dtype = $effectivePersonDataset.ToLowerInvariant()
        if (@("negative", "positive", "mixed", "battle_1v1") -notcontains $dtype) {
            throw "PersonDataset must be negative, positive, mixed, or battle_1v1."
        }
        $dsLabel = if ($DatasetLabel -ne "") { $DatasetLabel } else { "${RunLabel}_${dtype}" }
        $datasetKind = if ($effectiveEvidenceReplayDataset) { "evidence_replay" } else { "person_id" }
        $parts += "PERSON_ID_DATASET_ENABLE=1"
        $parts += "PERSON_ID_DATASET_KIND=$(Quote-Sh $datasetKind)"
        $parts += "PERSON_ID_DATASET_TYPE=$(Quote-Sh $dtype)"
        $parts += "PERSON_ID_DATASET_LABEL=$(Quote-Sh $dsLabel)"
        $parts += "PERSON_ID_DATASET_START_DELAY_S=$ControlDelay"
        $parts += "PERSON_ID_DATASET_STOP_MARGIN_S=$DatasetStopMargin"
        if ($effectiveEvidenceReplayDataset) {
            $parts += "PERSON_ID_DATASET_INCLUDE_MASK_POLYGONS=true"
            $parts += "PERSON_ID_DATASET_INCLUDE_BULLET_TRAJECTORY=true"
            $parts += "PERSON_ID_DATASET_INCLUDE_HIT_DEBUG=true"
        }
    }
    return ($parts -join " ")
}

function Start-RemoteRun {
    param([string]$Mode)
    Sync-Ops
    if ($DeployWorktree) {
        Deploy-WorktreeOverlay
    }
    $defaultLabel = switch ($Mode) {
        "stress" { "stress_remote_ops" }
        "smoke-live" { "v25_live_smoke" }
        "smoke-event-replay" { "v25_event_replay_smoke" }
        "smoke-mvs-replay" { "v25_mvs_replay_smoke" }
        "smoke-full-replay" { "v25_full_replay_smoke" }
        "longrun" { "longrun_remote_ops" }
        default { "smoke_remote_ops" }
    }
    $runLabel = Get-SafeLabel -Value $Label -Default $defaultLabel
    $runDuration = $Duration
    if ($LongRun -or $Mode -eq "longrun") {
        $runDuration = 0
    } elseif ($runDuration -le 0) {
        $runDuration = if ($Mode -eq "stress") { 1800 } else { 370 }
    }
    $resolvedEventReplayRawFiles = ""
    if (($Mode -eq "smoke-event-replay" -or $Mode -eq "smoke-full-replay") -and $DatasetDir -eq "") {
        $resolvedEventReplayRawFiles = Resolve-RemoteEventRawMapping
        if ($resolvedEventReplayRawFiles -eq "") {
            throw "$Mode requires -EventReplayRawFiles or a -DatasetDir with discoverable Event RAW files."
        }
    }
    $envPrefix = Build-RunEnvPrefix -RunLabel $runLabel -Mode $Mode -ResolvedEventReplayRawFiles $resolvedEventReplayRawFiles
    if ($Wait -or $FetchAfter -or $SummaryAfter) {
        Invoke-Remote "$envPrefix bash remote_ops/run_fullsys_test.sh $runLabel $runDuration"
        $remoteRun = (Invoke-RemoteCapture "cat sync_ipc/latest_${runLabel}_run_dir.txt").Trim()
        if ($SummaryAfter) {
            Show-SummaryForRun $remoteRun
        }
        if ($FetchAfter) {
            Fetch-RunForDir $remoteRun
        }
        return
    }
    $cmd = "$envPrefix setsid -f bash remote_ops/run_fullsys_test.sh $runLabel $runDuration >/tmp/${runLabel}_start.out 2>&1 < /dev/null; sleep 1; cat sync_ipc/latest_${runLabel}_run_dir.txt 2>/dev/null; cat /tmp/${runLabel}_start.out 2>/dev/null"
    Invoke-Remote $cmd
}

function Get-RemoteRunDir {
    if ($RunDir -ne "") { return $RunDir }
    if ($Label -eq "") { throw "Provide -Label or -RunDir." }
    return (Invoke-RemoteCapture "cat sync_ipc/latest_${Label}_run_dir.txt").Trim()
}

function Show-Summary {
    Sync-Ops
    $remoteRun = Get-RemoteRunDir
    Show-SummaryForRun $remoteRun
}

function Show-SummaryForRun {
    param([string]$RemoteRun)
    $compact = if ($FullSummary) { "" } else { " --compact" }
    Invoke-Remote "~/miniconda3/envs/ids_recv/bin/python3 remote_ops/check_fullsys_run.py summary --project . --run-dir $RemoteRun$compact"
}

function Assess-Run {
    Sync-Ops
    $remoteRun = Get-RemoteRunDir
    $summaryText = Invoke-RemoteCapture "~/miniconda3/envs/ids_recv/bin/python3 remote_ops/check_fullsys_run.py summary --project . --run-dir $remoteRun --compact"
    if (!(Test-Path -LiteralPath $LocalOutDir)) {
        New-Item -ItemType Directory -Path $LocalOutDir | Out-Null
    }
    $runName = Split-Path -Leaf (($remoteRun -replace "`r", "" -replace "`n", "").Trim())
    if ($runName -eq "") {
        $runName = Get-SafeLabel -Value $Label -Default "fullsys_run"
    }
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $prefix = Join-Path $LocalOutDir "${runName}_assessment_${stamp}"
    $summaryPath = "$prefix.compact.json"
    $assessmentJson = "$prefix.assessment.json"
    $assessmentMd = "$prefix.assessment.md"
    $summaryBody = [string]::Join("`n", @($summaryText))
    Write-TextNoBom -Path $summaryPath -Text ($summaryBody.Trim() + "`n")
    $nodeName = if ($AssessmentNode -ne "") { $AssessmentNode } elseif ($Label -ne "") { $Label } else { $runName }
    $py = Resolve-LocalPython
    Invoke-Checked -Exe $py -ArgList @(
        (Join-Path $RepoRoot "tools\evaluate_fullsys_node.py"),
        "--input", $summaryPath,
        "--node", $nodeName,
        "--out-json", $assessmentJson,
        "--out-md", $assessmentMd
    )
    Write-Host "[remote_test][assess] compact_json=$summaryPath"
    Write-Host "[remote_test][assess] assessment_json=$assessmentJson"
    Write-Host "[remote_test][assess] assessment_md=$assessmentMd"
    if (Test-Path -LiteralPath $assessmentMd) {
        Get-Content -LiteralPath $assessmentMd -TotalCount 80
    }
}

function Stop-RemoteRun {
    $runLabel = if ($Label -eq "") { "__latest__" } else { $Label }
    try {
        Sync-Ops
        Invoke-Remote "STOP_REASON=$(Quote-Sh 'operator_stop_from_tools_remote_test') bash remote_ops/stop_fullsys_test.sh $runLabel $StopWait"
    } catch {
        Write-Warning "SSH stop failed: $($_.Exception.Message)"
        Write-Warning "Falling back to SMB stop request. This requires the full-system runner to be new enough to watch stop_requested.json."
        Request-StopViaSmb -RunLabel $runLabel -WaitSeconds $StopWait -Reason "operator_stop_smb_fallback_after_ssh_failure"
    }
}

function Start-LatestSystem {
    Sync-Ops
    if ($DeployWorktree) {
        Deploy-WorktreeOverlay
    }
    $safeLabel = if ($Label -ne "") { Get-SafeLabel -Value $Label -Default 'desktop_latest' } else { "desktop_latest" }
    $envPrefix = "PROFILE=$(Quote-Sh $Profile)"
    if ($Label -ne "") {
        $envPrefix = "$envPrefix LABEL=$(Quote-Sh $safeLabel)"
    }
    Invoke-Remote "$envPrefix bash remote_ops/start_latest_system.sh"
    if (!$NoReadyWait) {
        Invoke-Remote "LABEL=$(Quote-Sh $safeLabel) WAIT_S=$ReadyWait bash remote_ops/wait_latest_system_ready.sh"
    }
}

function Install-DesktopShortcuts {
    Sync-Ops
    & (Join-Path $PSScriptRoot "install_desktop_shortcuts.ps1") `
        -HostName $HostName `
        -RemoteProject $RemoteProject `
        -RemoteOnly
    if (!$?) {
        throw "desktop shortcut installation failed"
    }
}

function Open-StartupPanel {
    Sync-Ops
    Invoke-Remote "DISPLAY=:1 nohup bash remote_ops/open_startup_control_panel.sh >/dev/null 2>&1 </dev/null &"
    Start-Sleep -Seconds 2
    Invoke-Remote "pgrep -af 'v25_replay_control_panel.py.*--gui' || { tail -n 40 sync_ipc/v25_startup_control_panel.log 2>/dev/null; exit 1; }"
}

function Restart-StartupPanel {
    Sync-Ops
    Invoke-Remote "pkill -f '[v]25_replay_control_panel.py.*--gui' || true"
    Start-Sleep -Seconds 1
    Open-StartupPanel
}

function Check-StartupPanel {
    Sync-Ops
    Invoke-Remote "bash -n remote_ops/start_latest_system.sh"
    Invoke-Remote "~/miniconda3/envs/ids_recv/bin/python3 remote_ops/v25_replay_control_panel.py --project-root . --profile $(Quote-Sh $Profile) --mode live --replay-timing realtime --write"
    Invoke-Remote "~/miniconda3/envs/ids_recv/bin/python3 remote_ops/start_system_from_panel.py --project-root . --control sync_ipc/v25_replay_status.json --status sync_ipc/v25_panel_launch_status.json --label desktop_latest --dry-run"
}

function Stop-LatestSystem {
    $envPrefix = ""
    if ($Label -ne "") {
        $envPrefix = "LABEL=$(Quote-Sh (Get-SafeLabel -Value $Label -Default 'desktop_latest')) "
    }
    try {
        Sync-Ops
        Invoke-Remote "${envPrefix}bash remote_ops/stop_latest_system.sh $StopWait"
    } catch {
        Write-Warning "SSH stop-latest failed: $($_.Exception.Message)"
        Write-Warning "Writing SMB safe-off control; persistent latest-system process still needs SSH or local desktop stop if it is not managed by run_fullsys_test.sh."
        $runLabel = if ($Label -eq "") { "desktop_latest" } else { Get-SafeLabel -Value $Label -Default 'desktop_latest' }
        Request-StopViaSmb -RunLabel $runLabel -WaitSeconds 5 -Reason "operator_stop_latest_smb_safe_off_after_ssh_failure"
    }
}

function Fetch-Run {
    $remoteRun = Get-RemoteRunDir
    Fetch-RunForDir $remoteRun
}

function Fetch-RunForDir {
    param([string]$RemoteRun)
    if (!(Test-Path -LiteralPath $LocalOutDir)) {
        New-Item -ItemType Directory -Path $LocalOutDir | Out-Null
    }
    $archive = "$RemoteRun.tar.gz"
    Invoke-Checked -Exe "scp" -ArgList @($ScpOptions + @("$HostName`:$RemoteProject/$archive", $LocalOutDir))
}

function Resolve-HotUpdateProfile {
    param([string]$ProfilePath, [string]$Kind)
    if ($ProfilePath -eq "") { return "" }
    $candidate = $ProfilePath
    if (!(Test-Path -LiteralPath $candidate)) {
        $candidate = Join-Path $RepoRoot $ProfilePath
    }
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $safeName = (Split-Path -Leaf $candidate) -replace "[^A-Za-z0-9_.-]+", "_"
        if ($safeName -eq "") { $safeName = "${Kind}_profile.json" }
        Invoke-Remote 'mkdir -p "$REMOTE_PROJECT/sync_ipc/remote_hot_update_profiles"'
        $remoteRel = "sync_ipc/remote_hot_update_profiles/$safeName"
        Invoke-Checked -Exe "scp" -ArgList @($ScpOptions + @($candidate, "$HostName`:$RemoteProject/$remoteRel"))
        return $remoteRel
    }
    return $ProfilePath
}

function Show-LiveMonitor {
    Sync-Ops
    $parts = @("~/miniconda3/envs/ids_recv/bin/python3 remote_ops/check_fullsys_run.py live-status --project . --write-latest --compact")
    if ($RunDir -ne "") {
        $parts += "--run-dir $(Quote-Sh $RunDir)"
    }
    if ($Label -ne "") {
        $parts += "--label $(Quote-Sh (Get-SafeLabel -Value $Label -Default 'desktop_latest'))"
    }
    Invoke-Remote ($parts -join " ")
    $statusPath = Join-Path $SmbProject "sync_ipc\test_run_status_latest.json"
    if (Test-Path -LiteralPath $statusPath) {
        Write-Host "[remote_test][monitor] status_json=$statusPath"
    }
}

function Invoke-HotUpdate {
    Sync-Ops
    $remoteHitProfile = Resolve-HotUpdateProfile -ProfilePath $HitProfile -Kind "hit"
    $remoteGlobalProfile = Resolve-HotUpdateProfile -ProfilePath $GlobalIdProfile -Kind "global_id"
    $parts = @("~/miniconda3/envs/ids_recv/bin/python3 remote_ops/check_fullsys_run.py set-experiment --project . --source tools_remote_test_hot_update")
    if ($ExperimentId -ne "") { $parts += "--experiment-id $(Quote-Sh $ExperimentId)" }
    if ($ExperimentStage -ne "") { $parts += "--stage $(Quote-Sh $ExperimentStage)" }
    if ($ExperimentNote -ne "") { $parts += "--note $(Quote-Sh $ExperimentNote)" }
    if ($remoteHitProfile -ne "") { $parts += "--hit-profile $(Quote-Sh $remoteHitProfile)" }
    if ($remoteGlobalProfile -ne "") { $parts += "--global-id-profile $(Quote-Sh $remoteGlobalProfile)" }
    if ($AuthorityMode -ne "") { $parts += "--authority-mode $(Quote-Sh $AuthorityMode)" }
    if ($HumanNoiseMode -ne "") { $parts += "--human-noise-mode $(Quote-Sh $HumanNoiseMode)" }
    if ($BboxStartExpandPx -ne "") { $parts += "--bbox-start-expand-px $(Quote-Sh $BboxStartExpandPx)" }
    if ($BboxStartPolicy -ne "") { $parts += "--bbox-start-policy $(Quote-Sh $BboxStartPolicy)" }
    if ($ProjectedMaskFinalMode -ne "") { $parts += "--projected-mask-final-mode $(Quote-Sh $ProjectedMaskFinalMode)" }
    if ($ProjectedMaskRequireTerminalOrTurn -ne "") { $parts += "--projected-mask-require-terminal-or-turn $(Quote-Sh $ProjectedMaskRequireTerminalOrTurn)" }
    if ($SameTrackEpisodeDedupMs -ne "") { $parts += "--same-track-episode-dedup-ms $(Quote-Sh $SameTrackEpisodeDedupMs)" }
    if ($ProjectExtendPx -ne "") { $parts += "--strong-reasoning-track-project-extend-px $(Quote-Sh $ProjectExtendPx)" }
    if ($ProjectMaxLastDistPx -ne "") { $parts += "--strong-reasoning-track-project-max-last-dist-px $(Quote-Sh $ProjectMaxLastDistPx)" }
    if ($GlobalPersonTarget -ne "") { $parts += "--global-person-target $(Quote-Sh $GlobalPersonTarget)" }
    if ($GlobalPersonFilterBackend -ne "") { $parts += "--global-person-filter-backend $(Quote-Sh $GlobalPersonFilterBackend)" }
    Invoke-Remote ($parts -join " ")
    Show-LiveMonitor
    $statusPath = Join-Path $SmbProject "sync_ipc\experiment_status_latest.json"
    if (Test-Path -LiteralPath $statusPath) {
        Write-Host "[remote_test][hot-update] experiment_status_json=$statusPath"
    }
}

switch ($Action) {
    "check" { & (Join-Path $PSScriptRoot "dev_check.ps1") }
    "sync-ops" { Sync-Ops }
    "deploy" { Deploy-Snapshot }
    "deploy-worktree" { Deploy-WorktreeOverlay }
    "validate" {
        Sync-Ops
        Invoke-Remote "bash remote_ops/validate_remote_env.sh"
    }
    "install-desktop-shortcuts" { Install-DesktopShortcuts }
    "open-startup-panel" { Open-StartupPanel }
    "restart-startup-panel" { Restart-StartupPanel }
    "check-startup-panel" { Check-StartupPanel }
    "build-native-brokers" { Build-NativeBrokers }
    "smoke-mvs-broker" { Smoke-MvsBrokerStandalone }
    "list-datasets" { List-Datasets }
    "check-dataset-manifest" { Check-DatasetManifest }
    "v25-replay-control" { Write-V25ReplayControl }
    "check-event-raw-exttrig" { Check-EventRawExtTrigger }
    "smoke" { Start-RemoteRun "smoke" }
    "smoke-live" { Start-RemoteRun "smoke-live" }
    "smoke-event-replay" { Start-RemoteRun "smoke-event-replay" }
    "smoke-mvs-replay" { Start-RemoteRun "smoke-mvs-replay" }
    "smoke-full-replay" { Start-RemoteRun "smoke-full-replay" }
    "stress" { Start-RemoteRun "stress" }
    "longrun" { Start-RemoteRun "longrun" }
    "start-latest" { Start-LatestSystem }
    "stop-latest" { Stop-LatestSystem }
    "stop" { Stop-RemoteRun }
    "stop-smb" {
        $runLabel = if ($Label -eq "") { "__latest__" } else { $Label }
        Request-StopViaSmb -RunLabel $runLabel -WaitSeconds $StopWait -Reason "operator_stop_smb_explicit"
    }
    "summary" { Show-Summary }
    "assess" { Assess-Run }
    "fetch" { Fetch-Run }
    "monitor" { Show-LiveMonitor }
    "hot-update" { Invoke-HotUpdate }
}
