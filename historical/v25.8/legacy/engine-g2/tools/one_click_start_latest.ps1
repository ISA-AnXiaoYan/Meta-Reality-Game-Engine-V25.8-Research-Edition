param(
    [string]$HostName = "ysxq@192.168.31.115",
    [string]$RemoteProject = "~/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627",
    [string]$Profile = "launch_profiles/627_event_overlay_bev.json",
    [string]$Label = "desktop_latest",
    [switch]$DeployWorktree
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$remoteTest = Join-Path $PSScriptRoot "remote_test.ps1"

$argsList = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $remoteTest,
    "start-latest",
    "-HostName", $HostName,
    "-RemoteProject", $RemoteProject,
    "-Profile", $Profile,
    "-Label", $Label
)
if ($DeployWorktree) {
    $argsList += "-DeployWorktree"
}

Write-Host "Starting latest system on $HostName"
Write-Host "Repo: $RepoRoot"
Write-Host "Profile: $Profile"
& powershell @argsList
if ($LASTEXITCODE -ne 0) {
    throw "one-click start failed with exit code $LASTEXITCODE"
}
