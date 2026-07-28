param(
    [string]$HostName = "ysxq@192.168.31.115",
    [string]$RemoteProject = "~/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627",
    [string]$Label = "desktop_latest"
)

$ErrorActionPreference = "Stop"
$remoteTest = Join-Path $PSScriptRoot "remote_test.ps1"

Write-Host "Stopping latest system on $HostName"
& powershell -ExecutionPolicy Bypass -File $remoteTest stop-latest -HostName $HostName -RemoteProject $RemoteProject -Label $Label
if ($LASTEXITCODE -ne 0) {
    throw "one-click stop failed with exit code $LASTEXITCODE"
}
