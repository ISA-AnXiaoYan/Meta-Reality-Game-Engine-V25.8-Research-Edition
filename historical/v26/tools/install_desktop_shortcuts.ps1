param(
    [string]$HostName = "ysxq@192.168.31.115",
    [string]$RemoteProject = "~/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627",
    [string]$LocalDesktop = "",
    [switch]$RemoteOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if ($LocalDesktop -eq "") {
    $LocalDesktop = [Environment]::GetFolderPath("Desktop")
}

function Invoke-Checked {
    param([string]$Exe, [string[]]$ArgList)
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe failed with exit code $LASTEXITCODE"
    }
}

function Write-Utf8NoBom {
    param([string]$Path, [string]$Value)
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($Path, $Value, $utf8NoBom)
}

if (!$RemoteOnly) {
    if (!(Test-Path -LiteralPath $LocalDesktop)) {
        New-Item -ItemType Directory -Path $LocalDesktop | Out-Null
    }

    $startCmdPath = Join-Path $LocalDesktop "MRG-OneClick-Start.cmd"
    $stopCmdPath = Join-Path $LocalDesktop "MRG-OneClick-Stop.cmd"

    $startCmd = @"
@echo off
cd /d "$RepoRoot"
powershell -ExecutionPolicy Bypass -File "$RepoRoot\tools\one_click_start_latest.ps1"
pause
"@
    $stopCmd = @"
@echo off
cd /d "$RepoRoot"
powershell -ExecutionPolicy Bypass -File "$RepoRoot\tools\one_click_stop_latest.ps1"
pause
"@

    Set-Content -LiteralPath $startCmdPath -Value $startCmd -Encoding ASCII
    Set-Content -LiteralPath $stopCmdPath -Value $stopCmd -Encoding ASCII
    Write-Host "local_start=$startCmdPath"
    Write-Host "local_stop=$stopCmdPath"
}

$remoteStart = @"
[Desktop Entry]
Type=Application
Name=MRG OneClick Start
Comment=Start latest fusion system profile
Exec=bash -lc 'cd $RemoteProject && bash remote_ops/start_latest_system.sh; echo; read -n 1 -s -r -p "Press any key to close..."'
Terminal=true
Categories=Utility;
"@
$remoteStop = @"
[Desktop Entry]
Type=Application
Name=MRG OneClick Stop
Comment=Stop latest fusion system profile
Exec=bash -lc 'cd $RemoteProject && bash remote_ops/stop_latest_system.sh 90; echo; read -n 1 -s -r -p "Press any key to close..."'
Terminal=true
Categories=Utility;
"@

$remoteProjectAbs = (& ssh $HostName "bash -lc 'cd $RemoteProject && pwd'").Trim()
if ($LASTEXITCODE -ne 0 -or $remoteProjectAbs -eq "") {
    throw "failed to resolve remote project path"
}

$remoteControlPanel = @"
[Desktop Entry]
Version=1.0
Type=Application
Name=元现实游戏引擎 - 启动控制面板
Comment=选择实时或离线数据源并生成可复盘启动配置
Exec=$remoteProjectAbs/remote_ops/open_startup_control_panel.sh
Path=$remoteProjectAbs
Icon=system-run
Terminal=false
StartupNotify=true
Categories=Utility;
"@

$tmpStart = Join-Path $env:TEMP "mrg_start_latest.desktop"
$tmpStop = Join-Path $env:TEMP "mrg_stop_latest.desktop"
$tmpControlPanel = Join-Path $env:TEMP "mrg_startup_control_panel.desktop"
Write-Utf8NoBom -Path $tmpStart -Value $remoteStart
Write-Utf8NoBom -Path $tmpStop -Value $remoteStop
Write-Utf8NoBom -Path $tmpControlPanel -Value $remoteControlPanel

$remoteDesktop = (& ssh $HostName 'xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop"').Trim()
if ($LASTEXITCODE -ne 0 -or $remoteDesktop -eq "") {
    throw "failed to resolve remote desktop path"
}

Invoke-Checked -Exe "ssh" -ArgList @($HostName, "mkdir -p '$remoteDesktop'")
Invoke-Checked -Exe "ssh" -ArgList @($HostName, "mkdir -p '$remoteProjectAbs/remote_ops'")
Invoke-Checked -Exe "scp" -ArgList @((Join-Path $RepoRoot "remote_ops\open_startup_control_panel.sh"), "$HostName`:$remoteProjectAbs/remote_ops/open_startup_control_panel.sh")
Invoke-Checked -Exe "scp" -ArgList @($tmpStart, "$HostName`:$remoteDesktop/MRG-OneClick-Start.desktop")
Invoke-Checked -Exe "scp" -ArgList @($tmpStop, "$HostName`:$remoteDesktop/MRG-OneClick-Stop.desktop")
Invoke-Checked -Exe "scp" -ArgList @($tmpControlPanel, "$HostName`:$remoteDesktop/MRG-Startup-Control-Panel.desktop")
Invoke-Checked -Exe "ssh" -ArgList @($HostName, "bash -n '$remoteProjectAbs/remote_ops/open_startup_control_panel.sh'")
Invoke-Checked -Exe "ssh" -ArgList @($HostName, "chmod +x '$remoteProjectAbs/remote_ops/open_startup_control_panel.sh'")
Invoke-Checked -Exe "ssh" -ArgList @($HostName, "chmod +x '$remoteDesktop/MRG-OneClick-Start.desktop' '$remoteDesktop/MRG-OneClick-Stop.desktop'")
Invoke-Checked -Exe "ssh" -ArgList @($HostName, "chmod +x '$remoteDesktop/MRG-Startup-Control-Panel.desktop'")

& ssh $HostName "gio set '$remoteDesktop/MRG-Startup-Control-Panel.desktop' metadata::trusted true"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "remote desktop did not accept metadata::trusted; use Allow Launching once if the desktop asks"
}

Write-Host "remote_desktop=$remoteDesktop"
Write-Host "remote_start=$remoteDesktop/MRG-OneClick-Start.desktop"
Write-Host "remote_stop=$remoteDesktop/MRG-OneClick-Stop.desktop"
Write-Host "remote_control_panel=$remoteDesktop/MRG-Startup-Control-Panel.desktop"
