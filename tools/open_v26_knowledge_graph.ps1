# SPDX-License-Identifier: AGPL-3.0-only
[CmdletBinding()]
param(
    [string]$ViewerRoot = (Join-Path $env:TEMP "mrge-v26-knowledge-graph-viewer")
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$graphRoot = Join-Path $repoRoot "historical\knowledge-graph\v26"
foreach ($file in @("knowledge-graph.json", "meta.json")) {
    if (-not (Test-Path (Join-Path $graphRoot $file))) {
        throw "Missing published graph file: $file"
    }
}

$uaDir = Join-Path $ViewerRoot ".ua"
New-Item -ItemType Directory -Force -Path $uaDir | Out-Null
Copy-Item (Join-Path $graphRoot "knowledge-graph.json") $uaDir -Force
Copy-Item (Join-Path $graphRoot "meta.json") $uaDir -Force
Set-Content -LiteralPath (Join-Path $uaDir "config.json") -Encoding utf8 -Value '{"outputLanguage":"zh"}'

Write-Host "Starting the pinned public V26 graph viewer. It only serves local files."
npx --yes "https://github.com/Egonex-AI/Understand-Anything/releases/download/v2.9.4/understand-anything-viewer.tgz" $ViewerRoot
