[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ServerRoot,
    [string]$BridgePlugin = "",
    [string]$IpcExtension = "",
    [switch]$RequireRunning
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath($ServerRoot)
if (-not (Test-Path -LiteralPath $root -PathType Container)) {
    throw "测试服目录不存在：$root"
}

$srcds = Join-Path $root "srcds.exe"
$gameDir = Join-Path $root "csgo"
$sourcemod = Join-Path $gameDir "addons\sourcemod"
$get5 = Join-Path $sourcemod "plugins\get5.smx"
if (-not (Test-Path -LiteralPath $srcds -PathType Leaf)) {
    throw "缺少 32 位 Source Dedicated Server：$srcds"
}
if (-not (Test-Path -LiteralPath $gameDir -PathType Container)) {
    throw "缺少 csgo 目录：$gameDir"
}
if (-not (Test-Path -LiteralPath $sourcemod -PathType Container)) {
    throw "缺少 SourceMod 目录：$sourcemod"
}
if (-not (Test-Path -LiteralPath $get5 -PathType Leaf)) {
    throw "缺少 Get5 插件：$get5"
}

if (-not [string]::IsNullOrWhiteSpace($BridgePlugin)) {
    if (-not (Test-Path -LiteralPath $BridgePlugin -PathType Leaf)) {
        throw "缺少 SL-Bots SourcePawn 插件：$BridgePlugin"
    }
}
if (-not [string]::IsNullOrWhiteSpace($IpcExtension)) {
    if (-not (Test-Path -LiteralPath $IpcExtension -PathType Leaf)) {
        throw "缺少 SL-Bots 32 位 IPC 扩展：$IpcExtension"
    }
}

$processes = @(Get-Process -Name "srcds" -ErrorAction SilentlyContinue)
if ($RequireRunning -and $processes.Count -eq 0) {
    throw "要求测试服运行，但未发现 srcds.exe 进程"
}

$requiredFiles = @($srcds, $gameDir, $sourcemod, $get5)
if (-not [string]::IsNullOrWhiteSpace($BridgePlugin)) { $requiredFiles += [IO.Path]::GetFullPath($BridgePlugin) }
if (-not [string]::IsNullOrWhiteSpace($IpcExtension)) { $requiredFiles += [IO.Path]::GetFullPath($IpcExtension) }
$hashes = foreach ($file in $requiredFiles) {
    $item = Get-Item -LiteralPath $file
    if ($item.PSIsContainer) {
        continue
    }
    $hash = Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256
    [pscustomobject]@{
        path = $item.FullName
        sha256 = $hash.Hash.ToLowerInvariant()
        length = $item.Length
    }
}

[pscustomobject]@{
    server_root = $root
    tickrate_required = 128
    map_required = "de_mirage"
    get5_present = $true
    srcds_running = $processes.Count -gt 0
    files = @($hashes)
} | ConvertTo-Json -Depth 4
