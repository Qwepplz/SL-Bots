[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ServerRoot,
    [string]$BridgePlugin = "",
    [string]$IpcExtension = "",
    [ValidateRange(1, 32)]
    [int]$InstanceCount = 1,
    [ValidateRange(1024, 65503)]
    [int]$RconPortBase = 27100,
    [string]$RunId = "",
    [switch]$RequireRunning,
    [switch]$RequireRcon
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath($ServerRoot)
if (-not (Test-Path -LiteralPath $root -PathType Container)) {
    throw "测试服目录不存在：$root"
}
if (($RconPortBase + $InstanceCount) -gt 65535) {
    throw "RCON 端口范围超出 TCP 端口上限"
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

$instances = @()
$ports = @()
if (-not [string]::IsNullOrWhiteSpace($RunId)) {
    for ($index = 1; $index -le $InstanceCount; $index++) {
        $instanceId = "{0:D2}" -f $index
        $instancePorts = @(
            [int](27100 + $index)
            [int](27200 + $index)
            [int](27300 + $index)
            [int](27400 + $index)
        )
        $rconPort = [int]$RconPortBase + [int]$index
        $ports += $instancePorts
        # Source 1 CS:GO exposes RCON on the same TCP endpoint as -port.
        $configPath = Join-Path $gameDir "cfg\get5\slbots\$RunId\instance-$instanceId.json"
        $instances += [pscustomobject]@{
            instance_id = $instanceId
            ports = $instancePorts
            rcon_port = $rconPort
            ipc_name = "SLBots_de_mirage_$instanceId"
            get5_config_path = [IO.Path]::GetFullPath($configPath)
            get5_config_present = Test-Path -LiteralPath $configPath -PathType Leaf
        }
    }
    if (@($ports | Select-Object -Unique).Count -ne $ports.Count) {
        throw "专服端口发生冲突"
    }
}

$processes = @(Get-Process -Name "srcds" -ErrorAction SilentlyContinue)
if ($RequireRunning -and $processes.Count -eq 0) {
    throw "要求测试服运行，但未发现 srcds.exe 进程"
}

$rconReady = $null
if ($RequireRcon) {
    $rconReady = @()
    foreach ($instance in $instances) {
        $probe = Test-NetConnection -ComputerName "127.0.0.1" -Port $instance.rcon_port -InformationLevel Quiet -WarningAction SilentlyContinue
        $rconReady += [bool]$probe
    }
    if ($rconReady -contains $false) {
        throw "至少一个测试服 RCON 端口无响应"
    }
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
    rcon_port_base = $RconPortBase
    rcon_required = [bool]$RequireRcon
    rcon_ready = $rconReady
    get5_present = $true
    srcds_running = $processes.Count -gt 0
    instances = @($instances)
    files = @($hashes)
} | ConvertTo-Json -Depth 4
