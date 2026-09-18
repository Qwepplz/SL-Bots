[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Config,
    [Parameter(Mandatory = $true)]
    [string]$DemoRoot,
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [Parameter(Mandatory = $true)]
    [string]$ServerRoot
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $scriptRoot ".."))
$python = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$configPath = [IO.Path]::GetFullPath($Config)
$demoPath = [IO.Path]::GetFullPath($DemoRoot)
$dataPath = [IO.Path]::GetFullPath($DataRoot)
$serverPath = [IO.Path]::GetFullPath($ServerRoot)

function Test-UnderPath {
    param(
        [Parameter(Mandatory = $true)][string]$Child,
        [Parameter(Mandatory = $true)][string]$Parent
    )
    $childPath = [IO.Path]::GetFullPath($Child).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $parentPath = [IO.Path]::GetFullPath($Parent).TrimEnd([IO.Path]::DirectorySeparatorChar)
    return $childPath.Equals($parentPath, [StringComparison]::OrdinalIgnoreCase) -or
        $childPath.StartsWith($parentPath + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)
}

function Write-AtomicJson {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][object]$Value
    )
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $temporary = "$Path.$PID.$([Guid]::NewGuid().ToString('N')).tmp"
    $Value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Stop-TrackedProcessTree {
    param(
        [Parameter(Mandatory = $true)][int]$RootPid
    )
    $root = Get-Process -Id $RootPid -ErrorAction SilentlyContinue
    if ($null -eq $root) {
        return
    }
    & taskkill.exe /PID $RootPid /T /F | Out-Null
}

if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "test-only config does not exist: $configPath"
}
if (-not (Test-Path -LiteralPath $demoPath -PathType Container)) {
    throw "Demo root does not exist: $demoPath"
}
if (Test-UnderPath -Child $dataPath -Parent $demoPath) {
    throw "data root must be outside the read-only Demo root"
}
if (-not (Test-Path -LiteralPath $serverPath -PathType Container)) {
    throw "server root does not exist: $serverPath"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "bundled Python runtime does not exist: $python"
}

$srcds = [IO.Path]::GetFullPath((Join-Path $serverPath "srcds.exe"))
if (-not (Test-Path -LiteralPath $srcds -PathType Leaf)) {
    throw "absolute srcds.exe is required: $srcds"
}

# Check the static server contract and all reserved RCON ports before launch.
& (Join-Path $scriptRoot "validate-test-server.ps1") `
    -ServerRoot $serverPath `
    -InstanceCount 32 `
    -RconPortBase 27100 | Out-Null
foreach ($rconPort in 27101..27132) {
    $inUse = Test-NetConnection -ComputerName "127.0.0.1" -Port $rconPort `
        -InformationLevel Quiet -WarningAction SilentlyContinue
    if ($inUse) {
        throw "RCON port is already in use: $rconPort"
    }
}

$freeBytes = (Get-PSDrive -Name ([IO.Path]::GetPathRoot($dataPath).TrimEnd('\').Substring(0, 1))).Free
if ($freeBytes -lt 10GB) {
    throw "data volume has less than 10 GB free"
}

$gpuSmoke = @'
import importlib.util
import torch
if not torch.cuda.is_available():
    raise SystemExit("GPU smoke requires a CUDA/HIP device")
if importlib.util.find_spec("onnxruntime") is None:
    raise SystemExit("onnxruntime is required")
import onnxruntime as ort
providers = tuple(ort.get_available_providers())
if "CPUExecutionProvider" not in providers or len(providers) < 2:
    raise SystemExit("dual ONNX providers are not available")
print(",".join(providers))
'@
& $python -c $gpuSmoke
if ($LASTEXITCODE -ne 0) {
    throw "GPU/dual-ONNX preflight failed"
}

$statePath = Join-Path $dataPath "state\test-run-state.json"
$runId = "test-only-$PID"
$state = [ordered]@{
    schema = "test-training-run-state-v1"
    run_id = $runId
    phase = "preflight_complete"
    demo_root = $demoPath
    data_root = $dataPath
    server_root = $serverPath
    active_pointer = Join-Path $dataPath "models\active.json"
    pending_pointer = Join-Path $dataPath "models\pending.json"
    recorded_pids = @()
}
Write-AtomicJson -Path $statePath -Value $state

$child = $null
$stopRequested = $false
$cancelHandler = [ConsoleCancelEventHandler]{
    param($sender, $eventArgs)
    $eventArgs.Cancel = $true
    $stopRequested = $true
    $state.phase = "stopping"
    Write-AtomicJson -Path $statePath -Value $state
    if ($null -ne $child -and -not $child.HasExited) {
        try {
            $child.StandardInput.WriteLine("SL_BOTS_CTRL_C")
            $child.StandardInput.Flush()
        }
        catch {
            # The child may have already closed stdin while exiting.
        }
    }
}
[Console]::add_CancelKeyPress($cancelHandler)
$exitCode = 1
try {
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $python
    $startInfo.WorkingDirectory = $repositoryRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardInput = $true
    $startInfo.ArgumentList.Add("-m")
    $startInfo.ArgumentList.Add("sl_bots.cli")
    $startInfo.ArgumentList.Add("train")
    $startInfo.ArgumentList.Add("test-pipeline")
    $startInfo.ArgumentList.Add("--config")
    $startInfo.ArgumentList.Add($configPath)
    $startInfo.ArgumentList.Add("--demo-root")
    $startInfo.ArgumentList.Add($demoPath)
    $startInfo.ArgumentList.Add("--data-root")
    $startInfo.ArgumentList.Add($dataPath)
    $startInfo.ArgumentList.Add("--server-root")
    $startInfo.ArgumentList.Add($serverPath)
    $startInfo.Environment["SL_BOTS_CONTROL_STDIN"] = "1"
    $child = [Diagnostics.Process]::new()
    $child.StartInfo = $startInfo
    $null = $child.Start()
    $state.phase = "running"
    $state.recorded_pids = @([int]$child.Id)
    Write-AtomicJson -Path $statePath -Value $state
    $child.WaitForExit()
    $child.Refresh()
    $exitCode = $child.ExitCode
    if ($stopRequested) {
        $state.phase = "interrupted"
    }
    else {
        $state.phase = if ($exitCode -eq 0) { "completed" } else { "failed" }
    }
}
catch {
    $stopRequested = $true
    $state.phase = "interrupted"
    Write-AtomicJson -Path $statePath -Value $state
    if ($null -ne $child -and -not $child.HasExited) {
        try {
            $child.StandardInput.WriteLine("SL_BOTS_CTRL_C")
            $child.StandardInput.Flush()
        }
        catch {
        }
        if (-not $child.WaitForExit(30000)) {
            $state.phase = "timeout"
            Write-AtomicJson -Path $statePath -Value $state
            Stop-TrackedProcessTree -RootPid ([int]$child.Id)
        }
    }
    throw
}
finally {
    $state.recorded_pids = @()
    Write-AtomicJson -Path $statePath -Value $state
    [Console]::remove_CancelKeyPress($cancelHandler)
    if ($null -ne $child) {
        $child.Dispose()
    }
}

exit $exitCode
