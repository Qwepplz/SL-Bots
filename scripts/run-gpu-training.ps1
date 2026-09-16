param(
    [Parameter(Mandatory = $true)]
    [string]$Config,
    [Parameter(Mandatory = $true)]
    [string]$DemoRoot,
    [Parameter(Mandatory = $true)]
    [string]$Extractor,
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [Parameter(Mandatory = $true)]
    [string]$ServerRoot,
    [ValidateRange(1, 4)]
    [int]$MaxServers = 4,
    [int]$DurationSeconds = 7200,
    [ValidateSet("production")]
    [string]$Purpose = "production",
    [string]$RunId = "gpu-training",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python runtime not found: $python"
}

function ConvertTo-WindowsCommandLineArgument {
    param([AllowNull()][object]$Value)

    $text = if ($null -eq $Value) { "" } else { [string]$Value }
    if ($text.Length -eq 0) {
        return '""'
    }
    if ($text -notmatch '[\s"]') {
        return $text
    }

    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($character in $text.ToCharArray()) {
        if ($character -eq '\') {
            $backslashes++
            continue
        }
        if ($character -eq '"') {
            if ($backslashes -gt 0) {
                [void]$builder.Append((('\' * ($backslashes * 2 + 1)) -join ''))
            }
            else {
                [void]$builder.Append('\')
            }
            [void]$builder.Append('"')
            $backslashes = 0
            continue
        }
        if ($backslashes -gt 0) {
            [void]$builder.Append((('\' * $backslashes) -join ''))
            $backslashes = 0
        }
        [void]$builder.Append($character)
    }
    if ($backslashes -gt 0) {
        [void]$builder.Append((('\' * ($backslashes * 2)) -join ''))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

$arguments = @(
    "-m", "sl_bots.cli", "train", "pipeline",
    "--config", [IO.Path]::GetFullPath($Config),
    "--demo-root", [IO.Path]::GetFullPath($DemoRoot),
    "--extractor", [IO.Path]::GetFullPath($Extractor),
    "--data-root", [IO.Path]::GetFullPath($DataRoot),
    "--server-root", [IO.Path]::GetFullPath($ServerRoot),
    "--duration-seconds", $DurationSeconds,
    "--max-servers", $MaxServers,
    "--purpose", $Purpose,
    "--run-id", $RunId
)
if ($Resume) {
    $arguments += "--resume"
}
$argumentString = [string]::Join(
    ' ',
    @($arguments | ForEach-Object { ConvertTo-WindowsCommandLineArgument $_ })
)

$child = $null
try {
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $python
    $startInfo.Arguments = $argumentString
    $startInfo.WorkingDirectory = $repoRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardInput = $true
    $startInfo.EnvironmentVariables["SL_BOTS_CONTROL_STDIN"] = "1"
    $child = [Diagnostics.Process]::new()
    $child.StartInfo = $startInfo
    if (-not $child.Start()) {
        throw "failed to start Python training process"
    }
    while (-not $child.WaitForExit(250)) {
        # Keep the PowerShell pipeline interruptible while Python owns cleanup.
    }
    exit $child.ExitCode
}
catch [System.Management.Automation.PipelineStoppedException] {
    if ($child -and -not $child.HasExited) {
        try {
            $child.StandardInput.WriteLine("SL_BOTS_CTRL_C")
            $child.StandardInput.Flush()
            $child.StandardInput.Close()
        }
        catch {
            # The child may have exited between the interrupt and this write.
        }
        if (-not $child.WaitForExit(15000) -and -not $child.HasExited) {
            $taskkill = Join-Path $env:SystemRoot "System32\taskkill.exe"
            if (-not (Test-Path -LiteralPath $taskkill -PathType Leaf)) {
                throw "无法安全终止未退出的训练子进程：未找到 $taskkill"
            }
            $killInfo = [Diagnostics.ProcessStartInfo]::new()
            $killInfo.FileName = $taskkill
            $killInfo.Arguments = "/PID $($child.Id) /T /F"
            $killInfo.UseShellExecute = $false
            $killInfo.CreateNoWindow = $true
            $killer = [Diagnostics.Process]::Start($killInfo)
            try {
                $killer.WaitForExit(5000) | Out-Null
            }
            finally {
                $killer.Dispose()
            }
            if (-not $child.HasExited) {
                throw "训练子进程未能在 Ctrl+C 后退出"
            }
        }
    }
    exit 130
}
finally {
    if ($child) {
        $child.Dispose()
    }
}
