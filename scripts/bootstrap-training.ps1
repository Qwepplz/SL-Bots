[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$PythonPath,
    [switch]$AllowCpuOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))

$os = Get-CimInstance Win32_OperatingSystem
if ($os.Caption -notmatch "Windows 11") {
    throw "需要 Windows 11；检测到 $($os.Caption)。"
}

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "未找到指定 Python：$PythonPath"
}
$python = (Resolve-Path -LiteralPath $PythonPath).Path

$pythonVersion = (& $python --version 2>&1) -join " "
if ($pythonVersion -notmatch "Python 3\.12\.") {
    throw "需要 Python 3.12；检测到 $pythonVersion。"
}

$gpu = Get-CimInstance Win32_VideoController |
    Where-Object { $_.Name -match "Radeon RX 9070 XT" }
if ($null -eq $gpu -and -not $AllowCpuOnly) {
    throw "未检测到 AMD Radeon RX 9070 XT；如仅执行 CPU 校验，请传入 -AllowCpuOnly。"
}

$probeDevice = if ($AllowCpuOnly) { "cpu" } else { "cuda" }
$torchProbe = @"
import importlib.util
import sys

torch_spec = importlib.util.find_spec("torch")
if torch_spec is None:
    print("torch=missing")
    raise SystemExit(2)

import torch
sys.path.insert(0, r"$repoRoot\src")
from sl_bots.training_device import run_training_device_smoke

print(f"torch={torch.__version__}")
print(f"hip={getattr(torch.version, 'hip', None)}")
print(f"cuda_available={torch.cuda.is_available()}")
report = run_training_device_smoke("$probeDevice")
print(
    f"training_device_smoke=passed device={report.device_type} "
    f"gradient_norm={report.gradient_norm:.6g} "
    f"parameter_updated={report.parameter_updated}"
)
"@

$torchOutput = & $python -c $torchProbe 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch 未安装或无法导入：$($torchOutput -join ' ')"
}

if (-not ($torchOutput -match "torch=2\.9\.")) {
    throw "需要 PyTorch 2.9.x；检测结果：$($torchOutput -join '; ')"
}

if (-not $AllowCpuOnly -and -not ($torchOutput -match "torch=2\.9\.[^\r\n]*\+rocm7\.2\.1")) {
    throw "需要 PyTorch 2.9.x+rocm7.2.1；检测结果：$($torchOutput -join '; ')"
}

if (-not $AllowCpuOnly -and -not ($torchOutput -match "hip=7\.2\.")) {
    throw "需要 ROCm 7.2 运行时；检测结果：$($torchOutput -join '; ')"
}

if (-not $AllowCpuOnly -and -not ($torchOutput -match "cuda_available=True")) {
    throw "正式训练需要可用 HIP 设备；检测结果：$($torchOutput -join '; ')"
}

Write-Output "环境检查通过；脚本未修改驱动、Python 或系统配置。"
Write-Output ($torchOutput -join [Environment]::NewLine)
