[CmdletBinding()]
param(
    [switch]$AllowCpuOnly
)

$ErrorActionPreference = "Stop"

function Get-CommandPath([string]$Name) {
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        return $null
    }
    return $command.Source
}

$os = Get-CimInstance Win32_OperatingSystem
if ($os.Caption -notmatch "Windows 11") {
    throw "需要 Windows 11；检测到 $($os.Caption)。"
}

$python = Get-CommandPath "python"
if ($null -eq $python) {
    throw "未找到 Python 3.12。"
}

$pythonVersion = (& $python --version 2>&1) -join " "
if ($pythonVersion -notmatch "Python 3\.12\.") {
    throw "需要 Python 3.12；检测到 $pythonVersion。"
}

$gpu = Get-CimInstance Win32_VideoController |
    Where-Object { $_.Name -match "Radeon RX 9070 XT" }
if ($null -eq $gpu -and -not $AllowCpuOnly) {
    throw "未检测到 AMD Radeon RX 9070 XT；如仅执行 CPU 校验，请传入 -AllowCpuOnly。"
}

$torchProbe = @'
import importlib.util
import sys

torch_spec = importlib.util.find_spec("torch")
if torch_spec is None:
    print("torch=missing")
    raise SystemExit(2)

import torch
print(f"torch={torch.__version__}")
print(f"hip={getattr(torch.version, 'hip', None)}")
print(f"cuda_available={torch.cuda.is_available()}")
'@

$torchOutput = & $python -c $torchProbe 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch 未安装或无法导入：$($torchOutput -join ' ')"
}

if (-not ($torchOutput -match "torch=2\.9\.")) {
    throw "需要 PyTorch 2.9.x；检测结果：$($torchOutput -join '; ')"
}

if (-not ($torchOutput -match "hip=7\.2\.1")) {
    throw "需要 ROCm 7.2.1；检测结果：$($torchOutput -join '; ')"
}

Write-Output "环境检查通过；脚本未修改驱动、Python 或系统配置。"
Write-Output ($torchOutput -join [Environment]::NewLine)
