[CmdletBinding()]
param(
    [string]$BuildDir = "",
    [string]$CMakePath = "",
    [string]$SourceModSdkPath = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$sourceDir = Join-Path $repoRoot "native\ipc"
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $repoRoot "build\native"
}

if (-not (Test-Path -LiteralPath (Join-Path $sourceDir "CMakeLists.txt"))) {
    throw "原生扩展源码尚未建立：$sourceDir"
}
if (-not [string]::IsNullOrWhiteSpace($SourceModSdkPath) -and
    -not (Test-Path -LiteralPath (Join-Path $SourceModSdkPath "public\smsdk_ext.cpp"))) {
    throw "SourceMod SDK 缺少 public\\smsdk_ext.cpp：$SourceModSdkPath"
}
$vsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vsInstallPath = ""
if (Test-Path -LiteralPath $vsWhere) {
    $vsInstallPath = (& $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.CMake.Project -property installationPath | Select-Object -First 1).Trim()
}

if ([string]::IsNullOrWhiteSpace($CMakePath)) {
    $cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
    if ($null -ne $cmakeCommand) {
        $CMakePath = $cmakeCommand.Source
    }
}
if ([string]::IsNullOrWhiteSpace($CMakePath)) {
    if (-not [string]::IsNullOrWhiteSpace($vsInstallPath)) {
        $bundledCMake = Join-Path $vsInstallPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        if (Test-Path -LiteralPath $bundledCMake) {
            $CMakePath = $bundledCMake
        }
    }
}
if ([string]::IsNullOrWhiteSpace($CMakePath) -or -not (Test-Path -LiteralPath $CMakePath)) {
    throw "未找到 CMake；请通过 -CMakePath 指定 cmake.exe。"
}

New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
$vcvarsAll = ""
if (-not [string]::IsNullOrWhiteSpace($vsInstallPath)) {
    $candidateVcvarsAll = Join-Path $vsInstallPath "VC\Auxiliary\Build\vcvarsall.bat"
    if (Test-Path -LiteralPath $candidateVcvarsAll) {
        $vcvarsAll = $candidateVcvarsAll
    }
}

if (-not [string]::IsNullOrWhiteSpace($vcvarsAll)) {
    $sourceModOptions = ""
    if (-not [string]::IsNullOrWhiteSpace($SourceModSdkPath)) {
        $sourceModOptions = ' -DSL_BOTS_ENABLE_SOURCEMOD=ON -DSL_BOTS_SOURCEMOD_SDK_DIR="' + $SourceModSdkPath + '"'
    }
    $configureCommand = '"' + $CMakePath + '" -S "' + $sourceDir + '" -B "' + $BuildDir + '" -A Win32' + $sourceModOptions
    $buildCommand = '"' + $CMakePath + '" --build "' + $BuildDir + '" --config Release --parallel'
    $commandLine = 'set "Path=" && call "' + $vcvarsAll + '" x64_x86 && ' + $configureCommand + ' && ' + $buildCommand
    & cmd.exe /d /s /c $commandLine
} else {
    $configureArguments = @("-S", $sourceDir, "-B", $BuildDir, "-A", "Win32")
    if (-not [string]::IsNullOrWhiteSpace($SourceModSdkPath)) {
        $configureArguments += @(
            "-DSL_BOTS_ENABLE_SOURCEMOD=ON",
            "-DSL_BOTS_SOURCEMOD_SDK_DIR=$SourceModSdkPath"
        )
    }
    & $CMakePath @configureArguments
    if ($LASTEXITCODE -ne 0) { throw "CMake 配置失败。" }
    & $CMakePath --build $BuildDir --config Release --parallel
}
if ($LASTEXITCODE -ne 0) { throw "原生扩展构建失败。" }
