[CmdletBinding()]
param(
    [string]$CompilerRoot = "C:\Users\30247\Downloads\sourcemod-1.12.0-git7227-windows",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$sourceDir = Join-Path $repoRoot "sourcemod\scripting"
$repoIncludeDir = Join-Path $sourceDir "include"
$compiler = Join-Path $CompilerRoot "addons\sourcemod\scripting\spcomp.exe"
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoRoot "build\sourcemod"
}

if (-not (Test-Path -LiteralPath $compiler)) {
    throw "未找到指定 SourcePawn 编译器：$compiler"
}
if (-not (Test-Path -LiteralPath $sourceDir)) {
    throw "SourcePawn 源码目录不存在：$sourceDir"
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$sources = Get-ChildItem -LiteralPath $sourceDir -Filter "*.sp" -File
if ($sources.Count -eq 0) {
    throw "未找到 SourcePawn 源文件：$sourceDir"
}

foreach ($source in $sources) {
    $output = Join-Path $OutputDir ($source.BaseName + ".smx")
    & $compiler -E -i (Join-Path $CompilerRoot "addons\sourcemod\scripting\include") -i $repoIncludeDir `
        -o $output $source.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "SourcePawn 编译失败：$($source.Name)"
    }
}
