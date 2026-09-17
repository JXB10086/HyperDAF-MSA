<#
.SYNOPSIS
  清理 HyperDAF-MSA 仓库中的可再生垃圾文件。

.DESCRIPTION
  默认只做 DRY-RUN（仅列出将删除的内容，不实际删除）。
  加 -Apply 才真正删除。

  只处理明确可再生的内容：
    - __pycache__/ 目录
    - *.pyc / *.pyo
  可选（-IncludeEmptyLogs）：
    - 体积为 0 的 *.err 文件

  所有待删路径都会先校验位于仓库根目录之内，任何越界路径都会导致整体中止。

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\cleanup_junk.ps1
  powershell -ExecutionPolicy Bypass -File scripts\cleanup_junk.ps1 -Apply
#>
[CmdletBinding()]
param(
    [switch]$Apply,
    [switch]$IncludeEmptyLogs
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Write-Host "仓库根目录: $root"
Write-Host ("模式: {0}" -f $(if ($Apply) { "APPLY（将实际删除）" } else { "DRY-RUN（仅预览）" }))

# 不触碰虚拟环境与 git 内部
$exclude = @("$root\.venv", "$root\.git")

function Test-InsideRoot {
    param([string]$Path)
    foreach ($ex in $exclude) {
        if ($Path.StartsWith($ex, [StringComparison]::OrdinalIgnoreCase)) { return $false }
    }
    return $Path.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)
}

$dirs = @(Get-ChildItem -LiteralPath $root -Recurse -Force -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
          Where-Object { Test-InsideRoot $_.FullName })
$files = @(Get-ChildItem -LiteralPath $root -Recurse -Force -File -ErrorAction SilentlyContinue |
           Where-Object { $_.Extension -in ".pyc", ".pyo" } |
           Where-Object { Test-InsideRoot $_.FullName })

$emptyLogs = @()
if ($IncludeEmptyLogs) {
    $emptyLogs = @(Get-ChildItem -LiteralPath $root -Recurse -Force -File -Filter "*.err" -ErrorAction SilentlyContinue |
                   Where-Object { $_.Length -eq 0 } |
                   Where-Object { Test-InsideRoot $_.FullName })
}

$dirBytes = ($dirs   | ForEach-Object { (Get-ChildItem -LiteralPath $_.FullName -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object -Sum Length).Sum } | Measure-Object -Sum).Sum
$fileBytes = ($files  | Measure-Object -Sum Length).Sum

Write-Host ""
Write-Host ("__pycache__ 目录 : {0}" -f $dirs.Count)
Write-Host ("*.pyc / *.pyo    : {0}" -f $files.Count)
if ($IncludeEmptyLogs) { Write-Host ("空 *.err 文件    : {0}" -f $emptyLogs.Count) }
Write-Host ("预计回收         : {0:N2} MB" -f ((($dirBytes + $fileBytes) / 1MB)))

if (-not ($dirs.Count -or $files.Count -or $emptyLogs.Count)) {
    Write-Host "没有需要清理的内容。"
    return
}

if (-not $Apply) {
    Write-Host ""
    Write-Host "预览（前 20 项）："
    ($dirs.FullName + $files.FullName + $emptyLogs.FullName) | Select-Object -First 20 | ForEach-Object { Write-Host "  $_" }
    Write-Host ""
    Write-Host "未做任何修改。确认无误后加 -Apply 重新运行。"
    return
}

foreach ($d in $dirs)  { Remove-Item -LiteralPath $d.FullName -Recurse -Force }
foreach ($f in $files) { Remove-Item -LiteralPath $f.FullName -Force }
foreach ($f in $emptyLogs) { Remove-Item -LiteralPath $f.FullName -Force }

Write-Host ""
Write-Host "清理完成。"
Write-Host ("剩余 __pycache__ 目录: {0}" -f (@(Get-ChildItem -LiteralPath $root -Recurse -Force -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Where-Object { Test-InsideRoot $_.FullName }).Count))