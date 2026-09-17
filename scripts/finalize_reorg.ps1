<#
.SYNOPSIS
  HyperDAF-MSA 项目整理收尾：清理 + 提交 + 推送。默认 DRY-RUN。

.DESCRIPTION
  阶段 1（默认包含）— 可再生垃圾：
      __pycache__/、*.pyc、*.pyo、体积为 0 的 *.err、根目录 4.30
  阶段 2（-IncludeLarge）— 已确认冗余的大文件（约 2.66 GB）：
      HyperDAF-MSA_server_bundle_2026-09-15.tar.gz   约 2.1 GB
      external/                                      约 528 MB
      依据：data/ 与 third_party/ 本地完整；data/mmsa/MOSI/Processed/unaligned_50.pkl
            的 SHA-256 已核对等于 78e0f8b5...2e02524；external/ 下两份均为重复副本。
  阶段 3（-Commit 提交，-Push 推送）：
      git add -A，若无变更则跳过提交；推送 origin/main。

  所有待操作路径先校验位于仓库根目录内，越界立即中止。

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\finalize_reorg.ps1
  powershell -ExecutionPolicy Bypass -File scripts\finalize_reorg.ps1 -Apply -IncludeLarge -Commit -Push
#>
[CmdletBinding()]
param(
    [switch]$Apply,
    [switch]$IncludeLarge,
    [switch]$IncludeEmptyLogs = $true,
    [switch]$Commit,
    [switch]$Push
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Write-Host "仓库根目录 : $root"
Write-Host ("模式       : {0}" -f $(if ($Apply) { "APPLY" } else { "DRY-RUN" }))
Write-Host ""

$protected = @("$root\.venv", "$root\.git")

function Assert-InsideRoot {
    param([string]$Path, [string]$Label)
    foreach ($p in $protected) {
        if ($Path.StartsWith($p, [StringComparison]::OrdinalIgnoreCase)) {
            throw "拒绝操作受保护路径 ($Label): $Path"
        }
    }
    if (-not $Path.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
        throw "拒绝操作仓库外路径 ($Label): $Path"
    }
}

Write-Host "=== 阶段 1: 可再生垃圾 ==="
$pycache = @(Get-ChildItem -LiteralPath $root -Recurse -Force -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
             Where-Object { $_.FullName -notlike "$root\.venv*" -and $_.FullName -notlike "$root\.git*" })
$pyc = @(Get-ChildItem -LiteralPath $root -Recurse -Force -File -ErrorAction SilentlyContinue |
         Where-Object { ($_.Extension -eq ".pyc" -or $_.Extension -eq ".pyo") -and $_.FullName -notlike "$root\.venv*" })
$errs = @()
if ($IncludeEmptyLogs) {
    $errs = @(Get-ChildItem -LiteralPath $root -Recurse -Force -File -Filter "*.err" -ErrorAction SilentlyContinue |
              Where-Object { $_.Length -eq 0 -and $_.FullName -notlike "$root\.venv*" })
}
$scratch = @()
if (Test-Path -LiteralPath (Join-Path $root "4.30")) { $scratch += (Get-Item -LiteralPath (Join-Path $root "4.30")) }
foreach ($x in @($pycache) + @($pyc) + @($errs) + @($scratch)) { Assert-InsideRoot $x.FullName "phase1" }

$bytes1 = 0
if ($pycache.Count) { $bytes1 += ($pycache | ForEach-Object { (Get-ChildItem -LiteralPath $_.FullName -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object -Sum Length).Sum } | Measure-Object -Sum).Sum }
if ($pyc.Count) { $bytes1 += ($pyc | Measure-Object -Sum Length).Sum }
if ($errs.Count) { $bytes1 += ($errs | Measure-Object -Sum Length).Sum }
Write-Host ("  __pycache__ 目录 : {0}" -f $pycache.Count)
Write-Host ("  *.pyc / *.pyo    : {0}" -f $pyc.Count)
Write-Host ("  空 *.err 文件    : {0}" -f $errs.Count)
Write-Host ("  根目录 4.30      : {0}" -f $scratch.Count)
Write-Host ("  预计回收         : {0:N2} MB" -f ($bytes1 / 1MB))

Write-Host ""
Write-Host "=== 阶段 2: 已确认冗余的大文件 ==="
$large = @()
$bytes2 = 0
if ($IncludeLarge) {
    $bundle = @(Get-ChildItem -LiteralPath $root -Force -File -Filter "HyperDAF-MSA_server_bundle_*.tar.gz" -ErrorAction SilentlyContinue)
    $large += $bundle
    $extDir = Join-Path $root "external"
    if (Test-Path -LiteralPath $extDir) { $large += (Get-Item -LiteralPath $extDir) }
    foreach ($x in $large) { Assert-InsideRoot $x.FullName "phase2" }
    foreach ($x in $large) {
        if ($x.PSIsContainer) { $bytes2 += (Get-ChildItem -LiteralPath $x.FullName -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object -Sum Length).Sum }
        else { $bytes2 += $x.Length }
        Write-Host ("  - {0}" -f $x.FullName.Replace("$root\",""))
    }
    Write-Host ("  预计回收         : {0:N2} MB" -f ($bytes2 / 1MB))
} else {
    Write-Host "  已跳过（未指定 -IncludeLarge）：HyperDAF-MSA_server_bundle_*.tar.gz 、 external/"
}

Write-Host ""
if (-not $Apply) {
    Write-Host "DRY-RUN 结束，未做任何修改。确认后加 -Apply 重新运行。"
    return
}

Write-Host "=== 执行清理 ==="
foreach ($d in $pycache) { Remove-Item -LiteralPath $d.FullName -Recurse -Force }
foreach ($f in $pyc)     { Remove-Item -LiteralPath $f.FullName -Force }
foreach ($f in $errs)    { Remove-Item -LiteralPath $f.FullName -Force }
foreach ($f in $scratch) { Remove-Item -LiteralPath $f.FullName -Force }
Write-Host ("  阶段 1 完成，回收约 {0:N2} MB" -f ($bytes1 / 1MB))
if ($IncludeLarge) {
    foreach ($x in $large) {
        if ($x.PSIsContainer) { Remove-Item -LiteralPath $x.FullName -Recurse -Force } else { Remove-Item -LiteralPath $x.FullName -Force }
        Write-Host ("  已删除: {0}" -f $x.Name)
    }
    Write-Host ("  阶段 2 完成，回收约 {0:N2} MB" -f ($bytes2 / 1MB))
}

if (-not $Commit) {
    Write-Host ""
    Write-Host "未指定 -Commit，跳过 git 操作。"
    return
}

Write-Host ""
Write-Host "=== 阶段 3: git ==="
Push-Location $root
try {
    git add -A -- .
    if ($LASTEXITCODE -ne 0) { throw "git add 失败" }
    $staged = (git diff --cached --name-only | Measure-Object).Count
    Write-Host ("  已暂存变更文件数: {0}" -f $staged)
    if ($staged -gt 0) {
        git commit -m "整理仓库：清理可再生垃圾与重复副本，纳入项目文档"
        if ($LASTEXITCODE -ne 0) { throw "git commit 失败" }
    } else {
        Write-Host "  没有需要提交的变更。"
    }
    Write-Host ""
    git log --oneline -5

    if ($Push) {
        Write-Host ""
        Write-Host "=== 推送 origin/main ==="
        git push origin main
        if ($LASTEXITCODE -ne 0) { throw "git push 失败" }
        Write-Host "远端 main:"
        git ls-remote origin main
    } else {
        Write-Host ""
        Write-Host "未指定 -Push，提交仅停留在本地。"
    }
    Write-Host ""
    Write-Host "剩余 git status："
    git status --short
}
finally { Pop-Location }
Write-Host ""
Write-Host "全部完成。"