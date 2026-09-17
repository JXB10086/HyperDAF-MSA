<#
.SYNOPSIS
    提交 run_r1.py 存盘改动, 同步到 GPU 服务器, 并做一次端到端写入校验。

.DESCRIPTION
    步骤:
      1. 本地跑 CMRP v2 单元测试 (改动必须通过)。
      2. git add 指定文件 -> commit -> push origin main。
      3. 服务器侧: 确认只有 CRLF 伪改动后才丢弃工作区, 然后 fast-forward 拉取。
         拉取失败(例如私有仓库凭据缺失)时自动退回第 4 步的直传通道。
      4. 对比本地与远程 run_r1.py 的 SHA-256; 不一致时 scp 整棵 experiments/cmrp_v2 再校验。
      5. 服务器侧 py_compile + 关键符号计数比对, 打印 PASS/FAIL。

.PARAMETER Message
    git commit 的提交信息。

.PARAMETER SkipPush
    只提交不推送。

.PARAMETER SkipSync
    只做本地测试与提交, 不碰服务器。

.EXAMPLE
    .\scripts\sync_run_r1_to_server.ps1
#>
param(
    [string]$Message = "run_r1 保存验证集最优权重, 便于后续逐样本表征审计",
    [switch]$SkipPush,
    [switch]$SkipSync
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
$SshHost = 'root@connect.nmb1.seetacloud.com'
$SshPort = 38107
$SshKey = Join-Path $env:USERPROFILE '.ssh\id_ed25519'
$RemoteRepo = '/root/HyperDAF-MSA'
$Target = 'experiments/cmrp_v2/r1_relational/run_r1.py'
$V2Dir = 'experiments/cmrp_v2'

$Tracked = @(
    $Target,
    'experiments/cmrp_v2/tests/test_r1_checkpoint.py',
    'experiments/cmrp_v2/tests/test_r1_checkpoint_integration.py',
    'experiments/cmrp_v2/README.md',
    'experiments/cmrp_v2/R1_PREREGISTRATION.md',
    'experiments/cmrp_v2/PROJECT_STATE.json',
    'scripts/sync_run_r1_to_server.ps1'
)
$Markers = @('def save_best_checkpoint', 'torch.save', 'checkpoint_dir')

function Write-Step([string]$Text) { Write-Host "`n=== $Text ===" -ForegroundColor Cyan }
function Write-Ok([string]$Text) { Write-Host "  [OK] $Text" -ForegroundColor Green }
function Write-Bad([string]$Text) { Write-Host "  [FAIL] $Text" -ForegroundColor Red }
function Write-Warn([string]$Text) { Write-Host "  [WARN] $Text" -ForegroundColor Yellow }

function Invoke-Remote([string]$Command) {
    $lines = & ssh -o BatchMode=yes -o ConnectTimeout=15 -p $SshPort -i $SshKey $SshHost $Command
    if ($LASTEXITCODE -ne 0) { throw "远程命令失败 (exit $LASTEXITCODE): $Command" }
    return $lines
}

function Get-RemoteHash([string]$RelativePath) {
    $out = (Invoke-Remote "sha256sum $RemoteRepo/$RelativePath") -join ''
    return ($out.Trim() -split '\s+')[0].ToLower()
}

Push-Location $RepoRoot
try {
    Write-Step '1/5 前置检查与本地单元测试'
    foreach ($file in $Tracked) {
        if (-not (Test-Path $file)) { throw "缺少待提交文件: $file" }
    }
    & .\.venv\Scripts\python.exe -m unittest discover -s experiments/cmrp_v2/tests -p 'test_*.py'
    if ($LASTEXITCODE -ne 0) { throw '本地测试未通过, 已中止同步。' }
    Write-Ok 'CMRP v2 测试全部通过'

    Write-Step '2/5 提交本地改动'
    & git add @Tracked
    if ($LASTEXITCODE -ne 0) { throw 'git add 失败' }
    & git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Ok '暂存区无变化, 跳过 commit'
    } else {
        & git commit -m $Message
        if ($LASTEXITCODE -ne 0) { throw 'git commit 失败' }
        Write-Ok "已提交: $Message"
    }
    if (-not $SkipPush) {
        & git push origin main
        if ($LASTEXITCODE -ne 0) { throw 'git push 失败 (检查远程凭据)' }
        Write-Ok '已推送到 origin/main'
    }

    if ($SkipSync) { Write-Ok '指定了 -SkipSync, 结束。'; return }

    Write-Step '3/5 服务器侧拉取'
    Invoke-Remote "test -d $RemoteRepo/.git && echo REPO_OK" | Out-Null
    $dirty = (Invoke-Remote "cd $RemoteRepo && git diff --stat --ignore-cr-at-eol") -join "`n"
    if ($dirty.Trim()) { throw "服务器工作区存在真实改动, 拒绝覆盖:`n$dirty" }
    $gitSynced = $false
    try {
        Invoke-Remote "cd $RemoteRepo && git checkout -- . && git fetch origin && git pull --ff-only origin main" | Out-Host
        $gitSynced = $true
        Write-Ok '服务器已 fast-forward 到 origin/main'
    } catch {
        Write-Warn "服务器 git 拉取失败, 退回 scp 直传: $($_.Exception.Message)"
    }

    Write-Step '4/5 校验文件一致性'
    $localHash = (Get-FileHash $Target -Algorithm SHA256).Hash.ToLower()
    $remoteHash = Get-RemoteHash $Target
    if ($localHash -ne $remoteHash) {
        Write-Bad "哈希不一致 (local=$localHash remote=$remoteHash), 改用 scp 直传整棵 $V2Dir"
        Invoke-Remote "mkdir -p $RemoteRepo/experiments" | Out-Null
        & scp -r -P $SshPort -i $SshKey $V2Dir "${SshHost}:$RemoteRepo/experiments/"
        if ($LASTEXITCODE -ne 0) { throw "scp 直传失败: $V2Dir" }
        $remoteHash = Get-RemoteHash $Target
        if ($localHash -ne $remoteHash) { throw "scp 后哈希仍不一致: local=$localHash remote=$remoteHash" }
        Write-Ok 'scp 直传后哈希一致'
        if (-not $gitSynced) { Write-Warn '本次未经 git 同步, 服务器工作区会显示这些文件为未提交改动' }
    }
    Write-Ok "run_r1.py SHA-256 一致: $localHash"

    Write-Step '5/5 服务器侧语法与存盘路径校验'
    Invoke-Remote "cd $RemoteRepo && python -m py_compile $Target && echo PY_COMPILE_OK" | Out-Host

    $localText = Get-Content $Target -Raw
    $patterns = ($Markers | ForEach-Object { "'$($_)'" }) -join ' -e '
    $remoteCounts = (Invoke-Remote "cd $RemoteRepo && grep -o -e $patterns $Target | sort | uniq -c") |
        ForEach-Object { $_.Trim() } | Where-Object { $_ }
    if ($remoteCounts.Count -ne $Markers.Count) {
        throw "远程标记数量不符: 期望 $($Markers.Count) 项, 实得 $($remoteCounts.Count)"
    }
    foreach ($line in $remoteCounts) {
        $parts = $line -split '\s+', 2
        $count = [int]$parts[0]
        $name = $parts[1]
        $expected = ([regex]::Matches($localText, [regex]::Escape($name))).Count
        if ($count -ne $expected) { throw "服务器上 '$name' 出现 $count 次, 本地为 $expected 次" }
        Write-Ok "'$name' 出现 $count 次, 与本地一致"
    }

    Write-Host "`nSYNC_PASS" -ForegroundColor Green
}
finally {
    Pop-Location
}