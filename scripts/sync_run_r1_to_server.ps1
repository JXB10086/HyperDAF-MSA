<#
.SYNOPSIS
    提交 run_r1.py 存盘改动, 同步到 GPU 服务器, 并做一次端到端写入校验。

.DESCRIPTION
    服务器无法直连 GitHub (GnuTLS recv error -110), 因此默认走 git bundle 通道:
    本地打包 main -> scp 上服务器 -> 服务器从本地 bundle fetch 并 fast-forward。
    这既绕开服务器出网限制, 又让服务器的 git 历史与 origin/main 保持一致。

    步骤:
      0. 打印服务器诊断信息 (GPU / python / git 状态)。
      1. 本地跑 CMRP v2 单元测试 (改动必须通过)。
      2. git add 指定文件 -> commit -> push origin main。
      3. 服务器侧: 确认只有 CRLF 伪改动后才丢弃工作区, 然后 bundle 同步。
      4. 对比本地与远程 run_r1.py 的 SHA-256; 不一致时 scp 直传整棵 experiments/cmrp_v2。
      5. 服务器侧 py_compile + 关键符号计数比对, 打印 PASS/FAIL。

.PARAMETER Message
    git commit 的提交信息。

.PARAMETER SkipPush
    只提交不推送。

.PARAMETER SkipSync
    只做本地测试与提交, 不碰服务器。

.PARAMETER DiagnoseOnly
    只打印服务器诊断信息, 不做任何改动。

.EXAMPLE
    .\scripts\sync_run_r1_to_server.ps1
.EXAMPLE
    .\scripts\sync_run_r1_to_server.ps1 -DiagnoseOnly
#>
param(
    [string]$Message = "run_r1 保存验证集最优权重, 便于后续逐样本表征审计",
    [switch]$SkipPush,
    [switch]$SkipSync,
    [switch]$DiagnoseOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
$SshHost = 'root@connect.nmb1.seetacloud.com'
$SshPort = 38107
$SshKey = Join-Path $env:USERPROFILE '.ssh\id_ed25519'
$RemoteRepo = '/root/HyperDAF-MSA'
$RemoteBundle = '/root/hyperdaf-main.bundle'
$Target = 'experiments/cmrp_v2/r1_relational/run_r1.py'
$V2Dir = 'experiments/cmrp_v2'

$Tracked = @(
    $Target,
    'experiments/cmrp_v2/tests/test_r1_checkpoint.py',
    'experiments/cmrp_v2/tests/test_r1_checkpoint_integration.py',
    'experiments/cmrp_v2/README.md',
    'experiments/cmrp_v2/R1_PREREGISTRATION.md',
    'experiments/cmrp_v2/PROJECT_STATE.json',
    'docs/SERVER_STATE.md',
    'scripts/sync_run_r1_to_server.ps1',
    'scripts/finalize_reorg.ps1'
)
$Markers = @('def save_best_checkpoint', 'torch.save', 'checkpoint_dir')

$SshArgs = @(
    '-o', 'BatchMode=yes',
    '-o', 'ConnectTimeout=15',
    '-o', 'StrictHostKeyChecking=accept-new',
    '-p', $SshPort,
    '-i', $SshKey,
    $SshHost
)

function Write-Step([string]$Text) { Write-Host "`n=== $Text ===" -ForegroundColor Cyan }
function Write-Ok([string]$Text) { Write-Host "  [OK] $Text" -ForegroundColor Green }
function Write-Bad([string]$Text) { Write-Host "  [FAIL] $Text" -ForegroundColor Red }
function Write-Warn([string]$Text) { Write-Host "  [WARN] $Text" -ForegroundColor Yellow }

function Invoke-RemoteRaw([string]$Command) {
    $lines = & ssh @SshArgs $Command 2>&1
    return [pscustomobject]@{
        ExitCode = $LASTEXITCODE
        Output   = (($lines | ForEach-Object { "$_" }) -join "`n").Trim()
    }
}

function Invoke-Remote([string]$Command) {
    $result = Invoke-RemoteRaw $Command
    if ($result.ExitCode -ne 0) {
        throw "远程命令失败 (exit $($result.ExitCode)): $Command`n$($result.Output)"
    }
    return ($result.Output -split "`n")
}

function Get-RemoteHash([string]$RelativePath) {
    $probe = "if [ -f $RemoteRepo/$RelativePath ]; then sha256sum $RemoteRepo/$RelativePath | cut -d' ' -f1; else echo MISSING; fi"
    $result = Invoke-RemoteRaw $probe
    if ($result.ExitCode -ne 0) { throw "无法读取远程哈希: $($result.Output)" }
    return $result.Output.ToLower()
}

function Get-LocalHash([string]$RelativePath) {
    return (Get-FileHash $RelativePath -Algorithm SHA256).Hash.ToLower()
}

function Show-RemoteDiagnostics {
    Write-Step '0/5 服务器诊断'
    $probes = [ordered]@{
        '主机'      = 'hostname'
        'Python'    = 'python -V 2>&1 || echo NO_PYTHON'
        'GPU'       = 'nvidia-smi -L 2>/dev/null || echo NO_GPU'
        '磁盘'      = "df -h /root | tail -1"
        '仓库'      = "test -d $RemoteRepo/.git && echo GIT_REPO_PRESENT || echo GIT_REPO_MISSING"
        'HEAD'      = "cd $RemoteRepo && git log --oneline -1"
        '分支'      = "cd $RemoteRepo && git rev-parse --abbrev-ref HEAD"
        '本地改动'  = "cd $RemoteRepo && git status --porcelain | head -5"
        'cmrp_v2'   = "test -d $RemoteRepo/$V2Dir && echo CMRP_V2_PRESENT || echo CMRP_V2_MISSING"
        'GitHub'    = "cd $RemoteRepo && timeout 15 git ls-remote origin HEAD >/dev/null 2>&1 && echo GITHUB_REACHABLE || echo GITHUB_UNREACHABLE"
    }
    foreach ($name in $probes.Keys) {
        $result = Invoke-RemoteRaw $probes[$name]
        $value = ($result.Output -replace "`n", ' | ')
        if (-not $value) { $value = '(空)' }
        if ($result.ExitCode -ne 0) { $value = "$value (exit $($result.ExitCode))" }
        Write-Host ("  {0,-10} {1}" -f $name, $value)
    }
}

Push-Location $RepoRoot
try {
    Show-RemoteDiagnostics
    if ($DiagnoseOnly) { Write-Ok '仅诊断模式, 结束。'; return }

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

    Write-Step '3/5 服务器侧同步'
    Invoke-Remote "test -d $RemoteRepo/.git && echo GIT_OK" | Out-Null

    $dirty = (Invoke-Remote "cd $RemoteRepo && git diff --stat --ignore-cr-at-eol") -join "`n"
    if ($dirty.Trim()) { throw "服务器工作区存在真实改动, 拒绝覆盖:`n$dirty" }

    $gitSynced = $false
    $remoteHead = (Invoke-Remote "cd $RemoteRepo && git rev-parse HEAD") -join ''
    $localHead = (& git rev-parse HEAD) -join ''
    if ($remoteHead.Trim() -eq $localHead.Trim()) {
        Write-Ok '服务器 HEAD 已与本地一致, 无需同步'
        $gitSynced = $true
    } else {
        $reachable = Invoke-RemoteRaw "cd $RemoteRepo && timeout 20 git fetch origin main >/dev/null 2>&1 && echo REACHABLE || echo UNREACHABLE"
        if ($reachable.Output -match 'REACHABLE') {
            Invoke-Remote "cd $RemoteRepo && git checkout -- . && git merge --ff-only origin/main" | Out-Host
            $gitSynced = $true
            Write-Ok '服务器已通过网络 fast-forward 到 origin/main'
        } else {
            Write-Warn '服务器无法直连 GitHub, 改用 git bundle 通道'
            $bundleLocal = Join-Path $env:TEMP 'hyperdaf-main.bundle'
            if (Test-Path $bundleLocal) { Remove-Item -LiteralPath $bundleLocal -Force }
            & git bundle create $bundleLocal main
            if ($LASTEXITCODE -ne 0) { throw 'git bundle 创建失败' }
            $sizeMb = [math]::Round((Get-Item $bundleLocal).Length / 1MB, 2)
            Write-Ok "bundle 已生成 ($sizeMb MB)"
            & scp -P $SshPort -i $SshKey $bundleLocal "${SshHost}:$RemoteBundle"
            if ($LASTEXITCODE -ne 0) { throw 'scp bundle 失败' }
            Invoke-Remote "cd $RemoteRepo && git checkout -- . && git fetch $RemoteBundle main:refs/remotes/origin/main && git merge --ff-only FETCH_HEAD" | Out-Host
            $gitSynced = $true
            Write-Ok '服务器已通过 bundle fast-forward'
        }
    }

    Write-Step '4/5 校验文件一致性'
    $localHash = Get-LocalHash $Target
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