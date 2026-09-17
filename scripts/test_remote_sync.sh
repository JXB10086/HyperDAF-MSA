#!/usr/bin/env bash
# remote_sync.sh 的本地回归测试 (在临时 git 仓库里跑, 不接触任何真实服务器)。
#
# 覆盖两个场景:
#   1. 服务器存在「合并要写入但未被跟踪」的同名文件 -> 必须先备份再合并;
#   2. 不存在冲突 -> 直接 fast-forward, 不产生备份目录。
#
# 用法: bash scripts/test_remote_sync.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SYNC="$SCRIPT_DIR/remote_sync.sh"
[ -f "$SYNC" ] || { echo "找不到 $SYNC"; exit 1; }

export GIT_AUTHOR_NAME=test GIT_AUTHOR_EMAIL=test@example.invalid
export GIT_COMMITTER_NAME=test GIT_COMMITTER_EMAIL=test@example.invalid
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null

FAIL=0
check() {
    if [ "$2" = "$3" ]; then
        echo "  [OK]   $1"
    else
        echo "  [FAIL] $1 (期望 '$3', 实得 '$2')"
        FAIL=1
    fi
}

build_scenario() {
    local root=$1
    mkdir -p "$root/source"
    cd "$root/source"
    git init -q -b main .
    printf 'A\n' > A.txt
    git add -A
    git commit -qm 'c1'

    git clone -q "$root/source" "$root/server"
    cd "$root/source"
}

echo "=== 场景 1: 未跟踪同名文件必须先备份 ==="

ROOT=$(mktemp -d)
build_scenario "$ROOT"

mkdir -p "$ROOT/server/experiments/cmrp_evidence"
printf 'SERVER_COPY\n' > "$ROOT/server/experiments/cmrp_evidence/x.json"

cd "$ROOT/source"
mkdir -p experiments/cmrp_evidence
printf 'COMMITTED\n' > experiments/cmrp_evidence/x.json
git add -A
git commit -qm 'c2'

git bundle create "$ROOT/main.bundle" main >/dev/null 2>&1
SOURCE_HEAD=$(git rev-parse HEAD)

OUT=$(bash "$SYNC" "$ROOT/server" bundle "$ROOT/main.bundle" "$ROOT" 2>&1)
SYNC_RC=$?

check "脚本退出码为 0" "$SYNC_RC" "0"

SERVER_HEAD=$(cd "$ROOT/server" && git rev-parse HEAD)
check "服务器已 fast-forward 到源 HEAD" "$SERVER_HEAD" "$SOURCE_HEAD"

BACKUP=$(cat "$ROOT/.hyperdaf_last_untracked_backup" 2>/dev/null || echo MISSING)
check "记录了备份目录" "$([ -d "$BACKUP" ] && echo yes || echo no)" "yes"
check "旧副本已备份" "$(cat "$BACKUP/experiments/cmrp_evidence/x.json" 2>/dev/null || echo MISSING)" "SERVER_COPY"
check "工作区已是合并后版本" "$(cat "$ROOT/server/experiments/cmrp_evidence/x.json")" "COMMITTED"
check "报告了 1 个差异文件" \
    "$(printf '%s' "$OUT" | grep -c '有差异的文件数: 1 / 1')" "1"
check "未跟踪的 A.txt 未被破坏" "$(cat "$ROOT/server/A.txt")" "A"
rm -rf "$ROOT"

echo "=== 场景 2: 无冲突时直接合并, 不产生备份 ==="

ROOT=$(mktemp -d)
build_scenario "$ROOT"
cd "$ROOT/source"
mkdir -p experiments/new_track
printf 'NEW\n' > experiments/new_track/y.json
git add -A
git commit -qm 'c2'
git bundle create "$ROOT/main.bundle" main >/dev/null 2>&1
SOURCE_HEAD=$(git rev-parse HEAD)

OUT=$(bash "$SYNC" "$ROOT/server" bundle "$ROOT/main.bundle" "$ROOT" 2>&1)
SYNC_RC=$?

check "脚本退出码为 0" "$SYNC_RC" "0"
check "服务器 HEAD 已更新" "$(cd "$ROOT/server" && git rev-parse HEAD)" "$SOURCE_HEAD"
check "未创建备份目录" "$(ls -d "$ROOT"/HyperDAF-MSA-untracked-backup-* 2>/dev/null | wc -l | tr -d ' ')" "0"
check "报告 0 个覆盖冲突" "$(printf '%s' "$OUT" | grep -c '合并会覆盖的未跟踪文件数: 0')" "1"
rm -rf "$ROOT"

echo
if [ "$FAIL" -eq 0 ]; then
    echo "REMOTE_SYNC_TEST_PASS"
else
    echo "REMOTE_SYNC_TEST_FAIL"
    exit 1
fi