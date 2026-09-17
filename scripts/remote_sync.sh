#!/usr/bin/env bash
# 在 GPU 服务器上执行一次仓库同步。
# 由 scripts/sync_run_r1_to_server.ps1 通过 scp 送达后调用, 不要本地直接执行。
#
# 用法: bash remote_sync.sh <repo> <mode> [source] [backup_root]
#   mode        = network       从 origin 拉取
#   mode        = bundle        从 <source> 指定的 git bundle 拉取
#   backup_root 备份目录的父目录, 默认 /root (本机自测时可指向临时目录)
#
# 退出码: 0 成功; 2 其他失败; 3 network 模式拉取失败 (调用方可改用 bundle)
#
# 安全性: 合并会覆盖的未跟踪文件一律先移动到时间戳备份目录, 不删除。
# 备份路径写入 /root/.hyperdaf_last_untracked_backup, 便于事后比对。
set -uo pipefail

REPO="${1:?需要仓库路径}"
MODE="${2:?需要模式 network|bundle}"
SOURCE="${3:-}"
BACKUP_ROOT="${4:-/root}"

cd "$REPO" || { echo "REPO_NOT_FOUND: $REPO"; exit 2; }

echo "--- 同步前 HEAD ---"
git log --oneline -1

echo "--- 同步前工作区状态 (最多 20 行) ---"
git status --porcelain | head -20

echo "--- 拉取 ---"
if [ "$MODE" = "network" ]; then
    if ! git fetch origin main; then
        echo "REMOTE_FETCH_FAILED"
        exit 3
    fi
else
    if [ -z "$SOURCE" ] || [ ! -f "$SOURCE" ]; then
        echo "BUNDLE_MISSING: $SOURCE"
        exit 2
    fi
    if ! git fetch "$SOURCE" main:refs/remotes/origin/main; then
        echo "REMOTE_BUNDLE_FETCH_FAILED"
        exit 2
    fi
fi

INCOMING=$(mktemp)
OVERLAP=$(mktemp)
git ls-tree -r --name-only FETCH_HEAD | sort > "$INCOMING"

# 找出「合并要写入、但工作区已存在且未被跟踪」的路径。这些正是 merge 会拒绝的情形。
while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    if [ -e "$rel" ] && ! git ls-files --error-unmatch -- "$rel" >/dev/null 2>&1; then
        printf '%s\n' "$rel" >> "$OVERLAP"
    fi
done < "$INCOMING"

OVERLAP_COUNT=$(wc -l < "$OVERLAP" | tr -d ' ')
echo "--- 合并会覆盖的未跟踪文件数: $OVERLAP_COUNT ---"

BACKUP=""
if [ "$OVERLAP_COUNT" -gt 0 ]; then
    BACKUP="$BACKUP_ROOT/HyperDAF-MSA-untracked-backup-$(date +%Y%m%d-%H%M%S)"
    while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        mkdir -p "$BACKUP/$(dirname "$rel")"
        mv "$rel" "$BACKUP/$rel"
    done < "$OVERLAP"
    printf '%s\n' "$BACKUP" > "$BACKUP_ROOT/.hyperdaf_last_untracked_backup"
    echo "--- 已备份到 $BACKUP ---"
    head -20 "$OVERLAP" | sed 's/^/    /'
    [ "$OVERLAP_COUNT" -gt 20 ] && echo "    ...(共 $OVERLAP_COUNT 条, 完整清单见备份目录)"

    # merge 可能因空目录残留而失败, 清掉这些空目录壳
    while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        dir=$(dirname "$rel")
        if [ -d "$dir" ] && [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
            rmdir -p "$dir" 2>/dev/null || true
        fi
    done < "$OVERLAP"
fi

git checkout -- .
if ! git merge --ff-only FETCH_HEAD; then
    echo "REMOTE_MERGE_FAILED"
    [ -n "$BACKUP" ] && echo "备份仍保留在 $BACKUP"
    exit 2
fi

echo "--- 同步后 HEAD ---"
git log --oneline -1

if [ -n "$BACKUP" ]; then
    echo "--- 备份内容与合并后版本的差异 ---"
    DIFF_COUNT=0
    while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        if [ -f "$rel" ] && ! cmp -s "$BACKUP/$rel" "$rel"; then
            echo "    有差异: $rel"
            DIFF_COUNT=$((DIFF_COUNT + 1))
        fi
    done < "$OVERLAP"
    echo "--- 有差异的文件数: $DIFF_COUNT / $OVERLAP_COUNT ---"
    echo "--- 备份目录（未删除, 请确认后再手工清理）: $BACKUP ---"
fi

rm -f "$INCOMING" "$OVERLAP"
echo "REMOTE_SYNC_OK"