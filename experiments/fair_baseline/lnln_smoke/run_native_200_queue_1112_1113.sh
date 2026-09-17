#!/usr/bin/env bash
set -u

queue_root=/root/autodl-fs/HyperDAF-MSA-runs
queue_log="$queue_root/lnln_native_queue_1112_1113.log"
queue_status="$queue_root/lnln_native_queue_1112_1113.status"

printf 'started %s\n' "$(date -Is)" > "$queue_status"
for seed in 1112 1113; do
    printf '[%s] seed %s started\n' "$(date -Is)" "$seed" >> "$queue_log"
    bash "$queue_root/lnln_native_200_seed${seed}/run_native_200_seed.sh" "$seed" >> "$queue_log" 2>&1
    code=$?
    printf '[%s] seed %s finished exit=%s\n' "$(date -Is)" "$seed" "$code" >> "$queue_log"
    if [ "$code" -ne 0 ]; then
        printf 'failed seed=%s exit=%s time=%s\n' "$seed" "$code" "$(date -Is)" > "$queue_status"
        exit "$code"
    fi
done
printf 'completed %s\n' "$(date -Is)" > "$queue_status"
