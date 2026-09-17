#!/usr/bin/env bash
set -u

workspace=/root/autodl-fs/HyperDAF-MSA-runs/lnln_native_200_seed1111
log_dir="$workspace/log"
python_bin=/root/miniconda3/bin/python

cd "$workspace" || exit 90
mkdir -p "$log_dir" ckpt/mosi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=0

date -Is > "$log_dir/start_time.txt"
sha256sum /root/HyperDAF-MSA/data/mmsa/MOSI/Processed/unaligned_50.pkl > "$log_dir/data_sha256.txt"
sha256sum train.py core/*.py models/*.py configs/train_mosi_native_200_seed1111.yaml > "$log_dir/source_sha256.txt"
nvidia-smi > "$log_dir/nvidia_smi_start.txt"
"$python_bin" -c 'import sys, torch, transformers, einops, sklearn, yaml; print("python", sys.version); print("torch", torch.__version__); print("cuda", torch.cuda.is_available(), torch.version.cuda); print("gpu", torch.cuda.get_device_name(0)); print("transformers", transformers.__version__); print("einops", einops.__version__); print("sklearn", sklearn.__version__); print("yaml", yaml.__version__)' > "$log_dir/environment.txt" 2>&1

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 5 > "$log_dir/gpu_monitor.csv" 2>&1 &
monitor_pid=$!

start_epoch_seconds=$(date +%s)
"$python_bin" -u train.py --config_file configs/train_mosi_native_200_seed1111.yaml --seed 1111 > "$log_dir/train_native_200_seed1111.log" 2>&1
return_code=$?
end_epoch_seconds=$(date +%s)

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true

printf '%s\n' "$return_code" > "$log_dir/exit_code.txt"
printf '%s\n' "$((end_epoch_seconds - start_epoch_seconds))" > "$log_dir/elapsed_sec.txt"
date -Is > "$log_dir/end_time.txt"
nvidia-smi > "$log_dir/nvidia_smi_end.txt"
find ckpt/mosi -maxdepth 1 -type f -printf '%f\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\n' | sort > "$log_dir/checkpoints.txt"

exit "$return_code"
