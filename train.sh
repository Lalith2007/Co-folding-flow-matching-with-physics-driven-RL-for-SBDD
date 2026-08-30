#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# train.sh — Universal Auto-Detect Training Launcher
#
# Automatically detects available GPUs / MIG instances:
#   - If 2+ GPUs/MIGs: Launches multi-GPU DDP via torchrun
#   - If 1 GPU/MIG:    Launches single-GPU training
# Always auto-resumes from the latest checkpoint.
#
# Usage:
#   bash train.sh
#   ./train.sh
# ──────────────────────────────────────────────────────────────

set -e

# Cap threads to prevent OpenMP thread exhaustion in DDP multi-GPU containers
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TORCH_NUM_THREADS=2

# ── Check if single GPU / first MIG slice is requested or CUDA_VISIBLE_DEVICES is set ──
if [ "$1" = "--single" ]; then
    shift
    SINGLE_GPU="1"
fi

if [ "$SINGLE_GPU" = "1" ] || [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        # Pick the FIRST MIG slice only
        FIRST_MIG=$(nvidia-smi -L 2>/dev/null | grep "MIG" | head -n 1 | awk -F 'UUID: ' '{print $2}' | tr -d ')' || true)
        if [ -n "$FIRST_MIG" ]; then
            export CUDA_VISIBLE_DEVICES="$FIRST_MIG"
        else
            export CUDA_VISIBLE_DEVICES="0"
        fi
    fi
    
    echo "================================================================="
    echo "⚡ Launching on Single GPU (1st Slice): $CUDA_VISIBLE_DEVICES"
    echo "   (2nd GPU slice is 100% free for teammates)"
    echo "================================================================="
    exec python run_training.py \
        --phase B \
        --checkpoint auto "$@"
fi

# Detect MIG instances first if in MIG mode
MIG_UUIDS=$(nvidia-smi -L 2>/dev/null | grep "MIG" | awk -F 'UUID: ' '{print $2}' | tr -d ')' | paste -sd, - || true)
if [ -n "$MIG_UUIDS" ]; then
    NUM_MIG=$(echo "$MIG_UUIDS" | tr ',' '\n' | grep -v '^$' | wc -l)
else
    NUM_MIG=0
fi

if [ "$NUM_MIG" -ge 2 ]; then
    echo "================================================================="
    echo "🚀 Detected $NUM_MIG MIG devices: Launching DDP Multi-GPU (torchrun)"
    echo "   Devices: $MIG_UUIDS"
    echo "================================================================="
    export CUDA_VISIBLE_DEVICES="$MIG_UUIDS"
    MASTER_PORT=$((29500 + RANDOM % 500))
    exec torchrun --nproc_per_node="$NUM_MIG" --master_port="$MASTER_PORT" run_training.py \
        --phase B \
        --checkpoint auto "$@"

elif [ "$NUM_MIG" -eq 1 ]; then
    echo "================================================================="
    echo "⚡ Detected 1 MIG device: Launching Single-GPU"
    echo "   Device: $MIG_UUIDS"
    echo "================================================================="
    export CUDA_VISIBLE_DEVICES="$MIG_UUIDS"
    exec python run_training.py \
        --phase B \
        --checkpoint auto "$@"

else
    # Standard GPU detection (non-MIG)
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    if [ "$NUM_GPUS" -ge 2 ]; then
        echo "================================================================="
        echo "🚀 Detected $NUM_GPUS GPUs: Launching DDP Multi-GPU (torchrun)"
        echo "================================================================="
        MASTER_PORT=$((29500 + RANDOM % 500))
        exec torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" run_training.py \
            --phase B \
            --checkpoint auto "$@"
    else
        echo "================================================================="
        echo "⚡ Detected 1 GPU / Default: Launching Single-GPU"
        echo "================================================================="
        exec python run_training.py \
            --phase B \
            --checkpoint auto "$@"
    fi
fi
