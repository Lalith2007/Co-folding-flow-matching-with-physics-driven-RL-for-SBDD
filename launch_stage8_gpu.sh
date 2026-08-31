#!/usr/bin/env bash
# ==============================================================================
# launch_stage8_gpu.sh — 1-Line Server Execution for Stage 8 Multi-Seed Validation
# ==============================================================================
set -euo pipefail

echo "======================================================================"
echo "PROTEUS STAGE 8: Multi-Seed Final-Scale GPU Validation"
echo "======================================================================"

# 1. Environment and Path setup
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

# 2. Check GPU availability
python3 -c "import torch; print(f'PyTorch CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# 3. Create checkpoints output directory
mkdir -p checkpoints/rl_final_scale

# 4. Launch GPU runner
echo "Starting Stage 8 Validation on GPU..."
python3 -u run_stage8_gpu.py \
    --golden_ckpt "checkpoints/rl_final.pt" \
    --output_dir "checkpoints/rl_final_scale" \
    --benchmark_json "data/benchmark_20_pockets/benchmark_pockets.json" \
    --seeds 42 123 2026 \
    --steps 500 \
    --lr 5e-6 \
    --beta 0.01 \
    --eps_clip 0.20

echo "======================================================================"
echo "Stage 8 Validation Complete! Summary saved to: checkpoints/rl_final_scale/stage8_summary.json"
echo "======================================================================"
