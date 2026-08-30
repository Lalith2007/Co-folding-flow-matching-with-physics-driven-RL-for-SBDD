#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# auto_eval.sh — Background Automated Evaluation Launcher
#
# Continuously monitors `checkpoints/` and evaluates every new
# checkpoint (every 1000 steps) as soon as it is saved.
#
# Usage:
#   bash auto_eval.sh
#   nohup bash auto_eval.sh > auto_eval_run.log 2>&1 &
# ──────────────────────────────────────────────────────────────

export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python auto_eval.py "$@"
