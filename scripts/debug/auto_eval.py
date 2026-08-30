#!/usr/bin/env python3
"""
auto_eval.py — Automated Continuous Evaluation Watcher for SBDD.

Continuously monitors the `checkpoints/` directory. Whenever a new
checkpoint is saved (e.g. rl_step1000.pt, rl_step2000.pt, ...), it automatically:
  1. Executes the comprehensive 11-metric evaluation suite (evaluate.py).
  2. Saves individual checkpoint metrics to `evaluation_results/results_step<N>.json`.
  3. Updates a live Markdown progression table in `evaluation_results/benchmark_progression.md`.

Usage:
  # Run in the background:
  python auto_eval.py &
  # or
  bash auto_eval.sh
"""

import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Thread control
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [auto_eval] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("auto_eval.log"),
    ],
)
logger = logging.getLogger("auto_eval")


def get_step_number(ckpt_path: str) -> int:
    """Extract step number from filename like rl_step2000.pt."""
    match = re.search(r"step(\d+)\.pt", Path(ckpt_path).name)
    if match:
        return int(match.group(1))
    if "final" in Path(ckpt_path).name:
        return 999999
    return 0


def update_progression_markdown(results_history: list[dict], output_md: Path):
    """Write an aggregated markdown scorecard table tracking progression across all steps."""
    lines = [
        "# SBDD Model Training Progression Scorecard",
        "",
        "| Step | Validity | PoseBusters (PB-Valid) | QED (0-1) | SA Score (raw) | Vina Score | Diversity | Test Loss |",
        "|---|---|---|---|---|---|---|---|",
    ]

    # Sort by step number
    sorted_history = sorted(results_history, key=lambda x: x.get("step", 0))

    for r in sorted_history:
        step_name = f"Step {r['step']}" if r['step'] != 999999 else "Final"
        validity = f"{r.get('validity', 0)*100:.1f}%"
        pb_valid = f"{r.get('pb_valid', 0)*100:.1f}%"
        qed = f"{r.get('qed_mean', 0):.4f}"
        sa_raw = f"{r.get('sa_raw_mean', 0):.4f}"
        vina = f"{r.get('vina_mean', 0):.2f}"
        diversity = f"{r.get('diversity', 0):.4f}"
        test_loss = f"{r.get('test_loss', 0):.4f}" if r.get('test_loss') is not None else "N/A"

        lines.append(
            f"| **{step_name}** | {validity} | **{pb_valid}** | {qed} | {sa_raw} | {vina} | {diversity} | {test_loss} |"
        )

    lines.append("")
    lines.append(f"*Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}*")

    with open(output_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Updated progression table: {output_md}")


def main():
    checkpoints_dir = Path("checkpoints")
    eval_results_dir = Path("evaluation_results")
    eval_results_dir.mkdir(parents=True, exist_ok=True)
    summary_md = eval_results_dir / "benchmark_progression.md"
    history_json = eval_results_dir / "history.json"

    # Load existing history if present
    results_history = []
    if history_json.exists():
        try:
            with open(history_json, "r") as f:
                results_history = json.load(f)
        except Exception:
            results_history = []

    evaluated_checkpoints = set(r.get("checkpoint") for r in results_history if "checkpoint" in r)

    logger.info("============================================================")
    logger.info("Auto-Evaluation Watcher started.")
    logger.info("Monitoring `checkpoints/` for new checkpoints every 30s...")
    logger.info("============================================================")

    while True:
        try:
            # Find all rl_step*.pt and rl_final.pt checkpoints
            ckpts = sorted(
                glob.glob("checkpoints/rl_step*.pt") + glob.glob("checkpoints/rl_final.pt"),
                key=get_step_number
            )

            for ckpt in ckpts:
                ckpt_path = Path(ckpt)
                if str(ckpt_path) in evaluated_checkpoints:
                    continue

                step_num = get_step_number(str(ckpt_path))
                logger.info(f"\n🚀 New checkpoint detected: {ckpt_path} (Step {step_num})")
                logger.info(f"Running automated evaluation benchmark...")

                # Run evaluate.py
                cmd = [
                    sys.executable,
                    "evaluate.py",
                    "--checkpoint", str(ckpt_path),
                    "--config", "configs/default.yaml",
                    "--max_test_samples", "50",
                    "--num_pockets", "20",
                    "--num_gen_mols", "10",
                    "--device", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu",
                    "--output", str(eval_results_dir / f"step_{step_num}"),
                ]

                env = os.environ.copy()
                env["OPENBLAS_NUM_THREADS"] = "4"
                env["OMP_NUM_THREADS"] = "4"
                env["MKL_NUM_THREADS"] = "4"

                t0 = time.time()
                res = subprocess.run(cmd, env=env, capture_output=True, text=True)
                elapsed = time.time() - t0

                if res.returncode != 0:
                    logger.error(f"Evaluation failed for {ckpt_path}:\n{res.stderr}")
                    continue

                logger.info(f"Evaluation completed in {elapsed:.1f}s.")

                # Read output JSON
                step_res_path = eval_results_dir / f"step_{step_num}" / "evaluation_results.json"
                if not step_res_path.exists():
                    # Fallback to main evaluation_results.json
                    step_res_path = eval_results_dir / "evaluation_results.json"

                if step_res_path.exists():
                    try:
                        with open(step_res_path, "r") as f:
                            data = json.load(f)

                        gen = data.get("generation", {})
                        tloss = data.get("test_loss", {})
                        
                        summary_entry = {
                            "checkpoint": str(ckpt_path),
                            "step": step_num,
                            "validity": gen.get("validity_rate", 0),
                            "pb_valid": gen.get("posebusters_valid_rate", 0),
                            "qed_mean": gen.get("qed_mean", 0),
                            "sa_raw_mean": gen.get("sa_raw_mean", 0),
                            "vina_mean": gen.get("vina_score_mean", 0),
                            "diversity": gen.get("diversity", 0),
                            "test_loss": tloss.get("test_loss", None),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }

                        results_history.append(summary_entry)
                        evaluated_checkpoints.add(str(ckpt_path))

                        # Save updated history
                        with open(history_json, "w") as f:
                            json.dump(results_history, f, indent=2)

                        # Update Markdown scorecard
                        update_progression_markdown(results_history, summary_md)

                        logger.info(
                            f"[EVAL Step {step_num}] PB-Valid={summary_entry['pb_valid']*100:.1f}% | "
                            f"QED={summary_entry['qed_mean']:.4f} | "
                            f"SA={summary_entry['sa_raw_mean']:.4f} | "
                            f"Diversity={summary_entry['diversity']:.4f}"
                        )
                    except Exception as parse_err:
                        logger.error(f"Error parsing evaluation output: {parse_err}")

        except Exception as loop_err:
            logger.error(f"Error in watcher loop: {loop_err}")

        time.sleep(30)  # Check every 30 seconds


if __name__ == "__main__":
    main()
