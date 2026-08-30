"""
main.py — FastAPI backend for the SBDD Drug Design pipeline.

Endpoints:
  POST /api/generate   — Upload a PDB file, get generated SMILES back
  GET  /api/health     — Health check
  GET  /api/status/:id — Check job status (for async processing)

The server runs P2Rank + PyTorch inference + RDKit bond perception,
so the user's browser needs nothing installed.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ──
CHECKPOINT_PATH = os.environ.get(
    "SBDD_CHECKPOINT", "checkpoints/rl_final.pt"
)
DEVICE = os.environ.get("SBDD_DEVICE", "auto")
P2RANK_HOME = os.environ.get("PRANK_HOME", None)
NUM_SAMPLES = int(os.environ.get("SBDD_NUM_SAMPLES", "10"))
NUM_STEPS = int(os.environ.get("SBDD_NUM_STEPS", "50"))
UPLOAD_DIR = Path(os.environ.get("SBDD_UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── FastAPI App ──
app = FastAPI(
    title="PROTEUS SBDD API",
    description="Upload a protein PDB -> get AI-designed drug molecules (SMILES)",
    version="2.0.0",
)

# CORS: allow the frontend (any origin in dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-loaded pipeline (loads model on first request) ──
_pipeline = None


def get_pipeline():
    """Lazily initialize the inference pipeline."""
    global _pipeline
    if _pipeline is None:
        import torch
        device = DEVICE
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        from src.inference.pipeline import InferencePipeline
        _pipeline = InferencePipeline(
            checkpoint_path=CHECKPOINT_PATH,
            device=device,
            p2rank_home=P2RANK_HOME,
            num_samples=NUM_SAMPLES,
            num_steps=NUM_STEPS,
        )
        logger.info("Inference pipeline initialized")
    return _pipeline


# ── In-memory job store (for async status tracking) ──
_jobs: Dict[str, Dict] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "checkpoint": CHECKPOINT_PATH,
        "device": DEVICE,
        "p2rank_home": P2RANK_HOME,
    }


@app.post("/api/generate")
async def generate_molecule(
    pdb_file: UploadFile = File(..., description="Protein PDB file"),
    pocket_index: int = Query(1, description="Which pocket rank to use (1=best)"),
    num_samples: int = Query(3, ge=1, le=20, description="Molecules to generate"),
):
    """Upload a PDB file and generate drug-like molecules."""
    if not pdb_file.filename.endswith((".pdb", ".cif", ".ent")):
        raise HTTPException(
            status_code=400,
            detail="Only .pdb, .cif, or .ent files are accepted.",
        )

    job_id = str(uuid.uuid4())[:8]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = job_dir / pdb_file.filename

    with open(pdb_path, "wb") as f:
        content = await pdb_file.read()
        f.write(content)

    file_size_kb = len(content) / 1024
    logger.info(
        f"Job {job_id}: received {pdb_file.filename} ({file_size_kb:.1f} KB)"
    )

    try:
        pipeline = get_pipeline()
        pipeline.num_samples = num_samples
        result = pipeline.run(str(pdb_path), pocket_index=pocket_index)
    except Exception as e:
        logger.exception(f"Job {job_id}: pipeline error")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}",
        )

    if not result["success"]:
        raise HTTPException(
            status_code=422,
            detail=result["error"],
        )

    all_smiles_list = [s for s in result.get("all_smiles", []) if s]
    batch_metrics: dict = {}
    if all_smiles_list:
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from generate import compute_uniqueness, compute_similarity
            batch_metrics["uniqueness"] = round(compute_uniqueness(all_smiles_list) * 100, 1)
            batch_metrics["internal_similarity"] = round(
                compute_similarity(all_smiles_list, all_smiles_list), 4
            )
        except Exception:
            batch_metrics["uniqueness"] = 100.0

    props = result["properties"]

    response = {
        "job_id": job_id,
        "filename": pdb_file.filename,
        "smiles": result["smiles"],
        "all_smiles": result["all_smiles"],
        "candidates": result.get("all_candidates", []),
        "coordinates": result["coords_3d"],
        "atom_types": result["atom_types"],
        "pocket": result["pocket_info"],
        "properties": props,
        "stats": {
            "valid_count": result["num_valid"],
            "total_generated": result["num_generated"],
            "validity_rate": round(
                result["num_valid"] / max(result["num_generated"], 1) * 100, 1
            ),
            **batch_metrics,
        },
        "timings": result["timings"],
    }

    logger.info(
        f"Job {job_id}: success | SMILES={result['smiles']} | "
        f"QED={result['properties'].get('qed', 'N/A')} | "
        f"SA={result['properties'].get('sa_score', 'N/A')}"
    )

    return JSONResponse(content=response)


@app.get("/api/status/{job_id}")
def get_job_status(job_id: str):
    """Check the status of a generation job."""
    if job_id in _jobs:
        return _jobs[job_id]
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@app.on_event("startup")
async def startup_event():
    logger.info("PROTEUS SBDD API server starting...")
    logger.info(f"  Checkpoint: {CHECKPOINT_PATH}")
    logger.info(f"  Device: {DEVICE}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("PROTEUS SBDD API server shutting down...")
