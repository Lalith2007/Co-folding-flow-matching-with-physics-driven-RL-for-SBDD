#!/usr/bin/env python3
"""
debug_scores.py — Deep Diagnostic Tool for SA and Vina Score Analysis
"""

import os
import sys
import json
from collections import Counter
import numpy as np

def run_diagnostics(results_json_path: str = "evaluation_results_fixed/per_molecule_details.json"):
    if not os.path.exists(results_json_path):
        for alt in ["evaluation_results_levers/per_molecule_details.json", "evaluation_results_final/per_molecule_details.json", "per_molecule_details.json"]:
            if os.path.exists(alt):
                results_json_path = alt
                break

    if not os.path.exists(results_json_path):
        print(f"Error: Could not find results file at {results_json_path}")
        return

    print("=" * 75)
    print(f"  DEEP DIAGNOSTIC ANALYSIS OF: {results_json_path}")
    print("=" * 75)

    with open(results_json_path, "r") as f:
        data = json.load(f)

    print(f"Total molecules evaluated: {len(data)}")

    # 1. SA SCORE DEEP-DIVE
    print("\n" + "─" * 40)
    print("1. SYNTHETIC ACCESSIBILITY (SA) SCORE ANALYSIS")
    print("─" * 40)

    sa_scores = [m["sa_score"] for m in data if m.get("sa_score") is not None]
    if sa_scores:
        print(f"Min SA Score    : {np.min(sa_scores):.2f} (Easiest to synthesize)")
        print(f"10th Percentile : {np.percentile(sa_scores, 10):.2f}")
        print(f"25th Percentile : {np.percentile(sa_scores, 25):.2f}")
        print(f"Median SA Score : {np.median(sa_scores):.2f}")
        print(f"Mean SA Score   : {np.mean(sa_scores):.2f}")
        print(f"Max SA Score    : {np.max(sa_scores):.2f} (Hardest to synthesize)")

        b_easy = sum(1 for s in sa_scores if s <= 4.0)
        b_med = sum(1 for s in sa_scores if 4.0 < s <= 6.0)
        b_poly = sum(1 for s in sa_scores if s > 6.0)

        print(f"\nSA Distribution Bands:")
        print(f"  - Drug-like Easy (SA <= 4.0)     : {b_easy:>3} mols ({b_easy/len(sa_scores)*100:.1f}%)")
        print(f"  - Moderate (4.0 < SA <= 6.0)     : {b_med:>3} mols ({b_med/len(sa_scores)*100:.1f}%)")
        print(f"  - Polycyclic / Caged (SA > 6.0)  : {b_poly:>3} mols ({b_poly/len(sa_scores)*100:.1f}%)")

    ring_counts = [m.get("num_rings", 0) for m in data if m.get("num_rings") is not None]
    if ring_counts:
        print(f"\nRing Count Distribution in Generated SMILES:")
        print(f"  Mean Rings per Mol: {np.mean(ring_counts):.1f}")
        print(f"  Max Rings per Mol : {np.max(ring_counts)}")
        rc_counter = Counter(ring_counts)
        for r_num in sorted(rc_counter.keys())[:8]:
            print(f"    {r_num} rings: {rc_counter[r_num]} molecules")

    # 2. VINA AFFINITY DEEP-DIVE
    print("\n" + "─" * 40)
    print("2. AUTODOCK VINA BINDING AFFINITY ANALYSIS")
    print("─" * 40)

    vina_scores = [m["vina_score_kcal"] for m in data if m.get("vina_score_kcal") is not None]
    if vina_scores:
        neg_scores = [v for v in vina_scores if v < 0.0]
        pos_scores = [v for v in vina_scores if v > 0.0]
        zero_scores = [v for v in vina_scores if v == 0.0]

        print(f"Total Vina Evaluations : {len(vina_scores)}")
        print(f"Favorable Binders (< 0): {len(neg_scores)} ({len(neg_scores)/len(vina_scores)*100:.1f}%)")
        print(f"Zero / Unscored (= 0)  : {len(zero_scores)} ({len(zero_scores)/len(vina_scores)*100:.1f}%)")
        print(f"Positive Clashes (> 0) : {len(pos_scores)} ({len(pos_scores)/len(vina_scores)*100:.1f}%)")

        if neg_scores:
            print(f"\nFavorable Binders Distribution (kcal/mol):")
            print(f"  - Peak Affinity (Lowest Energy): {np.min(neg_scores):.2f} kcal/mol")
            print(f"  - Mean Favorable Affinity      : {np.mean(neg_scores):.2f} kcal/mol")
            print(f"  - Median Favorable Affinity    : {np.median(neg_scores):.2f} kcal/mol")
            print(f"  - Potent Binders (<= -7.0)     : {sum(1 for v in neg_scores if v <= -7.0)} mols")

        trimmed = [v for v in vina_scores if v < 15.0]
        print(f"\nOutlier Impact on Mean Vina Score:")
        print(f"  - Raw Untrimmed Mean : {np.mean(vina_scores):.2f} kcal/mol (distorted by {sum(1 for v in vina_scores if v >= 15.0)} steric wall clashes)")
        print(f"  - Robust Trimmed Mean: {np.mean(trimmed):.2f} kcal/mol")
        print(f"  - Population Median  : {np.median(vina_scores):.2f} kcal/mol")

    # 3. SAMPLE MOLECULAR INSPECTION
    print("\n" + "─" * 40)
    print("3. SAMPLE CANDIDATE STRUCTURES")
    print("─" * 40)

    valid_mols = [m for m in data if m.get("valid") and m.get("smiles")]
    valid_mols.sort(key=lambda x: (x.get("sa_score", 10.0) - (x.get("qed", 0.0) * 2.0)))

    print("Top 3 Lowest-SA (Easiest to Synthesize) Candidates:")
    for i, m in enumerate(valid_mols[:3], 1):
        print(f"  #{i} SMILES : {m['smiles']}")
        print(f"     SA Score: {m.get('sa_score'):.2f} | QED: {m.get('qed'):.3f} | Vina: {m.get('vina_score_kcal')} kcal/mol | Rings: {m.get('num_rings')}")

    print("\n" + "=" * 75)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "evaluation_results_fixed/per_molecule_details.json"
    run_diagnostics(path)
