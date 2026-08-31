import copy, hashlib, json, time, sys, torch
from pathlib import Path
from src.data.featurizer import PocketFeaturizer
from src.model.flow_matching import FlowMatching
from src.model.pocket_encoder import PocketEncoder
from src.model.egnn import SE3EGNN
from src.model.reward import RewardOracle
from src.train.golden_pilot_experiment import (
    evaluate_golden_protocol,
    train_heuristic_step_g2,
    get_file_sha256
)

print("Starting G2 completion (steps 26 to 50)...", flush=True)
golden_ckpt_path = Path("checkpoints/rl_final.pt")
sha256_golden = get_file_sha256(golden_ckpt_path)

pocket_encoder = PocketEncoder(in_dim=40, hidden_dim=128, num_layers=4, knn_k=16)
egnn = SE3EGNN(ligand_in_dim=4, pocket_dim=128, hidden_dim=128, num_layers=9, knn_k=16, num_atom_types=6)
model_golden = FlowMatching(pocket_encoder=pocket_encoder, egnn=egnn, num_steps=50)
golden_ckpt = torch.load(golden_ckpt_path, map_location="cpu", weights_only=False)
model_golden.load_state_dict(golden_ckpt["model_state_dict"])
model_golden.eval()

model_ref = copy.deepcopy(model_golden)
for p in model_ref.parameters(): p.requires_grad_(False)
model_ref.eval()

model_g2 = copy.deepcopy(model_golden)
g2_step25_ckpt = torch.load("checkpoints/rl_pilot_500/step_025/g2_model.pt", map_location="cpu", weights_only=False)
model_g2.load_state_dict(g2_step25_ckpt["model_state_dict"])
optimizer_g2 = torch.optim.Adam(model_g2.parameters(), lr=5e-6)

pf = PocketFeaturizer()
candidate_pdb_paths = sorted(list(set(
    list(Path("figures").glob("*.pdb")) + list(Path("uploads").glob("**/*.pdb"))
)))
benchmark_pockets = []
for p_path in candidate_pdb_paths:
    try:
        fd = pf.featurize(str(p_path))
        if fd["pos"] is not None and fd["pos"].shape[0] >= 30:
            benchmark_pockets.append({
                "path": str(p_path), "name": p_path.stem, "pos": fd["pos"], "feat": fd["feat"]
            })
    except Exception:
        pass
benchmark_pockets = benchmark_pockets[:5]

reward_oracle = RewardOracle(
    vina_every_n=1, min_carbon_ratio=0.40, max_nitrogen_ratio=0.35, max_nn_bonds=2, max_sa_score=6.0, max_ring_nitrogen=2
)

for step in range(26, 51):
    p_curr = benchmark_pockets[(step - 1) % len(benchmark_pockets)]
    step_diag = train_heuristic_step_g2(
        model=model_g2,
        model_ref=model_ref,
        optimizer=optimizer_g2,
        pocket_pos=p_curr["pos"],
        pocket_feat=p_curr["feat"],
        reward_oracle=reward_oracle,
        pocket_path=p_curr["path"],
        G=4,
        K=20,
        beta=0.01,
    )
    if step % 5 == 0 or step == 50:
        print(f"  G2 step {step:02d}/50 complete...", flush=True)

print("Evaluating G2 at step 50 under official 50-step protocol...", flush=True)
m50 = evaluate_golden_protocol(model_g2, benchmark_pockets, reward_oracle)
m50["kl_loss"] = step_diag["kl_loss"]
step_dir = Path("checkpoints/rl_pilot_500/step_050")
step_dir.mkdir(parents=True, exist_ok=True)
torch.save({"step": 50, "model_state_dict": model_g2.state_dict(), "metrics": m50}, step_dir / "g2_model.pt")

print("G2 step 50 complete! Reward: %.4f, QED: %.4f, PB-Valid: %.1f%%, Div: %.4f" %
      (m50["reward_mean"], m50["qed_mean"], m50["pb_validity_rate"]*100, m50["internal_diversity"]), flush=True)

# Write full consolidated summary.json
g0_metrics = evaluate_golden_protocol(model_golden, benchmark_pockets, reward_oracle)
summary = {
    "metadata": {
        "golden_checkpoint": str(golden_ckpt_path),
        "golden_sha256": sha256_golden,
        "max_steps": 50,
        "G": 4, "K": 20, "lr": 5e-6, "beta": 0.01,
        "benchmark_pockets": [p["name"] for p in benchmark_pockets]
    },
    "G0_Golden_PROTEUS": g0_metrics,
    "G1_SDE_Flow_GRPO": {
        "step_000": torch.load("checkpoints/rl_pilot_500/step_000/g1_model.pt", map_location="cpu", weights_only=False)["metrics"],
        "step_010": torch.load("checkpoints/rl_pilot_500/step_010/g1_model.pt", map_location="cpu", weights_only=False)["metrics"],
        "step_025": torch.load("checkpoints/rl_pilot_500/step_025/g1_model.pt", map_location="cpu", weights_only=False)["metrics"],
        "step_050": torch.load("checkpoints/rl_pilot_500/step_050/g1_model.pt", map_location="cpu", weights_only=False)["metrics"],
    },
    "G2_Historical_Heuristic": {
        "step_000": torch.load("checkpoints/rl_pilot_500/step_000/g1_model.pt", map_location="cpu", weights_only=False)["metrics"],
        "step_010": torch.load("checkpoints/rl_pilot_500/step_010/g2_model.pt", map_location="cpu", weights_only=False)["metrics"],
        "step_025": torch.load("checkpoints/rl_pilot_500/step_025/g2_model.pt", map_location="cpu", weights_only=False)["metrics"],
        "step_050": m50,
    }
}
with open("checkpoints/rl_pilot_500/pilot_500_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Full summary written to checkpoints/rl_pilot_500/pilot_500_summary.json!", flush=True)
