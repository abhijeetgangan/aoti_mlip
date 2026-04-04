"""Benchmark: batch relaxation of 1000 WBM structures — AOTI vs TorchScript MatterSim."""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_sim as ts
from matbench_discovery.data import DataFiles, ase_atoms_from_zip
from mattersim.forcefield import Potential
from torch_sim.autobatching import InFlightAutoBatcher
from torch_sim.models.mattersim import MatterSimModel

from aoti_mlip.calculators.torchsim import MatterSimTorchSimModel
from aoti_mlip.utils.aoti_compile import compile_mattersim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
CHECKPOINT = "mattersim-v1.0.0-5M.pth"
N_STRUCTURES = 1000
MAX_STEPS = 500
MAX_ATOMS_TO_TRY = 2000
FORCE_TOL = 1e-2
SCRIPT_DIR = os.path.dirname(__file__)

AOTI_OOM_MESSAGES = ["CUDA out of memory", "API call failed"]


def load_wbm_structures(n: int):
    """Load the first *n* initial structures from the WBM dataset."""
    all_atoms = ase_atoms_from_zip(DataFiles.wbm_initial_atoms.path)
    return all_atoms[:n]


def relax_batch(model, structures, label):
    """Relax *structures* and return (energies, wall_time)."""
    print(f"[{label}] Relaxing {len(structures)} structures (max {MAX_STEPS} steps) ...")
    t0 = time.perf_counter()
    relaxed = ts.optimize(
        system=structures,
        model=model,
        optimizer=ts.Optimizer.fire,
        convergence_fn=ts.generate_force_convergence_fn(
            force_tol=FORCE_TOL, include_cell_forces=True
        ),
        max_steps=MAX_STEPS,
        autobatcher=InFlightAutoBatcher(
            model=model,
            memory_scales_with="n_atoms_x_density",
            oom_error_message=AOTI_OOM_MESSAGES,
            max_atoms_to_try=MAX_ATOMS_TO_TRY,
        ),
        pbar=True,
        init_kwargs={"cell_filter": ts.CellFilter.frechet},
    )
    wall = time.perf_counter() - t0

    relaxed_atoms = ts.io.state_to_atoms(relaxed)
    energies = []
    batch_size = 64
    for i in range(0, len(relaxed_atoms), batch_size):
        chunk = relaxed_atoms[i : i + batch_size]
        state = ts.io.atoms_to_state(chunk, device=model.device, dtype=model.dtype)
        energies.append(model(state)["energy"].detach().cpu())
    energies = torch.cat(energies).numpy()
    n_atoms = np.array([len(a) for a in relaxed_atoms])

    print(f"[{label}] Done in {wall:.1f}s")
    return energies, n_atoms, wall


print(f"Device: {DEVICE}")
print(f"Structures: {N_STRUCTURES}")

# Load structures
structures = load_wbm_structures(N_STRUCTURES)
print(f"Loaded {len(structures)} WBM structures")

# Build AOTI model
pkg_path = compile_mattersim(
    checkpoint_name=CHECKPOINT,
    cutoff=5.0,
    threebody_cutoff=4.0,
    compute_force=True,
    compute_stress=True,
    device=DEVICE,
)
aoti_model = MatterSimTorchSimModel(model_path=pkg_path, device=DEVICE, dtype=DTYPE)

# Build TorchScript MatterSim model
potential = Potential.from_checkpoint(load_path=CHECKPOINT, device=DEVICE)
ts_model = MatterSimModel(model=potential, device=torch.device(DEVICE), dtype=DTYPE)


# Relax with AOTI model
aoti_energies, aoti_natoms, aoti_time = relax_batch(
    aoti_model, [a.copy() for a in structures], "AOTI"
)

# Relax with TorchScript model
ts_energies, ts_natoms, ts_time = relax_batch(
    ts_model, [a.copy() for a in structures], "TorchScript"
)

# Per-atom energy MAE
energy_per_atom_diff = aoti_energies / aoti_natoms - ts_energies / ts_natoms
energy_per_atom_mae = np.mean(np.abs(energy_per_atom_diff))

# Print summary
print(f"{'AOTI':>16s}: {aoti_time:8.1f}s")
print(f"{'TorchScript':>16s}: {ts_time:8.1f}s")
print(f"{'Speedup':>16s}: {ts_time / aoti_time:8.2f}x")
print(f"{'E/atom MAE':>16s}: {energy_per_atom_mae:.4e} eV/atom")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

ax = axes[0]
aoti_epa = aoti_energies / aoti_natoms
ts_epa = ts_energies / ts_natoms
ax.scatter(ts_epa, aoti_epa, s=8, alpha=0.5, edgecolors="none")
lo = min(ts_epa.min(), aoti_epa.min())
hi = max(ts_epa.max(), aoti_epa.max())
ax.plot([lo, hi], [lo, hi], "k--", lw=1)
ax.set_xlabel("TorchScript energy (eV/atom)", fontsize=14)
ax.set_ylabel("AOTI energy (eV/atom)", fontsize=14)
ax.set_title(f"Relaxed energy parity\nMAE = {energy_per_atom_mae:.2e} eV/atom", fontsize=14)
ax.set_aspect("equal")

AOTI_COLOR = "#1f77b4"
TS_COLOR = "#ff7f0e"

ax = axes[1]
bar_labels = ["AOTI", "TorchScript"]
times = [aoti_time, ts_time]
colors = [AOTI_COLOR, TS_COLOR]
bars = ax.bar(bar_labels, times, color=colors, edgecolor="black", width=0.5)
for bar, t in zip(bars, times, strict=True):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{t:.1f}s",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
ax.set_ylabel("Wall time (s)", fontsize=14)
speedup = ts_time / aoti_time
ax.set_title(
    f"Relaxation of {N_STRUCTURES} WBM structures\nSpeedup: {speedup:.2f}x",
    fontsize=14,
)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "timing_rtx4070m_batch_relaxation_mattersim-v1.0.0-5M.png")
plt.savefig(out_path, bbox_inches="tight")
print(f"Plot saved to {out_path}")
plt.close()
