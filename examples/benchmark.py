import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from ase import units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from matplotlib.ticker import LogFormatter, LogLocator
from mattersim.forcefield.potential import MatterSimCalculator

from aoti_mlip.calculators.mattersim import MatterSimCalculator as aoti_MatterSimCalculator

# Setup MD parameters
RUN_MD = True
MD_SIZE = 5
MD_STEPS = 200
TEMPERATURE = 300  # Kelvin
TIMESTEP = 1.0  # fs

SCRIPT_DIR = os.path.dirname(__file__)
BASE_PATH = "~/.local/mattersim/pretrained_models/"
MODEL_NAME = "mattersim-v1.0.0-5M.pth"
MODEL_PATH = os.path.expanduser(f"{BASE_PATH}/{MODEL_NAME}")
PACKAGE_PATH = os.path.expanduser(f"{BASE_PATH}/{MODEL_NAME.replace('.pth', '.pt2')}")
OUT_JSON_AOTI = os.path.join(
    SCRIPT_DIR, f"timings_cuda_aoti_{MODEL_NAME.replace('.pth', '').replace('/', '_')}.json"
)
OUT_JSON_TORCH = os.path.join(
    SCRIPT_DIR, f"timings_cuda_torchscript_{MODEL_NAME.replace('.pth', '').replace('/', '_')}.json"
)
PERIODIC = True
N_CALLS = 100
aoti_calculator = aoti_MatterSimCalculator(model_path=PACKAGE_PATH, device="cuda")
calculator = MatterSimCalculator(
    load_path=MODEL_PATH, device="cuda", compute_stress=True, compute_force=True
)
atoms_1 = bulk("Fe", "bcc", a=2.86, cubic=True)
atoms_1.calc = aoti_calculator
print(atoms_1.get_potential_energy())
print(atoms_1.get_forces())
print(atoms_1.get_stress())


atoms_2 = atoms_1.copy()
atoms_2.calc = calculator
print(atoms_2.get_potential_energy())
print(atoms_2.get_forces())
print(atoms_2.get_stress())

test_sizes = [1, 2, 3, 4, 5, 6]

for N in test_sizes:
    print(f"Testing {N}x{N}x{N} structure ({2 * N**3} atoms)")
    atoms_1 = bulk("Fe", "bcc", a=2.86, cubic=True).repeat((N, N, N))
    atoms_1.calc = aoti_calculator
    start_time = time.perf_counter()
    for _ in range(N_CALLS):
        aoti_energy = atoms_1.get_potential_energy()
        aoti_forces = atoms_1.get_forces()
        aoti_stress = atoms_1.get_stress()
    end_time = time.perf_counter()
    aoti_time = (end_time - start_time) / N_CALLS
    atoms_2 = atoms_1.copy()
    atoms_2.calc = calculator
    start_time = time.perf_counter()
    for _ in range(N_CALLS):
        energy = atoms_2.get_potential_energy()
        forces = atoms_2.get_forces()
        stress = atoms_2.get_stress()
    end_time = time.perf_counter()
    original_time = (end_time - start_time) / N_CALLS
    print(f"Energy: {aoti_energy:.4f} eV")
    print(f"Time: {aoti_time * 1000:.2f} ms (avg of {N_CALLS} runs)")
    print(f"Time: {original_time * 1000:.2f} ms (avg of {N_CALLS} runs)")
    print(f"Speedup: {original_time / aoti_time:.1f}x")

    # Verify correctness
    energy_match = np.allclose(aoti_energy, energy, atol=1e-4)
    forces_match = np.allclose(aoti_forces, forces, atol=1e-4)
    stress_match = np.allclose(aoti_stress, stress, atol=1e-4)
    print(f"Correctness: Energy={energy_match}, Forces={forces_match}, Stress={stress_match}")


aoti_timings = {}
original_timings = {}
N_multiples_aoti = (
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    if MODEL_NAME == "mattersim-v1.0.0-5M.pth"
    else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
)
N_multiples_original = (
    [1, 2, 3, 4, 5, 6, 7] if MODEL_NAME == "mattersim-v1.0.0-5M.pth" else [1, 2, 3, 4, 5, 7, 8, 9]
)
for N in N_multiples_aoti:
    print(f"Testing {N}x{N}x{N} structure ({2 * N**3} atoms)")
    atoms_1 = bulk("Fe", "bcc", a=2.86, cubic=True).repeat((N, N, N))
    atoms_1.calc = aoti_calculator
    start_time = time.perf_counter()
    for _ in range(N_CALLS):
        aoti_energy = atoms_1.get_potential_energy()
        aoti_forces = atoms_1.get_forces()
        aoti_stress = atoms_1.get_stress()
    end_time = time.perf_counter()
    aoti_timings[len(atoms_1)] = (end_time - start_time) / N_CALLS

# Save separate files for each implementation
entries_aoti = [
    {"num_atoms": int(n), "time": float(aoti_timings[n])} for n in sorted(aoti_timings.keys())
]
payload_aoti = {"n_calls": int(N_CALLS), "periodic": bool(PERIODIC), "entries": entries_aoti}
with open(OUT_JSON_AOTI, "w", encoding="utf-8") as f:
    json.dump(payload_aoti, f, indent=2)
print(f"Saved aoti timings to {OUT_JSON_AOTI}")

for N in N_multiples_original:
    print(f"Testing {N}x{N}x{N} structure ({2 * N**3} atoms)")
    atoms_2 = bulk("Fe", "bcc", a=2.86, cubic=True).repeat((N, N, N))
    atoms_2.calc = calculator
    start_time = time.perf_counter()
    for _ in range(N_CALLS):
        energy = atoms_2.get_potential_energy()
        forces = atoms_2.get_forces()
        stress = atoms_2.get_stress()
    end_time = time.perf_counter()
    original_timings[len(atoms_2)] = (end_time - start_time) / N_CALLS


entries_orig = [
    {"num_atoms": int(n), "time": float(original_timings[n])}
    for n in sorted(original_timings.keys())
]
payload_orig = {"n_calls": int(N_CALLS), "periodic": bool(PERIODIC), "entries": entries_orig}

with open(OUT_JSON_TORCH, "w", encoding="utf-8") as f:
    json.dump(payload_orig, f, indent=2)
print(f"Saved torch script timings to {OUT_JSON_TORCH}")

print("\nGenerating timing comparison plot...")

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
ax.set_title(f"CUDA timings for {MODEL_NAME}", fontsize=18)

# Plot AOTI timings
aoti_xs = [e["num_atoms"] for e in entries_aoti]
aoti_ts = [e["time"] for e in entries_aoti]
aoti_throughput = [N_CALLS / t for t in aoti_ts]
ax.loglog(aoti_xs, aoti_throughput, label="AOTI", marker="o", markersize=8)

# Plot TorchScript timings
torch_xs = [e["num_atoms"] for e in entries_orig]
torch_ts = [e["time"] for e in entries_orig]
torch_throughput = [N_CALLS / t for t in torch_ts]
ax.loglog(torch_xs, torch_throughput, label="TorchScript", marker="o", markersize=8)

# Configure axes
ax.set_xscale("log", base=2)
ax.xaxis.set_major_locator(LogLocator(base=2))
ax.xaxis.set_major_formatter(LogFormatter(base=2))

ax.set_xlabel("Number of Atoms", fontsize=14)
ax.set_ylabel("Throughput [timesteps/sec]", fontsize=14)
ax.grid(True, which="major", linestyle="-")
ax.grid(True, which="minor", linestyle="--", alpha=0.5)
ax.set_ylim(1e4, 1e6)
ax.legend(fontsize=12)

# Save timing plot
timing_plot_path = os.path.join(
    os.getcwd(), f"mattersim_timings_cuda_{MODEL_NAME.replace('.pth', '').replace('/', '_')}.png"
)
plt.savefig(timing_plot_path, bbox_inches="tight")
print(f"Timing plot saved to: {timing_plot_path}")
plt.close()

if RUN_MD:
    print(f"Running MD simulation comparison ({MD_STEPS} steps)")

    # Create structure
    md_atoms_aoti = bulk("Fe", "bcc", a=2.86, cubic=True).repeat((MD_SIZE, MD_SIZE, MD_SIZE))
    md_atoms_orig = md_atoms_aoti.copy()

    # Set calculators
    md_atoms_aoti.calc = aoti_calculator
    md_atoms_orig.calc = calculator

    # Initialize velocities
    MaxwellBoltzmannDistribution(md_atoms_aoti, temperature_K=TEMPERATURE)
    md_atoms_orig.set_velocities(md_atoms_aoti.get_velocities())

    # Storage for trajectory data
    aoti_energies = []
    aoti_forces_all = []
    aoti_stresses = []

    orig_energies = []
    orig_forces_all = []
    orig_stresses = []

    # Run MD with AOTI calculator
    dyn_aoti = VelocityVerlet(md_atoms_aoti, timestep=TIMESTEP * units.fs)
    md_start_aoti = time.perf_counter()
    for step in range(MD_STEPS):
        dyn_aoti.run(1)
        aoti_energies.append(md_atoms_aoti.get_potential_energy())
        aoti_forces_all.append(md_atoms_aoti.get_forces().copy())
        aoti_stresses.append(md_atoms_aoti.get_stress().copy())
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{MD_STEPS}: E = {aoti_energies[-1]:.4f} eV")
    md_end_aoti = time.perf_counter()
    md_time_aoti = md_end_aoti - md_start_aoti
    timesteps_per_sec_aoti = MD_STEPS / md_time_aoti

    # Run MD with original calculator
    dyn_orig = VelocityVerlet(md_atoms_orig, timestep=TIMESTEP * units.fs)
    md_start_orig = time.perf_counter()
    for step in range(MD_STEPS):
        dyn_orig.run(1)
        orig_energies.append(md_atoms_orig.get_potential_energy())
        orig_forces_all.append(md_atoms_orig.get_forces().copy())
        orig_stresses.append(md_atoms_orig.get_stress().copy())
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{MD_STEPS}: E = {orig_energies[-1]:.4f} eV")
    md_end_orig = time.perf_counter()
    md_time_orig = md_end_orig - md_start_orig
    timesteps_per_sec_orig = MD_STEPS / md_time_orig

    # Convert to numpy arrays
    aoti_energies = np.array(aoti_energies)
    orig_energies = np.array(orig_energies)
    aoti_forces_all = np.array(aoti_forces_all)
    orig_forces_all = np.array(orig_forces_all)
    aoti_stresses = np.array(aoti_stresses)
    orig_stresses = np.array(orig_stresses)

    # Flatten forces for parity plot
    aoti_forces_flat = aoti_forces_all.flatten()
    orig_forces_flat = orig_forces_all.flatten()

    # Calculate statistics
    energy_mae = np.mean(np.abs(aoti_energies - orig_energies))
    energy_rmse = np.sqrt(np.mean((aoti_energies - orig_energies) ** 2))
    forces_mae = np.mean(np.abs(aoti_forces_flat - orig_forces_flat))
    forces_rmse = np.sqrt(np.mean((aoti_forces_flat - orig_forces_flat) ** 2))
    stress_mae = np.mean(np.abs(aoti_stresses - orig_stresses))
    stress_rmse = np.sqrt(np.mean((aoti_stresses - orig_stresses) ** 2))

    print("MD Comparison Statistics:")
    print(f"Energy  - MAE: {energy_mae:.3e} eV, RMSE: {energy_rmse:.3e} eV")
    print(f"Forces  - MAE: {forces_mae:.3e} eV/Å, RMSE: {forces_rmse:.3e} eV/Å")
    print(f"Stress  - MAE: {stress_mae:.3e} eV/Å³, RMSE: {stress_rmse:.3e} eV/Å³")
    print("\nMD Performance:")
    print(
        f"AOTI Calculator    - {timesteps_per_sec_aoti:.2f} steps/sec ({md_time_aoti:.2f}s total)"
    )
    print(
        f"Original Calculator - {timesteps_per_sec_orig:.2f} steps/sec ({md_time_orig:.2f}s total)"
    )
    print(f"Speedup: {timesteps_per_sec_aoti / timesteps_per_sec_orig:.2f}x")

    # Create parity plots with timing
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"MD Comparison: AOTI vs Original ({MD_SIZE}x{MD_SIZE}x{MD_SIZE} Fe, {MD_STEPS} steps)",
        fontsize=14,
        fontweight="bold",
    )

    # Energy parity plot
    ax = axes[0, 0]
    ax.scatter(orig_energies, aoti_energies, alpha=0.6, s=30, edgecolors="k", linewidth=0.5)
    min_e, max_e = orig_energies.min(), orig_energies.max()
    ax.plot([min_e, max_e], [min_e, max_e], "r--", lw=2, label="y=x")
    ax.set_xlabel("Original Calculator Energy (eV)", fontsize=12)
    ax.set_ylabel("AOTI Calculator Energy (eV)", fontsize=12)
    ax.set_title(f"Energy Parity\nMAE = {energy_mae:.2e} eV", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Forces parity plot
    ax = axes[0, 1]
    # Subsample for visualization if too many points
    max_points = 10000
    if len(aoti_forces_flat) > max_points:
        indices = np.random.choice(len(aoti_forces_flat), max_points, replace=False)
        plot_orig_forces = orig_forces_flat[indices]
        plot_aoti_forces = aoti_forces_flat[indices]
    else:
        plot_orig_forces = orig_forces_flat
        plot_aoti_forces = aoti_forces_flat

    ax.scatter(plot_orig_forces, plot_aoti_forces, alpha=0.3, s=15, edgecolors="none")
    min_f, max_f = orig_forces_flat.min(), orig_forces_flat.max()
    ax.plot([min_f, max_f], [min_f, max_f], "r--", lw=2, label="y=x")
    ax.set_xlabel("Original Calculator Forces (eV/Å)", fontsize=12)
    ax.set_ylabel("AOTI Calculator Forces (eV/Å)", fontsize=12)
    ax.set_title(f"Forces Parity\nMAE = {forces_mae:.2e} eV/Å", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Stress parity plot
    ax = axes[1, 0]
    stress_flat_orig = orig_stresses.flatten()
    stress_flat_aoti = aoti_stresses.flatten()
    ax.scatter(stress_flat_orig, stress_flat_aoti, alpha=0.4, s=20, edgecolors="k", linewidth=0.3)
    min_s, max_s = stress_flat_orig.min(), stress_flat_orig.max()
    ax.plot([min_s, max_s], [min_s, max_s], "r--", lw=2, label="y=x")
    ax.set_xlabel("Original Calculator Stress (eV/Å³)", fontsize=12)
    ax.set_ylabel("AOTI Calculator Stress (eV/Å³)", fontsize=12)
    ax.set_title(f"Stress Parity\nMAE = {stress_mae:.2e} eV/Å³", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Performance bar plot
    ax = axes[1, 1]
    calculators = ["AOTI", "Original"]
    timesteps_per_sec = [timesteps_per_sec_aoti, timesteps_per_sec_orig]
    colors = ["#2ecc71", "#3498db"]
    bars = ax.bar(calculators, timesteps_per_sec, color=colors, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Timesteps per Second", fontsize=12)
    speedup = timesteps_per_sec_aoti / timesteps_per_sec_orig
    ax.set_title(f"MD Performance\nSpeedup: {speedup:.2f}x", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    # Add value labels on bars
    for bar, value in zip(bars, timesteps_per_sec, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()

    # Save figure
    md_plot_path = os.path.join(SCRIPT_DIR, f"md_comparison_{MODEL_NAME.replace('.pth', '')}.png")
    plt.savefig(md_plot_path, dpi=300, bbox_inches="tight")
    print(f"MD comparison plot saved to: {md_plot_path}")
    plt.close()
