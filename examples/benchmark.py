import json
import os
import time

import numpy as np
from ase.build import bulk
from mattersim.forcefield.potential import MatterSimCalculator

from aoti_mlip.calculators.mattersim import MatterSimCalculator as aoti_MatterSimCalculator

SCRIPT_DIR = os.path.dirname(__file__)
BASE_PATH = "~/.local/mattersim/pretrained_models/"
MODEL_NAME = "mattersim-v1.0.0-1M.pth"
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
