"""Compile a MatterSim model and use it with ASE calculator."""

import torch
from ase.build import bulk

from aoti_mlip.calculators.mattersim import MatterSimCalculator
from aoti_mlip.utils.aoti_compile import compile_mattersim

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Step 1: Compile the model
package_path = compile_mattersim(
    checkpoint_name="mattersim-v1.0.0-5M.pth",
    cutoff=5.0,
    threebody_cutoff=4.0,
    compute_force=True,
    compute_stress=True,
    device=DEVICE,
)

# Step 2: Create atomic structure
atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))

# Step 3: Set up ASE calculator with compiled model
calc = MatterSimCalculator(model_path=package_path, device=DEVICE)
atoms.calc = calc

# Step 4: Calculate properties
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()

# Display results
print(f"Energy: {energy} eV")
print(f"Forces: {forces}")
print(f"Stress: {stress}")
