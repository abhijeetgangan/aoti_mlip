import os
from pathlib import Path

import numpy as np
import torch

from aoti_mlip.calculators.mattersim import MatterSimCalculator as aoti_MatterSimCalculator
from aoti_mlip.models.mattersim_modules.dataloader.build import build_dataloader, unpack_graph_batch
from aoti_mlip.utils.aoti_compile import compile_mattersim

try:
    from mattersim.forcefield.potential import (
        MatterSimCalculator,  # type: ignore[unresolved-import]
    )
except ImportError as err:
    raise ImportError("Mattersim is not installed") from err


def _ensure_checkpoint_available(checkpoint_name: str) -> str:
    target_dir = Path.home() / ".local" / "mattersim" / "pretrained_models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / checkpoint_name
    if target_path.exists():
        return str(target_path)
    repo_root = Path(__file__).resolve().parents[2]
    local_pth = repo_root / checkpoint_name
    if local_pth.exists():
        target_path.write_bytes(local_pth.read_bytes())
    return str(target_path)


def test_aot_output_match_casio3(casio3_atoms):
    checkpoint = "mattersim-v1.0.0-1M.pth"
    _ensure_checkpoint_available(checkpoint)

    pkg_path = compile_mattersim(
        checkpoint_name=checkpoint,
        cutoff=5.0,
        threebody_cutoff=4.0,
        compute_force=True,
        compute_stress=True,
        device="cpu",
    )
    assert os.path.exists(pkg_path)

    atoms_1 = casio3_atoms.copy()
    atoms_1.calc = aoti_MatterSimCalculator(model_path=pkg_path, device="cpu")

    energy = atoms_1.get_potential_energy()
    forces = atoms_1.get_forces()
    stress = atoms_1.get_stress()

    atoms_2 = atoms_1.copy()
    atoms_2.calc = MatterSimCalculator(
        load_path=checkpoint, device="cpu", compute_stress=True, compute_force=True
    )

    energy_ref = atoms_2.get_potential_energy()
    forces_ref = atoms_2.get_forces()
    stress_ref = atoms_2.get_stress()

    assert np.allclose(energy, energy_ref, atol=1e-4)
    assert np.allclose(forces, forces_ref, atol=1e-4)
    assert np.allclose(stress, stress_ref, atol=1e-4)


def test_aot_output_match_fe(fe_atoms):
    checkpoint = "mattersim-v1.0.0-5M.pth"
    _ensure_checkpoint_available(checkpoint)

    pkg_path = compile_mattersim(
        checkpoint_name=checkpoint,
        cutoff=5.0,
        threebody_cutoff=4.0,
        compute_force=True,
        compute_stress=True,
        device="cpu",
    )
    assert os.path.exists(pkg_path)

    atoms_1 = fe_atoms.copy()
    atoms_1.calc = aoti_MatterSimCalculator(model_path=pkg_path, device="cpu")

    energy = atoms_1.get_potential_energy()
    forces = atoms_1.get_forces()
    stress = atoms_1.get_stress()

    atoms_2 = atoms_1.copy()
    atoms_2.calc = MatterSimCalculator(
        load_path=checkpoint, device="cpu", compute_stress=True, compute_force=True
    )

    energy_ref = atoms_2.get_potential_energy()
    forces_ref = atoms_2.get_forces()
    stress_ref = atoms_2.get_stress()

    assert np.allclose(energy, energy_ref, atol=1e-4)
    assert np.allclose(forces, forces_ref, atol=1e-4)
    assert np.allclose(stress, stress_ref, atol=1e-4)


def test_aot_batch_matches_individual(fe_atoms, casio3_atoms):
    """Batched inference on two different-shaped structures should match individual runs."""
    checkpoint = "mattersim-v1.0.0-1M.pth"
    _ensure_checkpoint_available(checkpoint)

    pkg_path = compile_mattersim(
        checkpoint_name=checkpoint,
        cutoff=5.0,
        threebody_cutoff=4.0,
        compute_force=True,
        compute_stress=True,
        device="cpu",
    )
    assert os.path.exists(pkg_path)

    aot_model = torch._inductor.aoti_load_package(pkg_path)
    metadata = aot_model.get_metadata()
    cutoff = float(metadata["cutoff"])
    threebody_cutoff = float(metadata["threebody_cutoff"])

    atoms_list = [fe_atoms, casio3_atoms]

    # Run each structure individually
    individual_results = []
    for atoms in atoms_list:
        dl = build_dataloader(
            positions_list=[atoms.get_positions()],
            cell_list=[atoms.get_cell()],
            pbc_list=[atoms.get_pbc()],
            atomic_numbers_list=[atoms.get_atomic_numbers()],
            cutoff=cutoff,
            threebody_cutoff=threebody_cutoff,
            batch_size=1,
        )
        graph = next(iter(dl))
        result = aot_model(*unpack_graph_batch(graph))
        individual_results.append(result)

    # Run as a single batch of 2 structures with different shapes
    dl_batch = build_dataloader(
        positions_list=[a.get_positions() for a in atoms_list],
        cell_list=[a.get_cell() for a in atoms_list],
        pbc_list=[a.get_pbc() for a in atoms_list],
        atomic_numbers_list=[a.get_atomic_numbers() for a in atoms_list],
        cutoff=cutoff,
        threebody_cutoff=threebody_cutoff,
        batch_size=len(atoms_list),
    )
    batch_graph = next(iter(dl_batch))
    batch_result = aot_model(*unpack_graph_batch(batch_graph))

    # Compare per-structure outputs: batched should match individual
    atom_offset = 0
    for i, (atoms, ind) in enumerate(zip(atoms_list, individual_results, strict=True)):
        n_atoms_i = len(atoms)

        assert np.allclose(
            batch_result["energy"][i].detach().numpy(),
            ind["energy"][0].detach().numpy(),
            atol=1e-4,
        ), f"Energy mismatch for structure {i}"

        assert np.allclose(
            batch_result["forces"][atom_offset : atom_offset + n_atoms_i].detach().numpy(),
            ind["forces"].detach().numpy(),
            atol=1e-4,
        ), f"Forces mismatch for structure {i}"

        assert np.allclose(
            batch_result["stress"][i].detach().numpy(),
            ind["stress"][0].detach().numpy(),
            atol=1e-4,
        ), f"Stress mismatch for structure {i}"

        atom_offset += n_atoms_i
