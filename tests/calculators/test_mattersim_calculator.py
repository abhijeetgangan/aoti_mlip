import os
from pathlib import Path

import pytest
from ase.build import bulk

from aoti_mlip.calculators.mattersim import MatterSimCalculator as aoti_MatterSimCalculator
from aoti_mlip.utils.aoti_compile import compile_mattersim

try:
    from mattersim.forcefield.potential import MatterSimCalculator  # type: ignore[attr-defined]
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


def test_aot_matches_ase_energy():
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

    atoms_1 = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms_1.calc = aoti_MatterSimCalculator(model_path=pkg_path, device="cpu")
    energy = float(atoms_1.get_potential_energy())

    atoms_2 = atoms_1.copy()
    atoms_2.calc = MatterSimCalculator(device="cpu", compute_stress=True, compute_force=True)
    energy_mattersim = float(atoms_2.get_potential_energy())
    assert pytest.approx(energy, rel=0, abs=1e-4) == energy_mattersim
