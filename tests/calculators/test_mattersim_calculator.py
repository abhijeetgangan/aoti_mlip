import os
from pathlib import Path

import pytest
import torch

from aoti_mlip.utils.aoti_compile import compile_mattersim
from aoti_mlip.utils.batch_info import get_example_inputs


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
    from ase.build import bulk

    from aoti_mlip.calculators.mattersim import MatterSimCalculator

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
    example_inputs = get_example_inputs(device=torch.device("cpu"))
    out = aot_model(*example_inputs)
    energy_aot = float(out["energy"].detach().cpu().numpy()[0])

    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.calc = MatterSimCalculator(model_path=pkg_path, device="cpu")
    energy_ase = float(atoms.get_potential_energy())

    assert pytest.approx(energy_aot, rel=0, abs=1e-3) == energy_ase
