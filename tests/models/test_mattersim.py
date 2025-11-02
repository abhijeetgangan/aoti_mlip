import os
from pathlib import Path

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


def test_compile_mattersim_cpu():
    checkpoint = "mattersim-v1.0.0-1M.pth"
    _ensure_checkpoint_available(checkpoint)

    pkg_path = compile_mattersim(
        checkpoint_name=checkpoint,
        cutoff=5.0,
        threebody_cutoff=4.0,
        compute_force=False,
        compute_stress=False,
        device="cpu",
    )
    assert os.path.exists(pkg_path)

    aot_model = torch._inductor.aoti_load_package(pkg_path)
    example_inputs = get_example_inputs(device=torch.device("cpu"))
    out = aot_model(*example_inputs)
    assert isinstance(out, dict) and "energy" in out
