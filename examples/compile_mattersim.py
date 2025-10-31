import os

import torch
from ase.build import bulk

from aoti_mlip.models.mattersim import M3GnetModel, M3GnetWrapper
from aoti_mlip.models.mattersim_modules.dataloader.build import batch_to_dict, build_dataloader
from aoti_mlip.utils.aoti_tools import model_make_fx, prepare_model_for_compile
from aoti_mlip.utils.batch_info import (
    batch_to_tuples,
    get_example_inputs,
    matter_sim_dynamic_shapes,
)

os.environ["AOTI_RUNTIME_CHECK_INPUTS"] = "1"
os.makedirs("model_checkpoints", exist_ok=True)
MODEL = "1M"
CKPT_PATH = os.path.expanduser(f"~/.local/mattersim/pretrained_models/mattersim-v1.0.0-{MODEL}.pth")
print(f"Loading mattersim model from: {CKPT_PATH}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUTOFF = 5.0
THREEBODY_CUTOFF = 4.0
PACKAGE_PATH = os.path.join(os.getcwd(), f"model_checkpoints/mattersim_{DEVICE}_{MODEL}.pt2")

example_inputs = get_example_inputs(cutoff=CUTOFF, threebody_cutoff=THREEBODY_CUTOFF, device=DEVICE)
model_to_compile = M3GnetModel(
    model=torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True),
    device=str(DEVICE),
    compute_force=True,
    compute_stress=True,
)
wrapped_model = M3GnetWrapper(model_to_compile)
model_to_compile = prepare_model_for_compile(wrapped_model, DEVICE)

# Quick sanity check
print("Testing model_to_compile...")
energy_result = model_to_compile(*example_inputs)
print(f"Reference energy: {energy_result['energy'].item():.4f} eV\n")
print("Exporting model...")
fx_model = model_make_fx(model_to_compile, example_inputs)


exported_model = torch.export.export(
    fx_model,
    example_inputs,
    dynamic_shapes=matter_sim_dynamic_shapes,
)
_AOT_METADATA_KEY = "aot_inductor.metadata"
inductor_cfg = {
    _AOT_METADATA_KEY: {"cutoff": str(CUTOFF), "threebody_cutoff": str(THREEBODY_CUTOFF)},
}


torch._inductor.aoti_compile_and_package(
    exported_model,
    package_path=PACKAGE_PATH,
    inductor_configs=inductor_cfg,
)

print(f"AOT-compiled model saved to: {PACKAGE_PATH}\n")

MODEL_PATH = PACKAGE_PATH
print(f"Loading AOT-compiled dynamic model from: {MODEL_PATH}")
aot_model = torch._inductor.aoti_load_package(MODEL_PATH)
metadata = aot_model.get_metadata()
print(f"Metadata: {metadata}\n")

si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
positions_list = [si_dc.get_positions()]
cell_list = [si_dc.get_cell()]
pbc_list = [si_dc.get_pbc()]
atomic_numbers_list = [si_dc.get_atomic_numbers()]

# Build input
dataloader = build_dataloader(
    positions_list=positions_list,
    cell_list=cell_list,
    pbc_list=pbc_list,
    atomic_numbers_list=atomic_numbers_list,
    cutoff=float(metadata["cutoff"]),
    threebody_cutoff=float(metadata["threebody_cutoff"]),
    batch_size=1,
)

graph_batch = next(iter(dataloader))
example_dict = {
    k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
    for k, v in batch_to_dict(graph_batch).items()
}
example_inputs = batch_to_tuples(example_dict)
results = aot_model(*example_inputs)
print(f"AOT-compiled results: {results}")
print(f"AOT-compiled energy: {results['energy'].item():.4f} eV\n")
print(f"AOT-compiled forces: {results['forces'].shape}")
print(f"AOT-compiled stress: {results['stress'].shape}")
