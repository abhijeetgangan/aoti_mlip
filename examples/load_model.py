import torch
from ase.build import bulk

from aoti_mlip.models.mattersim_modules.dataloader.build import batch_to_dict, build_dataloader

# os.environ["AOTI_RUNTIME_CHECK_INPUTS"] = "1"
from aoti_mlip.utils.aoti_compile import compile_mattersim
from aoti_mlip.utils.batch_info import (
    batch_to_tuples,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PACKAGE_PATH = compile_mattersim("mattersim-v1.0.0-5M.pth", device=DEVICE)
model = torch._inductor.aoti_load_package(PACKAGE_PATH)
metadata = model.get_metadata()
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
    k: v.to(torch.device(DEVICE)) if isinstance(v, torch.Tensor) else v
    for k, v in batch_to_dict(graph_batch).items()
}
example_inputs = batch_to_tuples(example_dict)
results = model(*example_inputs)
print(f"AOT-compiled results: {results}")
print(f"AOT-compiled energy: {results['energy'].item():.4f} eV\n")
print(f"AOT-compiled forces: {results['forces'].shape}")
print(f"AOT-compiled stress: {results['stress'].shape}")
