import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import torch
import torchax
from jax import config
from jax._src.basearray import Array
from torchax.interop import jax_jit  # noqa: F401

from aoti_mlip.models.mattersim import M3GnetEnergyModel, M3GnetModel
from aoti_mlip.utils.aoti_tools import model_make_fx, prepare_model_for_compile
from aoti_mlip.utils.batch_info import get_example_inputs

config.update("jax_enable_x64", True)
BASE_PATH = "~/.local/mattersim/pretrained_models/"
CKPT_PATH = os.path.expanduser(f"{BASE_PATH}/mattersim-v1.0.0-1M.pth")
cutoff = 5.0
threebody_cutoff = 4.0
device = "cpu"

example_inputs = get_example_inputs(
    cutoff=cutoff, threebody_cutoff=threebody_cutoff, device=torch.device(device)
)
NCALLS = 20
batch_dict = {
    "atom_pos": jnp.array(example_inputs[0].to(device)),
    "cell": jnp.array(example_inputs[1].to(device)),
    "pbc_offsets": jnp.array(example_inputs[2].to(device)),
    "atom_attr": jnp.array(example_inputs[3].to(device)),
    "edge_index": jnp.array(example_inputs[4].to(device)),
    "three_body_indices": jnp.array(example_inputs[5].to(device)),
    "num_three_body": jnp.array(example_inputs[6].to(device)),
    "num_bonds": jnp.array(example_inputs[7].to(device)),
    "num_triple_ij": jnp.array(example_inputs[8].to(device)),
    "num_atoms": jnp.array(example_inputs[9].to(device)),
    "num_graphs": jnp.array(example_inputs[10].to(device)),
    "batch": jnp.array(example_inputs[11].to(device)),
}
batch_tuples = ()
for key, value in batch_dict.items():
    batch_tuples += (value,)
    print(key, value.shape, value.dtype)

model_to_compile = M3GnetEnergyModel(
    model=torch.load(CKPT_PATH, map_location=device, weights_only=True),
    device=str(device),
)
torch_energy = model_to_compile(*example_inputs)
print("Torch energy:", torch_energy)
model_to_compile = prepare_model_for_compile(model_to_compile, torch.device(device))
fx_model = model_make_fx(model_to_compile, example_inputs)
# 2. Extract JAX function and parameters
# params, jax_model = torchax.extract_jax(fx_model)
params, jax_model = torchax.extract_jax(fx_model)
print(len(params))
print(jax_model)
param_signatures = {
    name: jax.ShapeDtypeStruct(param.shape, param.dtype) for name, param in params.items()
}
# TODO: solve the tracing issue
# lowered = jax.jit(jax_model).trace(param_signatures, batch_tuples).lower()
# compiled = lowered.compile()
# print(lowered.cost_analysis())

model_fn = partial(jax_model, params)
output = model_fn(batch_tuples)

time_start = time.perf_counter()
for _ in range(NCALLS):
    output = model_fn(batch_tuples)
print("Raw output:", output)
time_end = time.perf_counter()
time_without_jit = (time_end - time_start) / NCALLS

# Does it only on the whole batch?
# grads = jax.grad(model_fn, argnums=(0,), allow_int=True)(batch_tuples)
# print(grads)
batch_const = tuple[Array, ...](
    jax.lax.stop_gradient(x) for x in batch_tuples[2:]
)  # exclude cell (idx 1)


@jax.jit
def energy_displacement_fn(positions, strain):
    A = jnp.eye(3, dtype=strain.dtype) + strain
    positions = positions @ A
    cell = jnp.matmul(batch_tuples[1], A)
    inputs = (positions, cell) + batch_const
    return model_fn(inputs)


@jax.jit
def energy_force_stress_fn(positions, cell):
    strain = jnp.zeros((3, 3), dtype=cell.dtype)
    output, grads = jax.value_and_grad(energy_displacement_fn, argnums=(0, 1))(positions, strain)
    return output, jnp.negative(grads[0]), grads[1] / jnp.linalg.det(cell)


print(energy_force_stress_fn(batch_tuples[0], batch_tuples[1]))

jit_time_start = time.perf_counter()
for _ in range(NCALLS):
    energy = energy_force_stress_fn(batch_tuples[0], batch_tuples[1])
jit_time_end = time.perf_counter()
jit_time_without_jit = (jit_time_end - jit_time_start) / NCALLS
print("Without JIT time:", time_without_jit, "s")
print("JIT time:", jit_time_without_jit, "s")
print("Speedup:", time_without_jit / jit_time_without_jit)

assert torch.allclose(torch_energy, torch.tensor(energy[0], device=device), atol=1e-5), (
    "Energy mismatch"
)
force_jit_time_start = time.perf_counter()
for _ in range(NCALLS):
    energy, force, stress = energy_force_stress_fn(batch_tuples[0], batch_tuples[1])
force_jit_time_end = time.perf_counter()
force_jit_time = (force_jit_time_end - force_jit_time_start) / NCALLS
print("Force JIT time:", force_jit_time, "s")

atom_pos_signature = jax.ShapeDtypeStruct(batch_tuples[0].shape, batch_tuples[0].dtype)
cell_signature = jax.ShapeDtypeStruct(batch_tuples[1].shape, batch_tuples[1].dtype)
lowered = jax.jit(energy_force_stress_fn).trace(atom_pos_signature, cell_signature).lower()
compiled = lowered.compile()
# print(lowered.cost_analysis())


model_to_compile = M3GnetModel(
    model=torch.load(CKPT_PATH, map_location=device, weights_only=True),
    device=str(device),
    compute_force=True,
    compute_stress=True,
)
model_to_compile = prepare_model_for_compile(model_to_compile, torch.device(device))
fx_model = model_make_fx(model_to_compile, example_inputs)

print(fx_model(*example_inputs))
"""JIT with JaxJit

from torchax.interop import jax_jit

model_jitted = jax_jit(model_fn)

print(model_jitted(batch_tuples))

@jax.jit
def energy_fn(atom_pos, rest):
    # prevent AD through the rest of the batch
    rest = tree_map(jax.lax.stop_gradient, rest)
    return model_jitted((atom_pos,) + rest)

# value and grad w.r.t. atom_pos only
force_fn = jax.jit(jax.value_and_grad(lambda a, r: jnp.sum(energy_fn(a, r))))

# First call compiles; subsequent calls with same shapes just run
energy, force = force_fn(batch_tuples[0], batch_tuples[1:])

"""
