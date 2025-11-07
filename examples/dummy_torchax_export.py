import pickle

import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
import torchax
from jax import export
from jax.experimental.serialize_executable import deserialize_and_load, serialize


# 1. Define your PyTorch model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x1, x2):
        x1 = torch.relu(self.fc1(x1))
        x2 = torch.relu(self.fc1(x2))
        x2 = self.fc2(x1 + x2)
        return x1 + x2


# Create ShapeDtypeStruct for each parameter
model = MyModel()
params, jax_model = torchax.extract_jax(model)
param_signatures = {
    name: jax.ShapeDtypeStruct(param.shape, param.dtype) for name, param in params.items()
}
input_data = (jnp.ones((1, 28 * 28)), jnp.ones((1, 28 * 28)))
sig_b = jax.ShapeDtypeStruct(input_data[0].shape, input_data[0].dtype)
sig_c = jax.ShapeDtypeStruct(input_data[1].shape, input_data[1].dtype)
lowered = jax.jit(jax_model).trace(param_signatures, input_data).lower()

compiled = lowered.compile()

print(lowered.cost_analysis())

"""Portable export"""

exported = export.export(jax.jit(jax_model))(param_signatures, input_data)
blob = exported.serialize()

with open("f_export_jax", "wb") as f_out:
    f_out.write(blob)

with open("f_export_jax", "rb") as f_in:
    imported_blob = f_in.read()

imported = export.deserialize(imported_blob)

print(imported.call(params, input_data))

"""Binary export"""

lowered_cpu = jax.jit(jax_model).trace(param_signatures, (sig_b, sig_c)).lower()

print(lowered_cpu.cost_analysis())

compiled_cpu = lowered_cpu.compile()

serialized_model, in_tree, out_tree = serialize(compiled_cpu)

with open("f_exec.jax", "wb") as fb:
    fb.write(serialized_model)

with open("f_exec_trees.pkl", "wb") as fb:
    pickle.dump((in_tree, out_tree), fb)


with open("f_exec.jax", "rb") as fb:
    loaded_serialized_model = fb.read()

with open("f_exec_trees.pkl", "rb") as fb:
    in_tree, out_tree = pickle.load(fb)

compiled_cpu_loaded = deserialize_and_load(loaded_serialized_model, in_tree, out_tree)

print(compiled_cpu_loaded(params, input_data))
