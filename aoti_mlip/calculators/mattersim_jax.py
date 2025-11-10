"""ASE Calculator wrapper around an AOTInductor-compiled MatterSim model."""

from functools import partial

import jax
from jax import config

config.update("jax_enable_x64", True)
import time

import jax.numpy as jnp
import numpy as np
import torch
import torchax
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import full_3x3_to_voigt_6_stress

from aoti_mlip.models.mattersim import M3GnetEnergyModel
from aoti_mlip.models.mattersim_modules.dataloader.build import batch_to_dict, build_dataloader
from aoti_mlip.utils.aoti_tools import model_make_fx
from aoti_mlip.utils.batch_info import get_example_inputs


class MatterSimJaxCalculator(Calculator):
    """ASE Calculator backed by an AOT-compiled MatterSim Jax model.

    Implements ``energy``, ``free_energy``, ``forces``, and ``stress``.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        """Initialize the calculator.

        Args:
            model_path: Filesystem path to the packaged ``.pt2`` model.
            device: Device string to run inference on (e.g., ``"cpu"``, ``"cuda"``).
            **kwargs: Additional ``ase.calculators.calculator.Calculator`` kwargs.
        """
        super().__init__(**kwargs)
        self.model_path = model_path
        self.jax_device = next((d for d in jax.devices() if d.platform == "gpu"), jax.devices()[0])
        self.cutoff = 5.0
        self.compile_backend = "cpu"
        self.threebody_cutoff = 4.0
        self.model = self.compile_jax_model(model_path)

    def calculate(
        self,
        atoms: Atoms,
        properties: list | None = None,
        system_changes: list | None = None,
    ):
        """Run a calculation and populate ``self.results``.

        Args:
            atoms: The input ``ase.Atoms`` structure.
            properties: Properties to compute; defaults to ``["energy"]``.
            system_changes: List of changes that trigger recalculation; defaults
                to standard ASE triggers.

        Returns:
            None. Results are stored in ``self.results`` with keys:
            ``energy``, ``free_energy``, ``forces``, and ``stress``.
        """

        all_changes = [
            "positions",
            "numbers",
            "cell",
            "pbc",
            "initial_charges",
            "initial_magmoms",
        ]

        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        batch_time_start = time.perf_counter()
        dataloader = build_dataloader(
            positions_list=[atoms.get_positions()],
            cell_list=[atoms.get_cell()],
            pbc_list=[atoms.get_pbc()],
            atomic_numbers_list=[atoms.get_atomic_numbers()],
            cutoff=self.cutoff,
            threebody_cutoff=self.threebody_cutoff,
            batch_size=1,
        )
        graph_batch = next(iter(dataloader))
        input_dict = {
            k: v.to(self.compile_backend) if isinstance(v, torch.Tensor) else v
            for k, v in batch_to_dict(graph_batch).items()
        }
        input_tuples_jax = ()
        for value in input_dict.values():
            input_tuples_jax += (jnp.array(value.to(self.compile_backend)),)
        input_tuples_jax = tuple(jax.device_put(x, self.jax_device) for x in input_tuples_jax)

        # batch_const = tuple(jax.lax.stop_gradient(x) for x in input_tuples_jax[2:])
        self.batch_const = tuple(x for x in input_tuples_jax[2:])
        batch_time_end = time.perf_counter()
        batch_time = batch_time_end - batch_time_start
        print(f"Batch time: {batch_time * 1000:.2f} ms")

        model_call_time_start = time.perf_counter()
        Energy, Force, Stress = jax.jit(self.energy_force_stress_fn)(
            input_tuples_jax[0], input_tuples_jax[1]
        )
        model_call_time_end = time.perf_counter()
        model_call_time = model_call_time_end - model_call_time_start
        print(f"Model call time: {model_call_time * 1000:.2f} ms")
        self.results.update(
            energy=np.array(Energy),
            free_energy=np.array(Energy),
            forces=np.array(Force),
            stress=full_3x3_to_voigt_6_stress(np.array(Stress)),
        )

    def compile_jax_model(self, CKPT_PATH: str):
        model_to_compile = M3GnetEnergyModel(
            model=torch.load(CKPT_PATH, map_location=self.compile_backend, weights_only=True),
            device=str(self.compile_backend),
        )
        example_inputs = get_example_inputs(
            cutoff=self.cutoff,
            threebody_cutoff=self.threebody_cutoff,
            device=torch.device(self.compile_backend),
        )
        fx_model = model_make_fx(model_to_compile, example_inputs)
        params, jax_model = torchax.extract_jax(fx_model)
        model = partial(jax_model, params)
        return model

    def energy_displacement_fn(self, positions, strain, cell):
        A = jnp.eye(3, dtype=strain.dtype) + strain
        positions = positions @ A
        cell = jnp.matmul(cell, A)
        inputs = (positions, cell) + self.batch_const
        return self.model(inputs)

    def energy_force_stress_fn(self, positions, cell):
        strain = jnp.zeros((3, 3), dtype=cell.dtype)
        output, grads = jax.value_and_grad(self.energy_displacement_fn, argnums=(0, 1))(
            positions, strain, cell
        )
        return output, jnp.negative(grads[0]), grads[1] / jnp.linalg.det(cell)
