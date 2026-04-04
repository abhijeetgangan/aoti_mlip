"""TorchSim ModelInterface wrapper for an AOTInductor-compiled MatterSim model."""

from __future__ import annotations

import traceback
import warnings
from typing import Any

import torch

from aoti_mlip.models.mattersim_modules.dataloader.build import build_dataloader, unpack_graph_batch

try:
    import torch_sim as ts  # ty: ignore[unresolved-import]
    from torch_sim.models.interface import ModelInterface  # ty: ignore[unresolved-import]
    from torch_sim.state import SimState  # ty: ignore[unresolved-import]
except ImportError:
    warnings.warn(f"torch-sim import failed: {traceback.format_exc()}", stacklevel=2)

    class ModelInterface(torch.nn.Module):  # type: ignore[no-redef]
        """Placeholder when torch-sim is not installed."""

        pass

    class SimState:  # type: ignore[no-redef]
        """Placeholder when torch-sim is not installed."""

        device: torch.device

        def to(self, device: torch.device) -> SimState:  # noqa: ARG002
            return self


class MatterSimTorchSimModel(ModelInterface):
    """TorchSim model backed by an AOT-compiled MatterSim ``.pt2`` package.

    Implements the ``ModelInterface`` from torch-sim, providing batched
    energy, force, and stress predictions for use with TorchSim integrators.

    Examples:
        >>> model = MatterSimTorchSimModel(model_path="mattersim.pt2")
        >>> output = model(sim_state)
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the model from a compiled ``.pt2`` package.

        Args:
            model_path: Filesystem path to the AOTInductor ``.pt2`` model package.
            device: Device for inference.  Defaults to CUDA if available.
            dtype: Floating-point dtype.  Defaults to ``torch.float32``.
        """
        super().__init__()

        if device is None:
            resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            resolved_device = torch.device(device)
        else:
            resolved_device = device

        self._device = resolved_device
        self._dtype = dtype or torch.float32
        self._compute_stress = True
        self._compute_forces = True
        self._memory_scales_with = "n_atoms_x_density"

        self.model = torch._inductor.aoti_load_package(model_path)
        metadata = self.model.get_metadata()
        self.cutoff = float(metadata["cutoff"])
        self.threebody_cutoff = float(metadata["threebody_cutoff"])

    def forward(self, state: SimState, **_kwargs: Any) -> dict[str, torch.Tensor]:
        """Compute energies, forces, and stresses for a batch of structures.

        Args:
            state: TorchSim ``SimState`` containing positions, cells, atomic
                numbers, and batch indexing.
            **_kwargs: Unused; accepted for interface compatibility.

        Returns:
            Dictionary with:
            - ``energy``: ``[n_systems]``
            - ``forces``: ``[n_atoms, 3]``
            - ``stress``: ``[n_systems, 3, 3]`` (if ``compute_stress`` is True)
        """
        if state.device != self._device:
            state = state.to(self._device)

        atoms_list = ts.io.state_to_atoms(state)

        dataloader = build_dataloader(
            positions_list=[a.get_positions() for a in atoms_list],
            cell_list=[a.get_cell() for a in atoms_list],
            pbc_list=[a.get_pbc() for a in atoms_list],
            atomic_numbers_list=[a.get_atomic_numbers() for a in atoms_list],
            cutoff=self.cutoff,
            threebody_cutoff=self.threebody_cutoff,
            batch_size=len(atoms_list),
        )
        graph_batch = next(iter(dataloader)).to(self._device)
        output = self.model(*unpack_graph_batch(graph_batch))

        results: dict[str, torch.Tensor] = {}
        results["energy"] = output["energy"].detach()
        results["forces"] = output["forces"].detach()
        if self._compute_stress:
            results["stress"] = output["stress"].detach()

        return results
