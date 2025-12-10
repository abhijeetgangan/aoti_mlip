# Third-party notice
# Portions of this module and its transitive dependencies are adapted from
# microsoft/mattersim (https://github.com/microsoft/mattersim).
# See third_party/mattersim/LICENSE for the original license terms.

"""M3GNet model wrappers"""

from typing import Any

import torch
from torch import nn

from aoti_mlip.models.mattersim_modules.m3gnet import M3Gnet


class M3GnetEnergyModel(nn.Module):
    """A simplified wrapper that returns only energy (no forces or stress)."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        allow_tf32: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the energy-only model.

        Args:
            model: Dictionary containing pretrained model weights and config with keys:
                - ``"model"``: State dict for M3GNet model
                - ``"model_args"``: Dictionary of model architecture arguments
            device: Device to run on, e.g. ``"cuda"``, ``"cuda:0"``, or ``"cpu"``.
            allow_tf32: Enable TF32 matmul on CUDA backends for speed at slight precision loss.
            **kwargs: Ignored; present for forward compatibility.
        """
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

        self.model = M3Gnet(device=device, **model["model_args"]).to(device)  # type: ignore
        self.model.load_state_dict(model["model"], strict=False)  # type: ignore
        self.model.eval()
        self.device = device  # type: ignore
        self.to(device)

    def forward(
        self,
        atom_pos: torch.Tensor,
        cell: torch.Tensor,
        pbc_offsets: torch.Tensor,
        atom_attr: torch.Tensor,
        edge_index: torch.Tensor,
        three_body_indices: torch.Tensor,
        num_three_body: torch.Tensor,
        num_bonds: torch.Tensor,
        num_triple_ij: torch.Tensor,
        num_atoms: torch.Tensor,
        num_graphs: torch.Tensor,
        batch: torch.Tensor,
        dataset_idx: int = -1,
    ) -> torch.Tensor:
        """Forward pass returning only energy.

        Args:
            atom_pos: Tensor [N, 3] - atomic positions
            cell: Tensor [1, 3, 3] - cell vectors
            pbc_offsets: Tensor [E, 3] - periodic boundary condition offsets
            atom_attr: Tensor [N, 1] - atomic attributes
            edge_index: LongTensor [2, E] - edge connectivity
            three_body_indices: LongTensor [T, 2] - three-body interaction indices
            num_three_body: Tensor [1] - number of three-body interactions
            num_bonds: Tensor [1] - number of bonds
            num_triple_ij: Tensor [E, 1] - number of triplets per edge
            num_atoms: Tensor [1] - number of atoms
            num_graphs: Scalar tensor - number of graphs in batch
            batch: LongTensor [N] - batch assignment for each atom
            dataset_idx: Optional dataset selector for multi-head models; ``-1`` uses default.

        Returns:
            Tensor: Energy values (not wrapped in a dict).
        """
        energies = self.model.forward(
            atom_pos,
            cell,
            pbc_offsets,
            atom_attr,
            edge_index,
            three_body_indices,
            num_three_body,
            num_bonds,
            num_triple_ij,
            num_atoms,
            num_graphs,
            batch,
            dataset_idx,
        )
        return energies


class M3GnetModel(nn.Module):
    """A wrapper class for the force field model."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        allow_tf32: bool = False,
        compute_force: bool = False,
        compute_stress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the potential.

        Args:
            model: Dictionary containing pretrained model weights and config with keys:
                - ``"model"``: State dict for M3GNet model
                - ``"model_args"``: Dictionary of model architecture arguments
            device: Device to run on, e.g. ``"cuda"``, ``"cuda:0"``, or ``"cpu"``.
            allow_tf32: Enable TF32 matmul on CUDA backends for speed at slight precision loss.
            compute_force: If True, compute forces via autograd during forward.
            compute_stress: If True, compute stress via autograd during forward.
            **kwargs: Ignored; present for forward compatibility.
        """
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

        self.compute_force = compute_force  # type: ignore
        self.compute_stress = compute_stress  # type: ignore

        self.model = M3Gnet(device=device, **model["model_args"]).to(device)  # type: ignore
        self.model.load_state_dict(model["model"], strict=False)  # type: ignore
        self.model.eval()
        self.device = device  # type: ignore
        self.to(device)

    def forward(
        self,
        atom_pos: torch.Tensor,
        cell: torch.Tensor,
        pbc_offsets: torch.Tensor,
        atom_attr: torch.Tensor,
        edge_index: torch.Tensor,
        three_body_indices: torch.Tensor,
        num_three_body: torch.Tensor,
        num_bonds: torch.Tensor,
        num_triple_ij: torch.Tensor,
        num_atoms: torch.Tensor,
        num_graphs: torch.Tensor,
        batch: torch.Tensor,
        dataset_idx: int = -1,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with individual tensor arguments.

        Args:
            atom_pos: Tensor [N, 3] - atomic positions
            cell: Tensor [1, 3, 3] - cell vectors
            pbc_offsets: Tensor [E, 3] - periodic boundary condition offsets
            atom_attr: Tensor [N, 1] - atomic attributes
            edge_index: LongTensor [2, E] - edge connectivity
            three_body_indices: LongTensor [T, 2] - three-body interaction indices
            num_three_body: Tensor [1] - number of three-body interactions
            num_bonds: Tensor [1] - number of bonds
            num_triple_ij: Tensor [E, 1] - number of triplets per edge
            num_atoms: Tensor [1] - number of atoms
            num_graphs: Scalar tensor - number of graphs in batch
            batch: LongTensor [N] - batch assignment for each atom
            dataset_idx: Optional dataset selector for multi-head models; ``-1`` uses default.

        Returns:
            Dict with at least ``{"energy": Tensor[...]}``. If ``compute_force`` was set
            during initialization, includes ``"forces"``. If ``compute_stress`` was set,
            includes ``"stress"``. Returned tensors are detached from autograd.
        """
        # Initialize strain and volume
        strain = torch.zeros_like(cell, device=self.device)
        volume = torch.linalg.det(cell)

        results = {}
        if self.compute_force:
            atom_pos.requires_grad_(True)
        if self.compute_stress:
            strain.requires_grad_(True)
            cell = torch.matmul(
                cell,
                (torch.eye(3, device=self.device)[None, ...] + strain),
            )
            strain_augment = torch.repeat_interleave(strain, num_atoms, dim=0)
            atom_pos = torch.einsum(
                "bi, bij -> bj",
                atom_pos,
                (torch.eye(3, device=self.device)[None, ...] + strain_augment),
            )
            volume = torch.linalg.det(cell)

        # Call model with tuple arguments
        energies = self.model.forward(
            atom_pos,
            cell,
            pbc_offsets,
            atom_attr,
            edge_index,
            three_body_indices,
            num_three_body,
            num_bonds,
            num_triple_ij,
            num_atoms,
            num_graphs,
            batch,
            dataset_idx,
        )
        results["energy"] = energies.detach()

        # Only take first derivative if only force is required
        if self.compute_force and not self.compute_stress:
            grad_outputs: list[torch.Tensor | None] = [
                torch.ones_like(
                    energies,
                )
            ]
            grad = torch.autograd.grad(
                outputs=[
                    energies,
                ],
                inputs=[atom_pos],
                grad_outputs=grad_outputs,  # type: ignore
                create_graph=self.model.training,
            )

            # Dump out gradient for forces
            force_grad = grad[0]
            if force_grad is not None:
                forces = torch.neg(force_grad)
                results["forces"] = forces.detach()

        if self.compute_force and self.compute_stress:
            grad_outputs: list[torch.Tensor | None] = [
                torch.ones_like(
                    energies,
                )
            ]
            grad = torch.autograd.grad(
                outputs=[
                    energies,
                ],
                inputs=[atom_pos, strain],
                grad_outputs=grad_outputs,  # type: ignore
                create_graph=self.model.training,
            )

            # Dump out gradient for forces and stresses
            force_grad = grad[0]
            stress_grad = grad[1]

            if force_grad is not None:
                forces = torch.neg(force_grad)
                results["forces"] = forces.detach()

            if stress_grad is not None:
                stresses = 1 / volume[:, None, None] * stress_grad
                results["stress"] = stresses.detach()

        return results
