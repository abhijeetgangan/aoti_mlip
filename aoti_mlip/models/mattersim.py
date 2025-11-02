"""M3GNet model wrappers and forward utilities."""

from typing import Any

import torch
from torch import nn

from aoti_mlip.models.mattersim_modules.m3gnet import M3Gnet


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
            model: State dict and configuration for the pretrained M3GNet model.
                Expected keys include ``"model_args"`` (constructor kwargs) and
                ``"model"`` (state dict).
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
        input_dict: dict[str, torch.Tensor],
        dataset_idx: int = -1,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_dict: Model inputs as a dictionary. Typical keys include
                ``atom_pos`` [N, 3], ``cell`` [1, 3, 3], ``pbc_offsets`` [E, 3],
                ``atom_attr`` [N, 1], ``edge_index`` [2, E], ``three_body_indices`` [T, 2],
                and associated counters/indices used by the model.
            dataset_idx: Optional dataset selector for multi-head models; ``-1`` uses default.

        Returns:
            Dict with at least ``{"energy": Tensor[...]}``. If ``compute_force`` was set
            during initialization, includes ``"forces"``. If ``compute_stress`` was set,
            includes ``"stress"``. Returned tensors are detached from autograd.
        """
        # Move input tensors to device
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(self.device)

        # Initialize strain and volume
        strain = torch.zeros_like(input_dict["cell"], device=self.device)
        volume = torch.linalg.det(input_dict["cell"])

        results = {}
        if self.compute_force:
            input_dict["atom_pos"].requires_grad_(True)
        if self.compute_stress:
            strain.requires_grad_(True)
            input_dict["cell"] = torch.matmul(
                input_dict["cell"],
                (torch.eye(3, device=self.device)[None, ...] + strain),
            )
            strain_augment = torch.repeat_interleave(strain, input_dict["num_atoms"], dim=0)
            input_dict["atom_pos"] = torch.einsum(
                "bi, bij -> bj",
                input_dict["atom_pos"],
                (torch.eye(3, device=self.device)[None, ...] + strain_augment),
            )
            volume = torch.linalg.det(input_dict["cell"])

        energies = self.model.forward(input_dict, dataset_idx)
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
                inputs=[input_dict["atom_pos"]],
                grad_outputs=grad_outputs,
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
                inputs=[input_dict["atom_pos"], strain],
                grad_outputs=grad_outputs,
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


class M3GnetWrapper(torch.nn.Module):
    """Wrap the M3GNet model to accept flat tensor arguments instead of a dict."""

    def __init__(self, m3gnet_model):
        """Initialize with an underlying M3GNet model instance.

        Args:
            m3gnet_model: An instance of :class:`M3GnetModel` (or compatible) that
                accepts an input dictionary and returns a results dictionary.
        """
        super().__init__()
        self.model = m3gnet_model

    def forward(
        self,
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
    ):
        """Forward pass with individual tensor arguments.

        Args:
            atom_pos: Tensor [N, 3]
            cell: Tensor [1, 3, 3]
            pbc_offsets: Tensor [E, 3]
            atom_attr: Tensor [N, 1]
            edge_index: LongTensor [2, E]
            three_body_indices: LongTensor [T, 2]
            num_three_body: Tensor [1]
            num_bonds: Tensor [1]
            num_triple_ij: Tensor [E, 1]
            num_atoms: Tensor [1]
            num_graphs: Scalar tensor or 0-D tensor
            batch: LongTensor [N]

        Returns:
            Dict with energy and optionally forces/stress, matching ``M3GnetModel``.
        """
        input_dict = {
            "atom_pos": atom_pos,
            "cell": cell,
            "pbc_offsets": pbc_offsets,
            "atom_attr": atom_attr,
            "edge_index": edge_index,
            "three_body_indices": three_body_indices,
            "num_three_body": num_three_body,
            "num_bonds": num_bonds,
            "num_triple_ij": num_triple_ij,
            "num_atoms": num_atoms,
            "num_graphs": num_graphs,
            "batch": batch,
        }

        return self.model(input_dict)
