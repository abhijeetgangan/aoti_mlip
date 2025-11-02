"""Dynamic shape specs and small helpers for MatterSim example batches.

This module defines dynamic shape constraints used during ``torch.export`` and
provides utilities to construct a minimal example batch and convert it to the
tuple form expected by the compiled model.
"""

import torch
from ase.build import bulk

from aoti_mlip.models.mattersim_modules.dataloader.build import batch_to_dict, build_dataloader

NODE_DIM = torch.export.dynamic_shapes.Dim("num_atoms", min=1, max=torch.inf)  # type: ignore
PBC_OFFSET_DIM = torch.export.dynamic_shapes.Dim("num_edges", min=1, max=torch.inf)  # type: ignore
THREE_BODY_DIM = torch.export.dynamic_shapes.Dim("num_three_body", min=1, max=torch.inf)  # type: ignore

# Dynamic shapes for each argument
MATTERSIM_DYNAMIC_SHAPES = (
    {0: NODE_DIM, 1: torch.export.Dim.STATIC},  # atom_pos [num_atoms, 3]
    {
        0: torch.export.Dim.STATIC,
        1: torch.export.Dim.STATIC,
        2: torch.export.Dim.STATIC,
    },  # cell [1, 3, 3]
    {0: PBC_OFFSET_DIM, 1: torch.export.Dim.STATIC},  # pbc_offsets [num_edges, 3]
    {0: NODE_DIM, 1: torch.export.Dim.STATIC},  # atom_attr [num_atoms, 1]
    {0: torch.export.Dim.STATIC, 1: PBC_OFFSET_DIM},  # edge_index [2, num_edges]
    {0: THREE_BODY_DIM, 1: torch.export.Dim.STATIC},  # three_body_indices [num_three_body, 2]
    {0: torch.export.Dim.STATIC},  # num_three_body [1]
    {0: torch.export.Dim.STATIC},  # num_bonds [1]
    {0: PBC_OFFSET_DIM, 1: torch.export.Dim.STATIC},  # num_triple_ij [num_edges, 1]
    {0: torch.export.Dim.STATIC},  # num_atoms [1]
    {},  # num_graphs (scalar)
    {0: NODE_DIM},  # batch [num_atoms]
)


def get_example_inputs(
    cutoff: float = 5.0,
    threebody_cutoff: float = 4.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, ...]:
    """Build a minimal example graph inputs on the given device.

    Creates a silicon diamond structure, converts it to a graph via the dataloader,
    and returns the batch inputs with tensors moved to the specified device.

    Args:
        cutoff: Radial cutoff used to build neighbors.
        threebody_cutoff: Cutoff for three-body neighborhood.
        device: Target device; defaults to CUDA if available, else CPU.

    Returns:
        Tuple of tensors ordered to match ``mattersim_dynamic_shapes``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    positions_list = [dummy_atoms.get_positions()]
    cell_list = [dummy_atoms.get_cell()]
    pbc_list = [dummy_atoms.get_pbc()]
    atomic_numbers_list = [dummy_atoms.get_atomic_numbers()]

    dataloader = build_dataloader(
        positions_list=positions_list,
        cell_list=cell_list,
        pbc_list=pbc_list,
        atomic_numbers_list=atomic_numbers_list,
        cutoff=cutoff,
        threebody_cutoff=threebody_cutoff,
        batch_size=len(positions_list),
    )

    graph_batch = next(iter(dataloader))
    example_dict = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch_to_dict(graph_batch).items()
    }
    example_inputs = batch_to_tuples(example_dict)
    return example_inputs


def batch_to_tuples(batch_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Convert a dataloader batch dict to the tuple expected by the model.

    The order matches ``mattersim_dynamic_shapes`` and the compiled model inputs.

    Args:
        batch_dict: Mapping produced by ``batch_to_dict`` containing the keys:
            - ``"atom_pos"``: Tensor [num_atoms, 3]
            - ``"cell"``: Tensor [1, 3, 3]
            - ``"pbc_offsets"``: Tensor [num_edges, 3]
            - ``"atom_attr"``: Tensor [num_atoms, 1]
            - ``"edge_index"``: LongTensor [2, num_edges]
            - ``"three_body_indices"``: LongTensor [num_three_body, 2]
            - ``"num_three_body"``: Tensor [1]
            - ``"num_bonds"``: Tensor [1]
            - ``"num_triple_ij"``: Tensor [num_edges, 1]
            - ``"num_atoms"``: Tensor [1]
            - ``"num_graphs"``: Scalar tensor or 0-D tensor
            - ``"batch"``: LongTensor [num_atoms]

    Returns:
        Tuple of tensors in the exact order required by the model:
        (atom_pos, cell, pbc_offsets, atom_attr, edge_index, three_body_indices,
        num_three_body, num_bonds, num_triple_ij, num_atoms, num_graphs, batch).
    """
    return (
        batch_dict["atom_pos"],
        batch_dict["cell"],
        batch_dict["pbc_offsets"],
        batch_dict["atom_attr"],
        batch_dict["edge_index"],
        batch_dict["three_body_indices"],
        batch_dict["num_three_body"],
        batch_dict["num_bonds"],
        batch_dict["num_triple_ij"],
        batch_dict["num_atoms"],
        batch_dict["num_graphs"],
        batch_dict["batch"],
    )
