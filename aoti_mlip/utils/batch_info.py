"""Dynamic shape specs and small helpers for MatterSim example batches.

This module defines dynamic shape constraints used during ``torch.export`` and
provides utilities to construct a minimal example batch and convert it to the
tuple form expected by the compiled model.
"""

import torch
from ase.build import bulk

from aoti_mlip.models.mattersim_modules.dataloader.build import build_dataloader, unpack_graph_batch

NODE_DIM = torch.export.dynamic_shapes.Dim("num_atoms", min=1, max=torch.inf)  # type: ignore
PBC_OFFSET_DIM = torch.export.dynamic_shapes.Dim("num_edges", min=1, max=torch.inf)  # type: ignore
THREE_BODY_DIM = torch.export.dynamic_shapes.Dim("num_three_body", min=1, max=torch.inf)  # type: ignore
GRAPH_DIM = torch.export.dynamic_shapes.Dim("num_graphs", min=1, max=torch.inf)  # type: ignore

MATTERSIM_DYNAMIC_SHAPES = (
    {0: NODE_DIM, 1: torch.export.Dim.STATIC},  # atom_pos [N_total, 3]
    {  # cell [num_graphs, 3, 3]
        0: GRAPH_DIM,
        1: torch.export.Dim.STATIC,
        2: torch.export.Dim.STATIC,
    },
    {0: PBC_OFFSET_DIM, 1: torch.export.Dim.STATIC},  # pbc_offsets [E_total, 3]
    {0: NODE_DIM, 1: torch.export.Dim.STATIC},  # atom_attr [N_total, 1]
    {0: torch.export.Dim.STATIC, 1: PBC_OFFSET_DIM},  # edge_index [2, E_total]
    {0: THREE_BODY_DIM, 1: torch.export.Dim.STATIC},  # three_body_indices [T_total, 2]
    {0: GRAPH_DIM},  # num_three_body [num_graphs]
    {0: GRAPH_DIM},  # num_bonds [num_graphs]
    {0: PBC_OFFSET_DIM, 1: torch.export.Dim.STATIC},  # num_triple_ij [E_total, 1]
    {0: GRAPH_DIM},  # num_atoms [num_graphs]
    {},  # num_graphs (scalar)
    {0: NODE_DIM},  # batch [N_total]
)


def get_example_inputs(
    cutoff: float = 5.0,
    threebody_cutoff: float = 4.0,
    device: torch.device | None = None,
    num_structures: int = 1,
) -> tuple[torch.Tensor, ...]:
    """Build a minimal example graph batch on the given device.

    Creates one or more small structures, converts them to a batched graph via the
    dataloader, and returns the inputs as a tuple of tensors.

    Args:
        cutoff: Radial cutoff used to build neighbors.
        threebody_cutoff: Cutoff for three-body neighborhood.
        device: Target device; defaults to CUDA if available, else CPU.
        num_structures: Number of structures in the example batch.  Use ``> 1``
            to produce a genuine multi-graph batch for export validation.

    Returns:
        Tuple of tensors ordered to match ``MATTERSIM_DYNAMIC_SHAPES``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_structures = [
        bulk("Si", "diamond", a=5.43, cubic=True),
        bulk("Fe", "bcc", a=2.86, cubic=True),
        bulk("Cu", "fcc", a=3.61, cubic=True),
    ]

    structures = [dummy_structures[i % len(dummy_structures)] for i in range(num_structures)]

    positions_list = [s.get_positions() for s in structures]
    cell_list = [s.get_cell() for s in structures]
    pbc_list = [s.get_pbc() for s in structures]
    atomic_numbers_list = [s.get_atomic_numbers() for s in structures]

    dataloader = build_dataloader(
        positions_list=positions_list,
        cell_list=cell_list,
        pbc_list=pbc_list,
        atomic_numbers_list=atomic_numbers_list,
        cutoff=cutoff,
        threebody_cutoff=threebody_cutoff,
        batch_size=len(positions_list),
    )

    graph_batch = next(iter(dataloader)).to(device)
    example_inputs = unpack_graph_batch(graph_batch)
    return example_inputs
