"""Utilities for building dataloaders and processing batches."""

from typing import Any

import numpy as np
import torch
from torch_geometric.loader import DataLoader as DataLoader_pyg

from aoti_mlip.models.mattersim_modules.dataloader.converter import GraphConverter


def batch_to_dict(
    graph_batch: Any,
) -> dict[str, torch.Tensor]:
    """Convert a PyTorch Geometric graph batch to a dictionary of tensors expected by the model.

    Args:
        graph_batch: A batched graph object containing
            attributes used by the MatterSim model.

    Returns:
        A dictionary with keys and typical shapes:
        - "atom_pos": float32 tensor [num_atoms, 3]
        - "cell": float32 tensor [1, 3, 3]
        - "pbc_offsets": float32 tensor [num_edges, 3]
        - "atom_attr": float32 tensor [num_atoms, 1]
        - "edge_index": int64 tensor [2, num_edges]
        - "three_body_indices": int64 tensor [num_three_body, 2]
        - "num_three_body": int64/float tensor [1]
        - "num_bonds": int64/float tensor [1]
        - "num_triple_ij": int64/float tensor [num_edges, 1]
        - "num_atoms": int64/float tensor [1]
        - "num_graphs": int64/float scalar tensor
        - "batch": int64 tensor [num_atoms]
    """
    atom_pos = graph_batch.atom_pos
    cell = graph_batch.cell
    pbc_offsets = graph_batch.pbc_offsets
    atom_attr = graph_batch.atom_attr
    edge_index = graph_batch.edge_index
    three_body_indices = graph_batch.three_body_indices
    num_three_body = graph_batch.num_three_body
    num_bonds = graph_batch.num_bonds
    num_triple_ij = graph_batch.num_triple_ij
    num_atoms = graph_batch.num_atoms
    num_graphs = graph_batch.num_graphs
    num_graphs = torch.tensor(num_graphs)
    batch = graph_batch.batch

    # Resemble input dictionary
    graph_dict = {}
    graph_dict["atom_pos"] = atom_pos
    graph_dict["cell"] = cell
    graph_dict["pbc_offsets"] = pbc_offsets
    graph_dict["atom_attr"] = atom_attr
    graph_dict["edge_index"] = edge_index
    graph_dict["three_body_indices"] = three_body_indices
    graph_dict["num_three_body"] = num_three_body
    graph_dict["num_bonds"] = num_bonds
    graph_dict["num_triple_ij"] = num_triple_ij
    graph_dict["num_atoms"] = num_atoms
    graph_dict["num_graphs"] = num_graphs
    graph_dict["batch"] = batch

    return graph_dict


def build_dataloader(
    positions_list: list[np.ndarray],
    cell_list: list[np.ndarray],
    pbc_list: list[np.ndarray],
    atomic_numbers_list: list[np.ndarray],
    cutoff: float = 5.0,
    threebody_cutoff: float = 4.0,
    batch_size: int = 64,
    shuffle: bool = False,  # noqa: FBT001, FBT002
    num_workers: int = 0,
    pin_memory: bool = False,  # noqa: FBT001, FBT002
) -> DataLoader_pyg:
    """Build a PyTorch Geometric dataloader from atomic structures.

    Args:
        positions_list: List of arrays of atomic positions, each [N_i, 3].
        cell_list: List of cell tensors/arrays for each structure, each [3, 3].
        pbc_list: List of periodic boundary condition flags for each structure, each [3] bools.
        atomic_numbers_list: List of arrays of atomic numbers for each structure, each [N_i].
        cutoff: Radial cutoff for neighbor finding.
        threebody_cutoff: Cutoff for three-body interactions.
        batch_size: Number of structures per batch.
        shuffle: Whether to shuffle the dataset.
        num_workers: Number of worker processes for data loading.
        pin_memory: If True, enable pinned memory for faster host-to-device transfer.

    Returns:
        A ``torch_geometric.loader.DataLoader`` yielding batched graph objects.
    """
    converter = GraphConverter(
        cutoff,
        has_threebody=True,
        threebody_cutoff=threebody_cutoff,
    )

    preprocessed_data = []

    for positions, cell, pbc, atomic_numbers in zip(
        positions_list,
        cell_list,
        pbc_list,
        atomic_numbers_list,
        strict=False,
    ):
        graph = converter.convert(
            positions,
            atomic_numbers,
            cell,
            pbc,
        )
        if graph is not None:
            preprocessed_data.append(graph)

    return DataLoader_pyg(
        preprocessed_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
