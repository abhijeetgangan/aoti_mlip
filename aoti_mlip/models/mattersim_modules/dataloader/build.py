"""Utilities for building dataloaders and processing batches."""

from typing import Any

import numpy as np
import torch
from torch_geometric.loader import DataLoader as DataLoader_pyg

from aoti_mlip.models.mattersim_modules.dataloader.converter import GraphConverter


def batch_to_dict(
    graph_batch: Any,
) -> dict[str, torch.Tensor]:
    """Convert a graph batch to a dictionary of tensors.

    Args:
        graph_batch: The graph batch to convert

    Returns:
        Dictionary containing the graph tensors
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
    """Build a dataloader given a list of atoms.

    Args:
        positions: A list of atomic positions (np.ndarray) with unit Å
        cell: A list of cell vectors (np.ndarray) with unit Å
        pbc: A list of pbc flags (np.ndarray)
        atomic_numbers: A list of atomic numbers (np.ndarray)
        cutoff: Cutoff radius for neighbor finding
        threebody_cutoff: Cutoff radius for three-body interactions
        batch_size: Number of graphs in each batch
        shuffle: Whether to shuffle the dataset
        num_workers: Number of workers for dataloader
        pin_memory: If True, the datasets will be stored in GPU or CPU memory

    Returns:
        A PyTorch Geometric DataLoader object
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
