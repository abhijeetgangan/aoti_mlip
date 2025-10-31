# -*- coding: utf-8 -*-
"""
Implementation of M3GNet model for materials property prediction.

This module implements the M3GNet model architecture described in the paper:
"M3GNet: Molecular Simulation with Multiple 3D Graph Networks"
(https://arxiv.org/abs/2111.06378)

The model uses message passing neural networks with 3-body interactions to learn
molecular and materials properties from atomic structures. Key features include:

- Edge features encoded using Bessel basis functions
- Angular features encoded using spherical harmonics
- Message passing blocks that update both node and edge features
- Support for periodic boundary conditions
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from aoti_mlip.models.mattersim_modules.angle_encoding import SphericalBasisLayer
from aoti_mlip.models.mattersim_modules.edge_encoding import SmoothBesselBasis
from aoti_mlip.models.mattersim_modules.layers import MLP, GatedMLP
from aoti_mlip.models.mattersim_modules.message_passing import MainBlock
from aoti_mlip.models.mattersim_modules.scaling import AtomScaling


class M3Gnet(nn.Module):
    """
    M3GNet model for materials property prediction.

    This class implements the core M3GNet architecture with configurable hyperparameters
    for the number of message passing blocks, embedding dimensions, and interaction cutoffs.

    Args:
        num_blocks (int): Number of message passing blocks. Default: 4
        units (int): Dimension of node and edge embeddings. Default: 128
        max_l (int): Maximum degree of spherical harmonics. Default: 4
        max_n (int): Maximum degree of Bessel basis. Default: 4
        cutoff (float): Cutoff radius for interactions in Angstroms. Default: 5.0
        device (torch.device): Device to place the model on. Default: CUDA if available, else CPU
        dtype (torch.dtype): Data type for model parameters. Default: float32
        max_z (int): Maximum atomic number supported. Default: 94
        threebody_cutoff (float): Cutoff radius for 3-body interactions. Default: 4.0
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        num_blocks: int = 4,
        units: int = 128,
        max_l: int = 4,
        max_n: int = 4,
        cutoff: float = 5.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        max_z: int = 94,
        threebody_cutoff: float = 4.0,
        **kwargs,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rbf = SmoothBesselBasis(r_max=cutoff, max_n=max_n)
        self.sbf = SphericalBasisLayer(
            max_n=max_n, max_l=max_l, cutoff=cutoff, device=device, dtype=dtype
        )
        self.edge_encoder = MLP(in_dim=max_n, out_dims=[units], activation="swish", use_bias=False)
        module_list = [
            MainBlock(max_n, max_l, cutoff, units, max_n, threebody_cutoff)
            for i in range(num_blocks)
        ]
        self.graph_conv = nn.ModuleList(module_list)
        self.final = GatedMLP(
            in_dim=units,
            out_dims=[units, units, 1],
            activation=["swish", "swish", None],
        )
        self.apply(self.init_weights)
        self.atom_embedding = MLP(
            in_dim=max_z + 1, out_dims=[units], activation=None, use_bias=False
        )
        self.atom_embedding.apply(self.init_weights_uniform)
        self.normalizer = AtomScaling(verbose=False, max_z=max_z)
        self.max_z = max_z  # type: ignore
        self.device = device  # type: ignore
        self.model_args = {  # type: ignore
            "num_blocks": num_blocks,
            "units": units,
            "max_l": max_l,
            "max_n": max_n,
            "cutoff": cutoff,
            "max_z": max_z,
            "threebody_cutoff": threebody_cutoff,
        }

    def forward(
        self,
        input: Dict[str, torch.Tensor],
        dataset_idx: int = -1,
    ) -> torch.Tensor:
        """
        Forward pass of the M3GNet model.

        Args:
            input (Dict[str, torch.Tensor]): Dictionary containing input tensors:
                - atom_pos: Atomic positions
                - cell: Unit cell vectors
                - pbc_offsets: Periodic boundary condition offsets
                - atom_attr: Atomic attributes (atomic numbers)
                - edge_index: Edge indices
                - three_body_indices: Three-body interaction indices
                - num_three_body: Number of three-body terms per graph
                - num_bonds: Number of bonds per graph
                - num_triple_ij: Number of triple interactions per graph
                - num_atoms: Number of atoms per graph
                - num_graphs: Total number of graphs
                - batch: Graph assignment for each atom
            dataset_idx (int): Index of dataset being used. Default: -1

        Returns:
            torch.Tensor: Predicted per-graph energies
        """
        # Exact data from input_dictionary
        pos = input["atom_pos"]
        cell = input["cell"]
        pbc_offsets = input["pbc_offsets"].float()
        atom_attr = input["atom_attr"]
        edge_index = input["edge_index"].long()
        three_body_indices = input["three_body_indices"].long()
        num_three_body = input["num_three_body"]
        num_bonds = input["num_bonds"]
        num_triple_ij = input["num_triple_ij"]
        num_atoms = input["num_atoms"]
        num_graphs = input["num_graphs"]
        batch = input["batch"]

        cumsum = torch.cumsum(num_bonds, dim=0) - num_bonds
        index_bias = torch.repeat_interleave(cumsum, num_three_body, dim=0).unsqueeze(-1)
        three_body_indices = three_body_indices + index_bias

        # === Refer to the implementation of M3GNet,        ===
        # === we should re-compute the following attributes ===
        # edge_length, edge_vector(optional), triple_edge_length, theta_jik
        atoms_batch = torch.repeat_interleave(repeats=num_atoms)
        edge_batch = atoms_batch[edge_index[0]]
        edge_vector = pos[edge_index[0]] - (
            pos[edge_index[1]] + torch.einsum("bi, bij->bj", pbc_offsets, cell[edge_batch])
        )
        edge_length = torch.linalg.norm(edge_vector, dim=1)
        vij = edge_vector[three_body_indices[:, 0]]
        vik = edge_vector[three_body_indices[:, 1]]
        rij = edge_length[three_body_indices[:, 0]]
        rik = edge_length[three_body_indices[:, 1]]
        cos_jik = torch.sum(vij * vik, dim=1) / (rij * rik)
        # eps = 1e-7 avoid nan in torch.acos function
        cos_jik = torch.clamp(cos_jik, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        triple_edge_length = rik.view(-1)
        edge_length = edge_length.unsqueeze(-1)
        atomic_numbers = atom_attr.squeeze(1).long()

        # featurize
        atom_attr = self.atom_embedding(self.one_hot_atoms(atomic_numbers))
        edge_attr = self.rbf(edge_length.view(-1))
        edge_attr_zero = edge_attr  # e_ij^0
        edge_attr = self.edge_encoder(edge_attr)
        three_basis = self.sbf(triple_edge_length, torch.acos(cos_jik))

        # Main Loop
        for _, conv in enumerate(self.graph_conv):
            atom_attr, edge_attr = conv(
                atom_attr,
                edge_attr,
                edge_attr_zero,
                edge_index,
                three_basis,
                three_body_indices,
                edge_length,
                num_bonds,
                num_triple_ij,
                num_atoms,
            )

        energies_i = self.final(atom_attr).view(-1)  # [batch_size*num_atoms]
        energies_i = self.normalizer(energies_i, atomic_numbers)

        # NOTE: This is with torch_runstats.scatter.scatter (similar to torch_scatter.scatter_add)
        # energies = scatter(energies_i, batch, dim=0, dim_size=num_graphs)

        # NOTE: This is with torch.scatter_add_
        output = torch.zeros(num_graphs, device=energies_i.device, dtype=energies_i.dtype)  # type: ignore
        energies = output.scatter_add_(0, batch, energies_i)

        return energies  # [batch_size]

    def init_weights(self, m):
        """Initialize weights using Xavier uniform initialization."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    def init_weights_uniform(self, m):
        """Initialize weights using uniform initialization."""
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, a=-0.05, b=0.05)

    def one_hot_atoms(self, species):
        """Convert atomic numbers to one-hot encoded vectors."""
        return F.one_hot(species, num_classes=self.max_z + 1).float()

    def set_normalizer(self, normalizer: AtomScaling):
        """Set the energy normalizer."""
        self.normalizer = normalizer

    def get_model_args(self):
        """Get model architecture arguments."""
        return self.model_args
