import pytest
import torch

from aoti_mlip.utils.batch_info import MATTERSIM_DYNAMIC_SHAPES, get_example_inputs


def test_get_example_inputs():
    """Test that get_example_inputs returns a tuple with correct number of elements."""
    example_inputs = get_example_inputs(cutoff=5.0, threebody_cutoff=4.0)

    # Should return a tuple with 12 elements matching MATTERSIM_DYNAMIC_SHAPES
    assert isinstance(example_inputs, tuple)
    assert len(example_inputs) == 12
    assert len(example_inputs) == len(MATTERSIM_DYNAMIC_SHAPES)

    # Verify all elements are tensors
    for tensor in example_inputs:
        assert isinstance(tensor, torch.Tensor)


def _unpack(example_inputs):
    return (
        example_inputs[0],  # atom_pos
        example_inputs[1],  # cell
        example_inputs[2],  # pbc_offsets
        example_inputs[3],  # atom_attr
        example_inputs[4],  # edge_index
        example_inputs[5],  # three_body_indices
        example_inputs[6],  # num_three_body
        example_inputs[7],  # num_bonds
        example_inputs[8],  # num_triple_ij
        example_inputs[9],  # num_atoms
        example_inputs[10],  # num_graphs
        example_inputs[11],  # batch
    )


def test_example_inputs_shapes_single():
    """Shapes for a single-structure batch (num_structures=1)."""
    example_inputs = get_example_inputs(cutoff=5.0, threebody_cutoff=4.0, num_structures=1)
    (
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
    ) = _unpack(example_inputs)

    n_atoms = atom_pos.shape[0]
    n_edges = pbc_offsets.shape[0]

    assert atom_pos.shape == (n_atoms, 3)
    assert cell.shape == (1, 3, 3)
    assert pbc_offsets.shape == (n_edges, 3)
    assert atom_attr.shape == (n_atoms, 1)
    assert edge_index.shape == (2, n_edges)
    assert three_body_indices.ndim == 2 and three_body_indices.shape[1] == 2
    assert num_three_body.shape == (1,)
    assert num_bonds.shape == (1,)
    assert num_triple_ij.shape == (n_edges, 1)
    assert num_atoms.shape == (1,)
    assert num_graphs.ndim == 0
    assert batch.shape == (n_atoms,)


@pytest.mark.parametrize("num_structures", [2, 3])
def test_example_inputs_shapes_batched(num_structures):
    """Shapes for a multi-structure batch."""
    example_inputs = get_example_inputs(
        cutoff=5.0, threebody_cutoff=4.0, num_structures=num_structures
    )
    (
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
    ) = _unpack(example_inputs)

    n_atoms_total = atom_pos.shape[0]
    n_edges_total = pbc_offsets.shape[0]

    assert cell.shape == (num_structures, 3, 3)
    assert atom_attr.shape == (n_atoms_total, 1)
    assert edge_index.shape == (2, n_edges_total)
    assert num_three_body.shape == (num_structures,)
    assert num_bonds.shape == (num_structures,)
    assert num_atoms.shape == (num_structures,)
    assert num_triple_ij.shape == (n_edges_total, 1)
    assert num_graphs.item() == num_structures
    assert batch.shape == (n_atoms_total,)
    assert int(num_atoms.sum()) == n_atoms_total
    assert batch.max().item() == num_structures - 1
