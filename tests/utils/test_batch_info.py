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


def test_example_inputs_shapes():
    """Test that get_example_inputs returns tensors with expected shapes."""
    example_inputs = get_example_inputs(cutoff=5.0, threebody_cutoff=4.0)

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
    ) = example_inputs

    # Check atom_pos shape [num_atoms, 3]
    assert atom_pos.ndim == 2
    assert atom_pos.shape[1] == 3

    # Check cell shape [1, 3, 3]
    assert cell.shape == (1, 3, 3)

    # Check pbc_offsets shape [num_edges, 3]
    assert pbc_offsets.ndim == 2
    assert pbc_offsets.shape[1] == 3

    # Check atom_attr shape [num_atoms, 1]
    assert atom_attr.ndim == 2
    assert atom_attr.shape[0] == atom_pos.shape[0]
    assert atom_attr.shape[1] == 1

    # Check edge_index shape [2, num_edges]
    assert edge_index.shape == (2, pbc_offsets.shape[0])

    # Check three_body_indices shape [num_three_body, 2]
    assert three_body_indices.ndim == 2
    assert three_body_indices.shape[1] == 2

    # Check scalar/1D tensors
    assert num_three_body.shape == (1,)
    assert num_bonds.shape == (1,)
    assert num_triple_ij.shape == (pbc_offsets.shape[0], 1)
    assert num_atoms.shape == (1,)
    assert num_graphs.ndim == 0 or num_graphs.shape == ()
    assert batch.shape == (atom_pos.shape[0],)
