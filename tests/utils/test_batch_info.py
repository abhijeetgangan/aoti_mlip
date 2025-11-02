import torch

from aoti_mlip.utils.batch_info import batch_to_tuples


def test_batch_to_tuples_ordering_and_shapes():
    num_atoms = 3
    num_edges = 2
    num_three_body = 4

    batch_dict = {
        "atom_pos": torch.zeros((num_atoms, 3)),
        "cell": torch.zeros((1, 3, 3)),
        "pbc_offsets": torch.zeros((num_edges, 3)),
        "atom_attr": torch.zeros((num_atoms, 1)),
        "edge_index": torch.zeros((2, num_edges), dtype=torch.long),
        "three_body_indices": torch.zeros((num_three_body, 2), dtype=torch.long),
        "num_three_body": torch.tensor([num_three_body]),
        "num_bonds": torch.tensor([num_edges]),
        "num_triple_ij": torch.zeros((num_edges, 1), dtype=torch.long),
        "num_atoms": torch.tensor([num_atoms]),
        "num_graphs": torch.tensor(1),
        "batch": torch.zeros((num_atoms,), dtype=torch.long),
    }

    tup = batch_to_tuples(batch_dict)
    assert len(tup) == 12
    assert tup[0].shape == (num_atoms, 3)
    assert tup[1].shape == (1, 3, 3)
    assert tup[2].shape == (num_edges, 3)
    assert tup[3].shape == (num_atoms, 1)
    assert tup[4].shape == (2, num_edges)
    assert tup[5].shape == (num_three_body, 2)
    assert tup[6].shape == (1,)
    assert tup[7].shape == (1,)
    assert tup[8].shape == (num_edges, 1)
    assert tup[9].shape == (1,)
    assert tup[11].shape == (num_atoms,)
