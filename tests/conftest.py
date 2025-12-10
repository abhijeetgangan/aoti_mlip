import pytest
from ase.atoms import Atoms
from ase.build import bulk
from ase.spacegroup import crystal


@pytest.fixture
def fe_atoms() -> Atoms:
    """Crystalline iron structure"""
    return bulk("Fe", "fcc", a=5.26, cubic=True)


@pytest.fixture
def casio3_atoms() -> Atoms:
    """Crystalline casio3 structure"""
    a, b, c = 7.9258, 7.3202, 7.0653
    alpha, beta, gamma = 90.055, 95.217, 103.426
    basis = [
        ("Ca", 0.19831, 0.42266, 0.76060),
        ("Ca", 0.20241, 0.92919, 0.76401),
        ("Ca", 0.50333, 0.75040, 0.52691),
        ("Si", 0.1851, 0.3875, 0.2684),
        ("Si", 0.1849, 0.9542, 0.2691),
        ("Si", 0.3973, 0.7236, 0.0561),
        ("O", 0.3034, 0.4616, 0.4628),
        ("O", 0.3014, 0.9385, 0.4641),
        ("O", 0.5705, 0.7688, 0.1988),
        ("O", 0.9832, 0.3739, 0.2655),
        ("O", 0.9819, 0.8677, 0.2648),
        ("O", 0.4018, 0.7266, 0.8296),
        ("O", 0.2183, 0.1785, 0.2254),
        ("O", 0.2713, 0.8704, 0.0938),
        ("O", 0.2735, 0.5126, 0.0931),
    ]
    atoms = crystal(
        symbols=[b[0] for b in basis],
        basis=[b[1:] for b in basis],
        spacegroup=2,
        cellpar=[a, b, c, alpha, beta, gamma],
    )
    return atoms
