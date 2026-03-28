"""Tests for the TorchSim AOTI MatterSim model interface."""

import traceback

import pytest
import torch

from aoti_mlip.utils.aoti_compile import compile_mattersim

try:
    from torch_sim.models.interface import validate_model_outputs
    from torch_sim.testing import SIMSTATE_GENERATORS, assert_model_calculator_consistency

    from aoti_mlip.calculators.torchsim import MatterSimTorchSimModel
except (ImportError, OSError, RuntimeError):
    _msg = f"torch-sim not installed: {traceback.format_exc()}"
    pytest.skip(_msg, allow_module_level=True)

from aoti_mlip.calculators.mattersim import MatterSimCalculator

DEVICE = torch.device("cpu")
DTYPE = torch.float64


@pytest.fixture(scope="module")
def compiled_model_path():
    """Compile a MatterSim model and return the .pt2 path."""
    return compile_mattersim(
        checkpoint_name="mattersim-v1.0.0-1M.pth",
        cutoff=5.0,
        threebody_cutoff=4.0,
        compute_force=True,
        compute_stress=True,
        device="cpu",
    )


@pytest.fixture
def torchsim_model(compiled_model_path: str) -> MatterSimTorchSimModel:
    """Create a MatterSimTorchSimModel from a compiled .pt2 package."""
    return MatterSimTorchSimModel(model_path=compiled_model_path, device=DEVICE)


@pytest.fixture
def aoti_calculator(compiled_model_path: str) -> MatterSimCalculator:
    """Create an ASE MatterSimCalculator from a compiled .pt2 package."""
    return MatterSimCalculator(model_path=compiled_model_path, device="cpu")


def test_torchsim_model_initialization(compiled_model_path: str) -> None:
    """Test that the AOTI TorchSim model initializes correctly."""
    model = MatterSimTorchSimModel(model_path=compiled_model_path, device=DEVICE)
    assert model.device == DEVICE
    assert model.compute_stress is True
    assert model.compute_forces is True
    assert model.cutoff > 0
    assert model.threebody_cutoff > 0


def test_torchsim_model_outputs(torchsim_model: MatterSimTorchSimModel) -> None:
    """Validate model outputs conform to the TorchSim ModelInterface contract.

    Checks output shapes, batched vs individual consistency, PBC invariance,
    and that outputs are detached from autograd.
    """
    validate_model_outputs(torchsim_model, DEVICE, DTYPE)


@pytest.mark.parametrize("sim_state_name", tuple(SIMSTATE_GENERATORS.keys()))
def test_torchsim_consistency(
    sim_state_name: str,
    torchsim_model: MatterSimTorchSimModel,
    aoti_calculator: MatterSimCalculator,
) -> None:
    """Test consistency between TorchSim model and ASE calculator."""
    sim_state = SIMSTATE_GENERATORS[sim_state_name](DEVICE, DTYPE)
    assert_model_calculator_consistency(
        model=torchsim_model,
        calculator=aoti_calculator,
        sim_state=sim_state,
    )
