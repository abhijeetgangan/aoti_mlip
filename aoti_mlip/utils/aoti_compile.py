import logging
import os

import torch

from aoti_mlip.models.mattersim import M3GnetModel, M3GnetWrapper
from aoti_mlip.utils.aoti_tools import model_make_fx, prepare_model_for_compile
from aoti_mlip.utils.batch_info import get_example_inputs, mattersim_dynamic_shapes
from aoti_mlip.utils.download_utils import download_mattersim_checkpoint

logger = logging.getLogger(__name__)


def compile_mattersim(
    checkpoint_name: str,
    cutoff: float = 5.0,
    threebody_cutoff: float = 4.0,
    compute_force: bool = True,
    compute_stress: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    """Compile and package a pre-trained MatterSim (M3GNet) model into a .pt2 artifact.

    This function prepares a MatterSim model for Ahead-of-Time compilation using
    PyTorch AOTInductor, validates the compiled model against the eager model on
    example inputs, and writes a redistributable .pt2 package to
    model_checkpoints/<checkpoint_name>_<device>.pt2 in the current working directory.

    Args:
        checkpoint_name: Base name (without extension) of a pre-trained MatterSim weights
            file under ~/.local/mattersim/pretrained_models/. If missing, it will be
            downloaded automatically, e.g. "mattersim-v1.0.0-1M".
        cutoff: Radial neighbor cutoff (in Å) for pair interactions used to build example inputs.
        threebody_cutoff: Cutoff (in Å) used for three-body neighborhood construction.
        compute_force: If True, compile a model that computes per-atom forces.
        compute_stress: If True, compile a model that computes stress.
        device: Target device string ("cpu", "cuda", or "cuda:N"). Defaults to "cuda" if
            available, otherwise "cpu". The compiled package will be device-specific.

    Returns:
        Absolute path to the generated AOTInductor package (.pt2).

    Raises:
        ValueError: If the AOT-compiled model's energy deviates from the eager model
            by more than atol=1e-4 during validation.
        RuntimeError: If export or AOTInductor packaging fails.
        OSError: If a CUDA device is requested but unavailable.
    """
    BASE_PATH = "~/.local/mattersim/pretrained_models/"
    CKPT_PATH = os.path.expanduser(f"{BASE_PATH}/{checkpoint_name}")
    if not os.path.exists(CKPT_PATH):
        download_mattersim_checkpoint(checkpoint_name)
    logger.info(f"Compiling mattersim model from {checkpoint_name}...")
    PACKAGE_PATH = os.path.expanduser(f"{BASE_PATH}/{checkpoint_name.replace('.pth', '.pt2')}")
    example_inputs = get_example_inputs(
        cutoff=cutoff, threebody_cutoff=threebody_cutoff, device=torch.device(device)
    )
    model_to_compile = M3GnetModel(
        model=torch.load(CKPT_PATH, map_location=device, weights_only=True),
        device=str(device),
        compute_force=compute_force,
        compute_stress=compute_stress,
    )

    wrapped_model = M3GnetWrapper(model_to_compile)
    model_to_compile = prepare_model_for_compile(wrapped_model, torch.device(device))
    logger.info("Validating model outputs...")
    results = model_to_compile(*example_inputs)
    logger.info(f"Keys for results: {results.keys()}")
    logger.info("Exporting model...")
    fx_model = model_make_fx(model_to_compile, example_inputs)

    exported_model = torch.export.export(
        fx_model,
        example_inputs,
        dynamic_shapes=mattersim_dynamic_shapes,
    )
    _AOT_METADATA_KEY = "aot_inductor.metadata"
    INDUCTOR_CFG = {
        _AOT_METADATA_KEY: {"cutoff": str(cutoff), "threebody_cutoff": str(threebody_cutoff)},
    }
    torch._inductor.aoti_compile_and_package(
        exported_model,
        package_path=PACKAGE_PATH,
        inductor_configs=INDUCTOR_CFG,
    )
    logger.info(f"Exported model saved to: {PACKAGE_PATH}")

    logger.info("Validating loading for exported model...")
    aot_model = torch._inductor.aoti_load_package(PACKAGE_PATH)
    aot_results = aot_model(*example_inputs)
    logger.info(f"Keys for results: {aot_results.keys()}")
    IF_ENERGY_CLOSE = torch.allclose(results["energy"], aot_results["energy"], atol=1e-4)
    if not IF_ENERGY_CLOSE:
        raise ValueError("Energy is not close between original and AOT-compiled model")
    logger.info("Printing metadata for model...")
    metadata = aot_model.get_metadata()
    logger.info(f"Metadata: {metadata}")
    logger.info("Model exported successfully")
    return PACKAGE_PATH
