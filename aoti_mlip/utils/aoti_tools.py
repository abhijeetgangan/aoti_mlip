"""Utility helpers for FX tracing and model compilation.

This module provides small helpers to:
- toggle FX duck-shaping when symbolically tracing models,
- build symbolic FX graphs for given inputs, and
- prepare models for compilation (freeze grads, eval mode, device placement).
"""

import contextlib

import torch
from torch.fx.experimental.proxy_tensor import make_fx


@contextlib.contextmanager
def fx_duck_shape(enabled: bool):
    """
    Temporarily switch the FX duck-shaping flag. Required when we use `make_fx`
    with symbolic tracing.
    """
    prev = torch.fx.experimental._config.use_duck_shape  # type: ignore
    torch.fx.experimental._config.use_duck_shape = enabled  # type: ignore
    try:
        yield
    finally:
        torch.fx.experimental._config.use_duck_shape = prev  # type: ignore


def _make_fx(model: torch.nn.Module, inputs: tuple):
    """Create a symbolic-traced FX graph for the model and inputs."""
    with fx_duck_shape(False):
        return make_fx(
            model,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            _error_on_data_dependent_ops=True,
        )(*[i.clone() for i in inputs])


def model_make_fx(model: torch.nn.Module, inputs: tuple):
    """Return a symbolic-traced FX graph and clear CUDA cache."""
    fx_model = _make_fx(model, inputs)
    torch.cuda.empty_cache()
    return fx_model


def prepare_model_for_compile(model: torch.nn.Module, device: torch.device):
    """Disable grads, set eval mode, and move model to the device."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    model.to(device)
    return model
