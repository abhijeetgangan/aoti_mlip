# Third-party notice
# Some FX tracing patterns and utilities for AOTInductor are adapted from NequIP
# (https://github.com/mir-group/nequip) under the MIT license.

"""Utility helpers for FX tracing and model compilation."""

import contextlib
from typing import Callable

import torch
from torch.fx.experimental.proxy_tensor import make_fx

# for `test_model_output_similarity`, we perform evaluation
# `num_eval_trials` times to account for numerical randomness in the model
_NUM_EVAL_TRIALS = 5


@contextlib.contextmanager
def fx_duck_shape(enabled: bool):
    """
    Temporarily switch the FX duck-shaping flag. Required when we use ``make_fx``
    with symbolic tracing.

    Args:
        enabled: Whether to enable (True) or disable (False) duck-shaping during tracing.

    Yields:
        None. Used as a context manager to wrap tracing code.
    """
    prev = torch.fx.experimental._config.use_duck_shape  # type: ignore
    torch.fx.experimental._config.use_duck_shape = enabled  # type: ignore
    try:
        yield
    finally:
        torch.fx.experimental._config.use_duck_shape = prev  # type: ignore


def _make_fx(model: torch.nn.Module, inputs: tuple):
    """Create a symbolic-traced FX graph for the model and inputs.

    Args:
        model: The eager ``torch.nn.Module`` to trace.
        inputs: Example inputs as a tuple of tensors.

    Returns:
        An FX-traced GraphModule representing ``model`` under the provided inputs.
    """
    with fx_duck_shape(False):
        return make_fx(
            model,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            _error_on_data_dependent_ops=True,
        )(*[i.clone() for i in inputs])


def model_make_fx(model: torch.nn.Module, inputs: tuple):
    """Return a symbolic-traced FX graph and clear CUDA cache.

    Args:
        model: The eager ``torch.nn.Module`` to trace.
        inputs: Example inputs as a tuple of tensors.

    Returns:
        The FX-traced GraphModule for the given ``model`` and ``inputs``.
    """
    fx_model = _make_fx(model, inputs)
    torch.cuda.empty_cache()
    return fx_model


def prepare_model_for_compile(model: torch.nn.Module, device: torch.device):
    """Disable grads, set eval mode, and move model to the device.

    Args:
        model: The model to prepare.
        device: Target device to place the model on.

    Returns:
        The same model after ``requires_grad_(False)``, ``eval()``, and ``to(device)``.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    model.to(device)
    return model


def _default_error_message(key, tol, err, absval):
    """Format a standardized assertion message for similarity checks.

    Args:
        key: Output field name being compared (e.g., "energy", "forces").
        tol: Tolerance used for both ``atol`` and ``rtol`` in similarity tests.
        err: Maximum absolute difference observed between reference and test tensors.
        absval: Maximum absolute value observed for the reference tensor.

    Returns:
        A concise, human-readable string describing the mismatch for ``key``.
    """
    return f"MaxAbsError: {err:.6g} (tol: {tol}) for field `{key}`. MaxAbs value: {absval:.6g}."


def test_model_output_similarity_by_dtype(
    aot_model: Callable,
    model: Callable,
    example_inputs: tuple,
    error_message: Callable = _default_error_message,
    tol: float = 1e-4,
    num_eval_trials: int = _NUM_EVAL_TRIALS,
):
    """Validate numerical agreement between two model callables.

    Runs both ``aot_model`` and ``model`` multiple times on the same
    ``example_inputs``, upcasts outputs to ``float64`` to avoid dtype-related
    noise, averages results across trials to smooth numerical nondeterminism,
    and asserts that each output field is ``allclose`` within ``tol``.

    Assumptions:
    - ``aot_model`` and ``model`` accept positional unpacking of ``example_inputs``
      (i.e., signatures like ``(*args) -> dict[str, torch.Tensor]``).
    - Both return dictionaries with identical keys and tensor values.

    Args:
        aot_model: Callable invoked as ``aot_model(*example_inputs) -> dict``.
        model: Callable invoked as ``model(*example_inputs) -> dict``.
        example_inputs: Tuple of tensors passed positionally to both callables.
        error_message: Function to format assertion messages. Receives
            ``(key, tol, err, absval)`` and should return a string.
        tol: Absolute and relative tolerance used for ``torch.allclose`` comparisons.
        num_eval_trials: Number of repeated evaluations to average for stability.

    Returns:
        None. Raises ``AssertionError`` if any output field diverges beyond ``tol``.
    """
    fields = aot_model(*example_inputs).keys()

    # perform `num_eval_trials` evaluations with each model and
    # average the results to account for numerical randomness
    out1_list, out2_list = {k: [] for k in fields}, {k: [] for k in fields}
    for _ in range(num_eval_trials):
        out1 = aot_model(*example_inputs)
        out2 = model(*example_inputs)
        for k in fields:
            out1_list[k].append(out1[k].detach().double())
            out2_list[k].append(out2[k].detach().double())
        del out1, out2

    for k in fields:
        t1, t2 = (
            torch.mean(torch.stack(out1_list[k], -1), -1),
            torch.mean(torch.stack(out2_list[k], -1), -1),
        )
        err = torch.max(torch.abs(t1 - t2)).item()
        absval = t1.abs().max().item()

        assert torch.allclose(t1, t2, atol=tol, rtol=tol), error_message(k, tol, err, absval)

        del t1, t2, err, absval
