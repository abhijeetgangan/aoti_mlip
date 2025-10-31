"""
Atomic scaling module for predicting extensive properties.

This module provides scaling functionality for atomic properties like energies and forces.
It allows per-atom normalization by applying a scale and shift factor based on atomic numbers.
This is particularly useful for training models on datasets with varying energy scales.
"""

from typing import Union

import torch
import torch.nn as nn


class AtomScaling(nn.Module):
    """
    Atomic extensive property rescaling module.

    This module scales atomic properties (like energies) on a per-atom basis using
    learned or provided scale and shift parameters for each atomic number.

    The transformation applied is:
        normalized = scale * original + shift

    where scale and shift are vectors indexed by atomic number.
    """

    def __init__(
        self,
        max_z: int = 94,
        init_scale: Union[torch.Tensor, float] | None = None,
        init_shift: Union[torch.Tensor, float] | None = None,
        verbose: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        """Initialize the atomic scaling module.

        Args:
            max_z: Maximum atomic number to support. Determines size of scale/shift vectors.
                If using scale_key/shift_key, should match maximum atomic number in data.
            init_scale: Initial scale factors. Can be:
                - None: Defaults to ones vector
                - float: Value repeated for all atomic numbers
                - torch.Tensor: Vector of size max_z + 1
            init_shift: Initial shift values. Can be:
                - None: Defaults to zeros vector
                - float: Value repeated for all atomic numbers
                - torch.Tensor: Vector of size max_z + 1
            verbose: Whether to print scale/shift values during initialization
            device: Device to place tensors on
            **kwargs: Additional arguments
        """
        super().__init__()

        self.max_z = max_z  # type: ignore
        self.device = device  # type: ignore

        # === initial values are given ===
        if init_scale is None:
            init_scale = torch.ones(max_z + 1)
        elif isinstance(init_scale, float):
            init_scale = torch.tensor(init_scale).repeat(max_z + 1)
        else:
            assert init_scale.size()[0] == max_z + 1  # type: ignore

        if init_shift is None:
            init_shift = torch.zeros(max_z + 1)
        elif isinstance(init_shift, float):
            init_shift = torch.tensor(init_shift).repeat(max_z + 1)
        else:
            assert init_shift.size()[0] == max_z + 1  # type: ignore

        init_shift = init_shift.float()  # type: ignore
        init_scale = init_scale.float()  # type: ignore
        self.register_buffer("scale", init_scale)
        self.register_buffer("shift", init_shift)

        if verbose is True:
            print("Current scale: ", init_scale, self.scale)
            print("Current shift: ", init_shift, self.shift)

        self.to(device)

    def transform(
        self, atomic_energies: torch.Tensor, atomic_numbers: torch.Tensor
    ) -> torch.Tensor:
        """Transform atomic values using the scale and shift parameters.

        Takes raw values from the model and applies the normalization transform:
            normalized = scale * original + shift

        Args:
            atomic_energies: Per-atom energy values to transform
            atomic_numbers: Atomic numbers corresponding to each energy value

        Returns:
            Transformed (normalized) energy values
        """
        curr_shift = self.shift[atomic_numbers]  # type: ignore
        curr_scale = self.scale[atomic_numbers]  # type: ignore
        normalized_energies = curr_scale * atomic_energies + curr_shift
        return normalized_energies

    def inverse_transform(
        self, atomic_energies: torch.Tensor, atomic_numbers: torch.Tensor
    ) -> torch.Tensor:
        """Inverse transform normalized values back to original scale.

        Takes normalized values and reverses the transform:
            original = (normalized - shift) / scale

        Args:
            atomic_energies: Normalized per-atom energy values
            atomic_numbers: Atomic numbers corresponding to each energy value

        Returns:
            Original (unnormalized) energy values
        """
        curr_shift = self.shift[atomic_numbers]  # type: ignore
        curr_scale = self.scale[atomic_numbers]  # type: ignore
        unnormalized_energies = (atomic_energies - curr_shift) / curr_scale
        return unnormalized_energies

    def forward(self, atomic_energies: torch.Tensor, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """Forward pass applies the transform to input values.

        Args:
            atomic_energies: Per-atom energy values to transform
            atomic_numbers: Atomic numbers corresponding to each energy value
                Must be same size as atomic_energies

        Returns:
            Transformed energy values
        """
        return self.transform(atomic_energies, atomic_numbers)
