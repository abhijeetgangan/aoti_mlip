"""
Smooth Bessel basis functions for encoding radial distances in neural networks.

References:
    - Implementation based on: https://github.com/mir-group/nequip
    - Theory from: https://www.nature.com/articles/s41467-022-29939-5
    - Original paper: https://arxiv.org/abs/2003.03123 (DimeNet)
    - Mathematical details: https://arxiv.org/pdf/1907.02374.pdf

This module implements smooth radial Bessel basis functions that are used to encode
interatomic distances in message passing neural networks. The basis functions are
derived from spherical Bessel functions of order 0 and are designed to have smooth
first and second derivatives at the cutoff radius. This ensures continuous forces
and is particularly important for molecular dynamics applications.

The basis set is orthogonal and is expanded using different zero roots of the
spherical Bessel functions. The smoothness at the boundary is achieved through
careful construction of linear combinations of these basis functions.
"""

import math

import torch
from torch import nn


class SmoothBesselBasis(nn.Module):
    def __init__(self, r_max: float, max_n: int = 10):
        """Initialize smooth Bessel basis functions.

        Args:
            r_max (float): Cutoff radius beyond which basis functions go to zero
            max_n (int, optional): Maximum number of basis functions to use. Defaults to 10.
                                 Higher values give more complete basis but increase computation.

        The basis functions are constructed to satisfy several important properties:
        1. Orthogonality
        2. Smooth first and second derivatives at r_max (the cutoff)
        3. Complete basis in the range [0, r_max]
        """
        super(SmoothBesselBasis, self).__init__()
        self.max_n = max_n  # type: ignore
        n = torch.arange(0, max_n).float()[None, :]
        PI = math.pi
        SQRT2 = math.sqrt(2)

        # Compute normalization factors and coefficients
        fnr = (
            (-1) ** n
            * SQRT2
            * PI
            / r_max**1.5
            * (n + 1)
            * (n + 2)
            / torch.sqrt(2 * n**2 + 6 * n + 5)
        )
        en = n**2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)

        # Compute denominator terms recursively
        dn = [torch.tensor(1.0).float()]
        for i in range(1, max_n):
            dn.append(1 - en[0, i] / dn[-1])
        dn = torch.stack(dn)

        # Register buffers for efficient computation
        self.register_buffer("dn", dn)
        self.register_buffer("en", en)
        self.register_buffer("fnr_weights", fnr)
        self.register_buffer(
            "n_1_pi_cutoff",
            ((torch.arange(0, max_n).float() + 1) * PI / r_max).reshape(1, -1),
        )
        self.register_buffer(
            "n_2_pi_cutoff",
            ((torch.arange(0, max_n).float() + 2) * PI / r_max).reshape(1, -1),
        )
        self.register_buffer("r_max", torch.tensor(r_max))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate smooth Bessel basis functions at given radial distances.

        Args:
            x (torch.Tensor): Radial distances at which to evaluate basis functions

        Returns:
            torch.Tensor: Basis function values of shape (num_distances, max_n)
        """
        x_1 = x.unsqueeze(-1) * self.n_1_pi_cutoff
        x_2 = x.unsqueeze(-1) * self.n_2_pi_cutoff
        fnr = self.fnr_weights * (torch.sin(x_1) / x_1 + torch.sin(x_2) / x_2)

        # Build basis functions recursively
        gn = [fnr[:, 0]]
        for i in range(1, self.max_n):
            gn.append(
                1
                / torch.sqrt(self.dn[i])  # type: ignore
                * (fnr[:, i] + torch.sqrt(self.en[0, i] / self.dn[i - 1]) * gn[-1])  # type: ignore
            )
        return torch.transpose(torch.stack(gn), 1, 0)
