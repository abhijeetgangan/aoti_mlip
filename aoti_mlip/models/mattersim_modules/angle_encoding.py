"""
Spherical harmonics implementation for angle encoding in neural networks.

References:
    - Implementation based on: https://github.com/akirasosa/pytorch-dimenet
    - Theory from: https://arxiv.org/abs/2003.03123 (DimeNet paper)

This module provides efficient computation of real spherical harmonics up to l=3,
which are used to encode angular information in message passing neural networks.
The spherical harmonics provide a complete basis for representing functions on
the sphere and are particularly useful for describing angular dependencies in
molecular systems.
"""

import math

import torch
import torch.nn as nn


def _spherical_harmonics(
    lmax: int, x: torch.Tensor, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Compute real spherical harmonics up to order lmax for input x = cos(θ).

    Implements analytical forms of real spherical harmonics Y_l^m evaluated at
    cos(θ) up to l=3. Each order l contains the m=0 term only, giving the
    Legendre polynomials P_l(cos θ).

    Args:
        lmax: Maximum order of spherical harmonics (0-3)
        x: Input tensor containing cos(θ) values
        device: Device to place tensors on
        dtype: Data type to use

    Returns:
        torch.Tensor: Stacked tensor of spherical harmonics values [Y_0^0, Y_1^0, ...]

    Raises:
        ValueError: If lmax > 3 is requested
    """
    # l = 0 term (constant)
    sh_0_0 = torch.ones_like(x, device=device, dtype=dtype) * 0.5 * math.sqrt(1.0 / math.pi)
    if lmax == 0:
        return torch.stack([sh_0_0], dim=-1)

    # l = 1 term (linear)
    sh_1_1 = math.sqrt(3.0 / (4.0 * math.pi)) * x
    if lmax == 1:
        return torch.stack([sh_0_0, sh_1_1], dim=-1)

    # l = 2 term (quadratic)
    sh_2_2 = math.sqrt(5.0 / (16.0 * math.pi)) * (3.0 * x**2 - 1.0)
    if lmax == 2:
        return torch.stack([sh_0_0, sh_1_1, sh_2_2], dim=-1)

    # l = 3 term (cubic)
    sh_3_3 = math.sqrt(7.0 / (16.0 * math.pi)) * x * (5.0 * x**2 - 3.0)
    if lmax == 3:
        return torch.stack([sh_0_0, sh_1_1, sh_2_2, sh_3_3], dim=-1)

    raise ValueError("lmax must be less than 4")


class SphericalBasisLayer(nn.Module):
    """Layer that computes spherical basis functions for encoding 3D geometric information.

    This layer combines radial basis functions (RBFs) and spherical harmonics to create a complete
    basis for representing 3D geometric features in molecular systems. The basis functions
    are particularly useful for encoding distances and angles between atoms.

    The layer implements:
    1. Radial basis functions (RBFs) up to order max_n using analytical forms
    2. Spherical harmonics up to order max_l for angular dependencies
    3. Combination of RBFs and spherical harmonics to form the final basis

    Args:
        max_n (int): Maximum order of radial basis functions (1-4)
        max_l (int): Maximum order of spherical harmonics (1-4)
        cutoff (float): Radial cutoff distance in Angstroms
        device (torch.device): Device to place tensors on
        dtype (torch.dtype): Data type to use for computations

    References:
        - DimeNet: https://arxiv.org/abs/2003.03123
        - M3GNet: https://www.nature.com/articles/s43588-022-00349-3
    """

    def __init__(self, max_n, max_l, cutoff, device: torch.device, dtype: torch.dtype):
        super(SphericalBasisLayer, self).__init__()

        assert max_l <= 4, "lmax must be less than 5"
        assert max_n <= 4, "max_n must be less than 5"

        self.max_n = max_n
        self.max_l = max_l
        self.cutoff = cutoff

        # Normalization factor for the basis functions
        self.register_buffer("factor", torch.sqrt(torch.tensor(2.0 / (self.cutoff**3))))

        # Initialize coefficient tensors for the radial basis functions
        # Shape: [4, 9, 4] to store coefficients for different orders
        self.coef = torch.zeros(4, 9, 4, dtype=dtype, device=device)

        # Coefficients for n=0 (first order)
        self.coef[0, 0, :] = torch.tensor(
            [
                3.14159274101257,
                6.28318548202515,
                9.42477798461914,
                12.5663709640503,
            ],
            device=device,
            dtype=dtype,
        )

        # Coefficients for n=1 (second order)
        self.coef[1, :4, :] = torch.tensor(
            [
                [
                    -1.02446483277785,
                    -1.00834335996107,
                    -1.00419641763893,
                    -1.00252381898662,
                ],
                [
                    4.49340963363647,
                    7.7252516746521,
                    10.9041213989258,
                    14.0661935806274,
                ],  # noqa: E501
                [
                    0.22799275301076,
                    0.130525632358311,
                    0.092093290316619,
                    0.0712718627992818,
                ],
                [
                    4.49340963363647,
                    7.7252516746521,
                    10.9041213989258,
                    14.0661935806274,
                ],  # noqa: E501
            ],
            device=device,
            dtype=dtype,
        )

        # Coefficients for n=2 (third order)
        self.coef[2, :6, :] = torch.tensor(
            [
                [
                    -1.04807944170731,
                    -1.01861796359391,
                    -1.01002272174988,
                    -1.00628955560036,
                ],
                [
                    5.76345920562744,
                    9.09501171112061,
                    12.322940826416,
                    15.5146026611328,
                ],  # noqa: E501
                [
                    0.545547077361439,
                    0.335992298618515,
                    0.245888396928293,
                    0.194582402961821,
                ],
                [
                    5.76345920562744,
                    9.09501171112061,
                    12.322940826416,
                    15.5146026611328,
                ],  # noqa: E501
                [
                    0.0946561878721665,
                    0.0369424811413594,
                    0.0199537107571916,
                    0.0125418876146463,
                ],
                [
                    5.76345920562744,
                    9.09501171112061,
                    12.322940826416,
                    15.5146026611328,
                ],  # noqa: E501
            ],
            device=device,
            dtype=dtype,
        )

        # Coefficients for n=3 (fourth order)
        self.coef[3, :8, :] = torch.tensor(
            [
                [
                    1.06942831392075,
                    1.0292173312802,
                    1.01650804843248,
                    1.01069656069999,
                ],  # noqa: E501
                [
                    6.9879322052002,
                    10.4171180725098,
                    13.6980228424072,
                    16.9236221313477,
                ],  # noqa: E501
                [
                    0.918235852195231,
                    0.592803493701152,
                    0.445250264272671,
                    0.358326327374518,
                ],
                [
                    6.9879322052002,
                    10.4171180725098,
                    13.6980228424072,
                    16.9236221313477,
                ],  # noqa: E501
                [
                    0.328507713452024,
                    0.142266673367543,
                    0.0812617757677838,
                    0.0529328657590962,
                ],
                [
                    6.9879322052002,
                    10.4171180725098,
                    13.6980228424072,
                    16.9236221313477,
                ],  # noqa: E501
                [
                    0.0470107184508114,
                    0.0136570088173405,
                    0.0059323726279831,
                    0.00312775039221944,
                ],
                [
                    6.9879322052002,
                    10.4171180725098,
                    13.6980228424072,
                    16.9236221313477,
                ],  # noqa: E501
            ],
            device=device,
            dtype=dtype,
        )

    def forward(self, r, theta_val):
        """Compute the spherical basis functions.

        Args:
            r (torch.Tensor): Radial distances between atoms
            theta_val (torch.Tensor): Angular values between triplets of atoms

        Returns:
            torch.Tensor: Combined radial and angular basis functions
        """
        # Scale distances by cutoff
        r = r / self.cutoff

        # Initialize empty list for radial basis functions
        rbfs = []

        # Compute first order (n=0) radial basis functions
        for j in range(self.max_l):
            rbfs.append(torch.sin(self.coef[0, 0, j] * r) / r)

        # Add higher order terms if requested
        if self.max_n > 1:
            # Second order (n=1) terms
            for j in range(self.max_l):
                rbfs.append(
                    (
                        self.coef[1, 0, j] * r * torch.cos(self.coef[1, 1, j] * r)  # noqa: E501
                        + self.coef[1, 2, j] * torch.sin(self.coef[1, 3, j] * r)  # noqa: E501
                    )
                    / r**2
                )

            if self.max_n > 2:
                # Third order (n=2) terms
                for j in range(self.max_l):
                    rbfs.append(
                        (
                            self.coef[2, 0, j] * (r**2) * torch.sin(self.coef[2, 1, j] * r)
                            - self.coef[2, 2, j] * r * torch.cos(self.coef[2, 3, j] * r)  # noqa: E501
                            + self.coef[2, 4, j] * torch.sin(self.coef[2, 5, j] * r)  # noqa: E501
                        )
                        / (r**3)
                    )

                if self.max_n > 3:
                    # Fourth order (n=3) terms
                    for j in range(self.max_l):
                        rbfs.append(
                            (
                                self.coef[3, 0, j] * (r**3) * torch.cos(self.coef[3, 1, j] * r)
                                - self.coef[3, 2, j] * (r**2) * torch.sin(self.coef[3, 3, j] * r)
                                - self.coef[3, 4, j] * r * torch.cos(self.coef[3, 5, j] * r)
                                + self.coef[3, 6, j] * torch.sin(self.coef[3, 7, j] * r)  # noqa: E501
                            )
                            / r**4
                        )

        # Stack and normalize radial basis functions
        rbfs = torch.stack(rbfs, dim=-1)
        rbfs = rbfs * self.factor  # type: ignore

        # Compute spherical harmonics and combine with radial basis
        cbfs = _spherical_harmonics(self.max_l - 1, torch.cos(theta_val), rbfs.device, rbfs.dtype)
        cbfs = cbfs.repeat_interleave(self.max_n, dim=1)

        # Return combined radial and angular basis functions
        return rbfs * cbfs
