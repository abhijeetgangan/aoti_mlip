"""
Neural network layer implementations for M3GNet.

This module provides various layer types used throughout the M3GNet architecture:

- Linear layers with optional bias
- Activation layers (Sigmoid, Swish/SiLU, ReLU)
- Multi-layer perceptrons (MLP) with configurable activations
- Gated MLPs that combine two parallel networks with sigmoid gating

The layers are designed to be modular and reusable, with consistent interfaces
and flexible configuration options. The gated MLPs in particular are a key
component of M3GNet's message passing architecture.
"""

from typing import Union

import torch.nn as nn


class LinearLayer(nn.Module):
    """Simple linear transformation layer.

    Wraps torch.nn.Linear with a cleaner interface.

    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        bias: Whether to include a bias term (default: True)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)


class SigmoidLayer(nn.Module):
    """Linear layer followed by sigmoid activation.

    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        bias: Whether to include a bias term (default: True)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


class SwishLayer(nn.Module):
    """Linear layer followed by Swish/SiLU activation.

    Swish activation: x * sigmoid(x)

    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        bias: Whether to include a bias term (default: True)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        return x * self.sigmoid(x)


class ReLULayer(nn.Module):
    """Linear layer followed by ReLU activation.

    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        bias: Whether to include a bias term (default: True)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class GatedMLP(nn.Module):
    """Gated multi-layer perceptron combining two parallel networks.

    Implements a gated MLP that combines two parallel networks:
    - Main network g(x) with configurable activations
    - Gate network σ(x) with sigmoid final activation

    The final output is computed as: g(x) * σ(x)

    Args:
        in_dim: Input feature dimension
        out_dims: List of layer output dimensions
        activation: Activation function(s) for main network. Can be:
            - Single string/None applied to all layers
            - List of strings/None matching out_dims length
            Supported values: "swish", "sigmoid", None
        use_bias: Whether to include bias terms in linear layers
    """

    def __init__(
        self,
        in_dim: int,
        out_dims: list,
        activation: Union[list[Union[str, None]], str] = "swish",
        use_bias: bool = True,
    ):
        super().__init__()
        input_dim = in_dim
        if isinstance(activation, str) or activation is None:
            activation = [activation] * len(out_dims)
        else:
            assert len(activation) == len(out_dims), (
                "activation and out_dims must have the same length"
            )
        module_list_g = []
        for i in range(len(out_dims)):
            if activation[i] == "swish":
                module_list_g.append(SwishLayer(input_dim, out_dims[i], bias=use_bias))
            elif activation[i] == "sigmoid":
                module_list_g.append(SigmoidLayer(input_dim, out_dims[i], bias=use_bias))
            elif activation[i] is None:
                module_list_g.append(LinearLayer(input_dim, out_dims[i], bias=use_bias))
            input_dim = out_dims[i]
        module_list_sigma = []
        activation[-1] = "sigmoid"
        input_dim = in_dim
        for i in range(len(out_dims)):
            if activation[i] == "swish":
                module_list_sigma.append(SwishLayer(input_dim, out_dims[i], bias=use_bias))
            elif activation[i] == "sigmoid":
                module_list_sigma.append(SigmoidLayer(input_dim, out_dims[i], bias=use_bias))
            elif activation[i] is None:
                module_list_sigma.append(LinearLayer(input_dim, out_dims[i], bias=use_bias))
            else:
                raise NotImplementedError
            input_dim = out_dims[i]
        self.g = nn.Sequential(*module_list_g)
        self.sigma = nn.Sequential(*module_list_sigma)

    def forward(self, x):
        return self.g(x) * self.sigma(x)


class MLP(nn.Module):
    """Standard multi-layer perceptron with configurable activations.

    Args:
        in_dim: Input feature dimension
        out_dims: List of layer output dimensions
        activation: Activation function(s) to use. Can be:
            - Single string/None applied to all layers
            - List of strings/None matching out_dims length
            Supported values: "swish", "sigmoid", None
        use_bias: Whether to include bias terms in linear layers
    """

    def __init__(
        self,
        in_dim: int,
        out_dims: list,
        activation: Union[list[Union[str, None]], str, None] = "swish",
        use_bias: bool = True,
    ):
        super().__init__()
        input_dim = in_dim
        if isinstance(activation, str) or activation is None:
            activation = [activation] * len(out_dims)
        else:
            assert len(activation) == len(out_dims), (
                "activation and out_dims must have the same length"
            )
        module_list = []
        for i in range(len(out_dims)):
            if activation[i] == "swish":
                module_list.append(SwishLayer(input_dim, out_dims[i], bias=use_bias))
            elif activation[i] == "sigmoid":
                module_list.append(SigmoidLayer(input_dim, out_dims[i], bias=use_bias))
            elif activation[i] is None:
                module_list.append(LinearLayer(input_dim, out_dims[i], bias=use_bias))
            else:
                raise NotImplementedError
            input_dim = out_dims[i]
        self.mlp = nn.Sequential(*module_list)

    def forward(self, x):
        return self.mlp(x)
