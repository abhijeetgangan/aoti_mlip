import importlib.metadata
from typing import Optional

import packaging.version
import torch

_ALL_PKGS = importlib.metadata.packages_distributions()


def get_version_safe(module_name: str) -> Optional[str]:
    """Safely get the version of an installed package based on its module name.

    Args:
        module_name: name of the module to get version for

    Returns:
        version string if package is found, None otherwise
    """
    try:
        if module_name in _ALL_PKGS:
            module_name = _ALL_PKGS[module_name][0]
        return importlib.metadata.version(module_name)
    except importlib.metadata.PackageNotFoundError:
        return None


_VERSION = get_version_safe(torch.__name__)
assert _VERSION is not None, "PyTorch is not installed"
_TORCH_VERSION = packaging.version.parse(_VERSION)
_TORCH_GE_2_8 = packaging.version.parse(_TORCH_VERSION.base_version) >= packaging.version.parse(
    "2.8"
)
_TORCH_GE_2_9 = packaging.version.parse(_TORCH_VERSION.base_version) >= packaging.version.parse(
    "2.9"
)


def check_pt2_compile_compatibility():
    assert _TORCH_GE_2_8, (
        f"PyTorch >= 2.8 required for PT2 compilation functionality, but {_TORCH_VERSION} found."
    )
