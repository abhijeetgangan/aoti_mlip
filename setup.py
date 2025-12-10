import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "aoti_mlip.models.mattersim_modules.dataloader.threebody_indices",
        ["aoti_mlip/models/mattersim_modules/dataloader/threebody_indices.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_API_VERSION")],
    )
]

setup(ext_modules=cythonize(extensions))
