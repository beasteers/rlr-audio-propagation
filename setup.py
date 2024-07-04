from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import shutil

class CustomBuildExt(build_ext):
    def run(self):
        super().run()
        # Ensure the shared library is copied to the right location
        build_lib = self.build_lib
        target_dir = os.path.join(build_lib, 'RLRAudioPropagationPkg', 'libs', 'linux', 'x64')
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy('RLRAudioPropagationPkg/libs/linux/x64/libRLRAudioPropagation.so', target_dir)

ext_modules = [
    Pybind11Extension(
        "_rlr_audio_propagation",
        ["src/rlr_audio_bindings.cpp"],
        include_dirs=["RLRAudioPropagationPkg/headers"],  # Path to the headers directory
        libraries=["RLRAudioPropagation"],               # Name of the library without prefix and suffix
        library_dirs=["RLRAudioPropagationPkg/libs/linux/x64"],  # Path to the directory containing the shared library
        extra_link_args=['-Wl,-rpath,$ORIGIN/RLRAudioPropagationPkg/libs/linux/x64'],  # Set rpath
    ),
    Pybind11Extension(
        "_rlr_audio_propagation_v1",
        ["src/rlr_audio_bindings_v1.cpp"],
        include_dirs=["RLRAudioPropagationPkg/headers"],  # Path to the headers directory
        libraries=["RLRAudioPropagation"],               # Name of the library without prefix and suffix
        library_dirs=["RLRAudioPropagationPkg/libs/linux/x64"],  # Path to the directory containing the shared library
        extra_link_args=['-Wl,-rpath,$ORIGIN/RLRAudioPropagationPkg/libs/linux/x64'],  # Set rpath
    ),
]

setup(
    name="rlr_audio_propagation",
    version="0.0.1",
    description="Python bindings for RLRAudioPropagation",
    license="Attribution-NonCommercial 4.0 International",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    package_data={
        'RLRAudioPropagationPkg': ['libs/linux/x64/libRLRAudioPropagation.so']
    },
    packages=find_packages(),  # Automatically find packages in the directory
    include_package_data=True,  # Include package data files specified in package_data
)
