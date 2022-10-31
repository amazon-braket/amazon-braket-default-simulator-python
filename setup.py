# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


from setuptools import find_namespace_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("src/braket/default_simulator/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")


setup(
    name="amazon-braket-default-simulator",
    version=version,
    license="Apache License 2.0",
    python_requires=">= 3.7",
    packages=find_namespace_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    package_data={"": ["*.g4", "*.inc"]},
    include_package_data=True,
    install_requires=[
        "numpy",
        "opt_einsum",
        "pydantic==1.9.0",
        "scipy",
        # pinned for compatibility with strawberry fields
        "antlr4-python3-runtime==4.9.2",
        "amazon-braket-schemas>=1.12.0",
    ],
    entry_points={
        "braket.simulators": [
            "default = braket.default_simulator.state_vector_simulator:StateVectorSimulator",
            "braket_sv = braket.default_simulator.state_vector_simulator:StateVectorSimulator",
            "braket_dm = braket.default_simulator.density_matrix_simulator:DensityMatrixSimulator",
            (
                "braket_oq3_sv = "
                "braket.default_simulator.openqasm_state_vector_simulator:"
                "OpenQASMStateVectorSimulator"
            ),
            (
                "braket_oq3_dm = "
                "braket.default_simulator.openqasm_density_matrix_simulator:"
                "OpenQASMDensityMatrixSimulator"
            ),
            (
                "braket_ahs = "
                "braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator:"
                "RydbergAtomSimulator",
            ),
        ]
    },
    extras_require={
        "test": [
            "black",
            "coverage",
            "flake8",
            "isort",
            "pre-commit",
            "pylint",
            "pytest==7.1.2",
            "pytest-benchmark",
            "pytest-cov",
            "pytest-rerunfailures",
            "pytest-xdist",
            "sphinx",
            "sphinx-rtd-theme",
            "sphinxcontrib-apidoc",
            "tox",
        ]
    },
    url="https://github.com/aws/amazon-braket-default-simulator-python",
    author="Amazon Web Services",
    description=(
        "An open source quantum circuit simulator to be run locally with the Amazon Braket SDK"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="Amazon AWS Quantum",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
