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

import distutils.cmd
import os
import subprocess
from distutils.command.install import install
from pathlib import Path

from setuptools import find_namespace_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("src/braket/default_simulator/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")


class InstallWithOQ3Command(distutils.cmd.Command):
    """A custom command to install OpenQASM 3 and build grammars"""

    description = "install OQ3"
    user_options = []

    def initialize_options(self):
        """Set default values for options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        """Run command."""
        curdir = os.getcwd()
        if not Path("antlr-4.9.2-complete.jar").is_file():
            subprocess.check_call(
                ["curl", "-O", "https://www.antlr.org/download/antlr-4.9.2-complete.jar"]
            )
        classpath = Path(
            f".:{curdir}",
            f"antlr-4.9.2-complete.jar:{os.environ.get('CLASSPATH', '')}",
        )
        antlr4 = (
            f'java -Xmx500M -cp "{Path(curdir, f"antlr-4.9.2-complete.jar:{classpath}")}" '
            f"org.antlr.v4.Tool"
        )

        os.chdir(Path("src", "braket", "default_simulator", "openqasm"))
        subprocess.check_call(
            [
                *antlr4.split(),
                "-Dlanguage=Python3",
                "-visitor",
                "BraketPragmas.g4",
                "-o",
                "dist",
            ]
        )
        os.chdir(Path("..", "..", "..", ".."))

        if not Path("openqasm").is_dir():
            subprocess.check_call(["git", "clone", "https://github.com/Qiskit/openqasm.git"])
        os.chdir(Path("openqasm", "source", "openqasm"))
        subprocess.check_call(["python", "-m", "pip", "install", "."])
        os.chdir(Path("..", "grammar"))

        subprocess.check_call(
            [
                *antlr4.split(),
                "-o",
                "openqasm_reference_parser",
                "-Dlanguage=Python3",
                "-visitor",
                "qasm3Lexer.g4",
                "qasm3Parser.g4",
            ]
        )

        subprocess.check_call(["pip", "install", "."])

        subprocess.check_call(
            [
                *antlr4.split(),
                "-o",
                Path("..", "openqasm", "openqasm3", "antlr"),
                "-Dlanguage=Python3",
                "-visitor",
                "qasm3Lexer.g4",
                "qasm3Parser.g4",
            ]
        )
        os.chdir(Path("..", "..", ".."))


setup(
    name="amazon-braket-default-simulator",
    version=version,
    license="Apache License 2.0",
    python_requires=">= 3.7",
    packages=find_namespace_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    package_data={"": ["*.g4"]},
    include_package_data=True,
    install_requires=[
        "numpy",
        "opt_einsum",
        "antlr4-python3-runtime==4.9.2",
        "amazon_braket_schemas",
        # uncomment to build locally
        # (
        #     "amazon-braket-schemas "
        #     "@ git+https://github.com/aws/amazon-braket-schemas-python.git"
        #     "@feature/openqasm-local-simulator"
        # ),
    ],
    entry_points={
        "braket.simulators": [
            "default = braket.default_simulator.state_vector_simulator:StateVectorSimulator",
            "braket_sv = braket.default_simulator.state_vector_simulator:StateVectorSimulator",
            "braket_dm = braket.default_simulator.density_matrix_simulator:DensityMatrixSimulator",
            (
                "braket_oq3_sv = "
                "braket.default_simulator.oq3_state_vector_simulator:OQ3StateVectorSimulator"
            ),
            (
                "braket_oq3_native_sv = "
                "braket.default_simulator.openqasm_state_vector_simulator:"
                "OpenQASMStateVectorSimulator"
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
            "pytest",
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
    cmdclass={
        "install_oq3": InstallWithOQ3Command,
    },
)
