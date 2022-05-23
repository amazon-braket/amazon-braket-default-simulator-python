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
import distutils.log
import os
import subprocess
from pathlib import Path

from setuptools import find_namespace_packages, setup
from setuptools.command.build_py import build_py

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("src/braket/default_simulator/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")


class InstallOQ3Command(distutils.cmd.Command):
    """A custom command to run Pylint on all Python source files."""

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
        if not Path("antlr-4.10-complete.jar").is_file():
            subprocess.check_call(
                ["curl", "-O", "https://www.antlr.org/download/antlr-4.10-complete.jar"]
            )
        classpath = Path(
            f".:{curdir}",
            f"antlr-4.10-complete.jar:{os.environ.get('CLASSPATH', '')}",
        )
        antlr4 = (
            f'java -Xmx500M -cp "{Path(curdir, f"antlr-4.10-complete.jar:{classpath}")}" '
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
        subprocess.check_call(["pip", "install", "-e", "."])

        if not Path("openqasm").is_dir():
            subprocess.check_call(["git", "clone", "https://github.com/Qiskit/openqasm.git"])
        os.chdir(Path("openqasm", "source", "grammar"))

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
        subprocess.check_call(["python", "-m", "pip", "install", "-e", "."])
        subprocess.check_call(["pip", "install", "-e", ".[tests]"])

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
        os.chdir(Path("..", "openqasm"))
        subprocess.check_call(["pip", "install", "-e", "."])
        subprocess.check_call(["pip", "install", "-e", ".[tests]"])
        os.chdir(Path("..", "..", ".."))


class BuildPyCommand(build_py):
    """Custom build command."""

    def run(self):
        self.run_command("install_oq3")
        build_py.run(self)


setup(
    name="amazon-braket-default-simulator",
    version=version,
    license="Apache License 2.0",
    python_requires=">= 3.7",
    packages=find_namespace_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "opt_einsum",
        "openqasm3",
        "antlr4-python3-runtime==4.10",
        (
            "amazon-braket-schemas "
            "@ git+https://github.com/aws/amazon-braket-schemas-python.git"
            "@feature/openqasm-local-simulator"
        ),
        (
            "amazon-braket-sdk "
            "@ git+https://github.com/aws/amazon-braket-sdk-python.git"
            "@openqasm-local-simulator"
        ),
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
        "install_oq3": InstallOQ3Command,
    },
)
