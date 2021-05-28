# Copyright 2019-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import numpy as np
import pytest

from braket.default_simulator import gate_operations
from braket.default_simulator.unitary_matrix_simulation import UnitaryMatrixSimulation

evolve_testdata = [
    ([gate_operations.Hadamard([0])], 1, gate_operations.Hadamard([]).matrix),
    ([gate_operations.CX([0, 1])], 2, gate_operations.CX([]).matrix),
    ([gate_operations.CCNot([0, 1, 2])], 3, gate_operations.CCNot([]).matrix),
    (
        [gate_operations.PauliY([0]), gate_operations.CX([1, 2])],
        3,
        np.kron(gate_operations.CX([]).matrix, gate_operations.PauliY([]).matrix),
    ),
    (
        [gate_operations.Hadamard([0]), gate_operations.CX([0, 1]), gate_operations.CX([1, 2])],
        3,
        np.dot(
            np.kron(gate_operations.CX([]).matrix, np.eye(2)),
            np.dot(
                np.kron(np.eye(2), gate_operations.CX([]).matrix),
                np.kron(np.eye(4), gate_operations.Hadamard([]).matrix),
            ),
        ),
    ),
    (
        [
            gate_operations.PauliX([0]),
            gate_operations.PauliY([0]),
            gate_operations.PauliY([1]),
            gate_operations.PauliZ([1]),
            gate_operations.PauliZ([2]),
            gate_operations.PauliX([2]),
        ],
        3,
        np.kron(
            np.kron(
                np.dot(gate_operations.PauliX([]).matrix, gate_operations.PauliZ([]).matrix),
                np.dot(gate_operations.PauliZ([]).matrix, gate_operations.PauliY([]).matrix),
            ),
            np.dot(gate_operations.PauliY([]).matrix, gate_operations.PauliX([]).matrix),
        ),
    ),
]


@pytest.mark.parametrize("instructions, qubit_count, unitary_matrix", evolve_testdata)
def test_simulation_simple_circuits(instructions, qubit_count, unitary_matrix):
    simulation = UnitaryMatrixSimulation(qubit_count, 0)
    simulation.evolve(instructions)
    assert np.allclose(unitary_matrix, simulation.unitary_matrix)
