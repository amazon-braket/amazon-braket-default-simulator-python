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

import math

import numpy as np
import pytest
from braket.default_simulator import gate_operations, observables
from braket.default_simulator.simulation_strategies.simulation_helpers import get_matrix

gate_testdata = [
    gate_operations.Identity([0]),
    gate_operations.Hadamard([0]),
    gate_operations.PauliX([0]),
    gate_operations.PauliY([0]),
    gate_operations.PauliZ([0]),
    gate_operations.CX([0, 1]),
    gate_operations.CY([0, 1]),
    gate_operations.CZ([0, 1]),
    gate_operations.S([0]),
    gate_operations.Si([0]),
    gate_operations.T([0]),
    gate_operations.Ti([0]),
    gate_operations.V([0]),
    gate_operations.Vi([0]),
    gate_operations.PhaseShift([0], math.pi),
    gate_operations.CPhaseShift([0, 1], math.pi),
    gate_operations.CPhaseShift00([0, 1], math.pi),
    gate_operations.CPhaseShift01([0, 1], math.pi),
    gate_operations.CPhaseShift10([0, 1], math.pi),
    gate_operations.RotX([0], math.pi),
    gate_operations.RotY([0], math.pi),
    gate_operations.RotZ([0], math.pi),
    gate_operations.Swap([0, 1]),
    gate_operations.ISwap([0, 1]),
    gate_operations.PSwap([0, 1], math.pi),
    gate_operations.XY([0, 1], math.pi),
    gate_operations.XX([0, 1], math.pi),
    gate_operations.YY([0, 1], math.pi),
    gate_operations.ZZ([0, 1], math.pi),
    gate_operations.CCNot([0, 1, 2]),
    gate_operations.CSwap([0, 1, 2]),
    gate_operations.Unitary([0], [[0, 1j], [1j, 0]]),
]

observable_testdata = [
    observables.Identity([0]),
    observables.PauliX([0]),
    observables.PauliY([0]),
    observables.PauliZ([0]),
    observables.Hadamard([0]),
    observables.Hermitian(np.array([[1, 1 - 1j], [1 + 1j, -1]])),
]


@pytest.mark.parametrize("operation", gate_testdata)
def test_get_matrix_gate_operation(operation):
    assert np.allclose(get_matrix(operation), operation.matrix)


@pytest.mark.parametrize("operation", observable_testdata)
def test_get_matrix_observable(operation):
    matrix = get_matrix(operation)
    if matrix is not None:
        assert np.allclose(matrix, operation.diagonalizing_matrix)
    else:
        assert operation.diagonalizing_matrix is None
