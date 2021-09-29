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

import functools
import math

import numpy as np
import pytest
from braket.ir.jaqcd import shared_models

from braket.default_simulator import gate_operations, observables, operation_helpers
from braket.default_simulator.operation_helpers import (
    check_cptp,
    check_hermitian,
    check_matrix_dimensions,
    check_unitary,
    from_braket_instruction,
    ir_matrix_to_ndarray,
)

z_matrix = np.array([[1, 0], [0, -1]])

invalid_dimension_matrices = [
    (np.array([[1]])),
    (np.array([1])),
    (np.array([0, 1, 2])),
    (np.array([[0, 1], [1, 2], [3, 4]])),
    (np.array([[0, 1, 2], [2, 3]])),
    (np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])),
]

invalid_unitary_matrices = [(np.array([[0, 1], [1, 1]])), (np.array([[1, 2], [3, 4]]))]

invalid_hermitian_matrices = [(np.array([[1, 0], [0, 1j]])), (np.array([[1, 2], [3, 4]]))]

invalid_CPTP_matrices = [[np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])]]

valid_CPTP_matrices = [
    [np.array([[1, 0], [0, 1]]) * np.sqrt(0.7), np.array([[0, 1], [1, 0]]) * np.sqrt(0.3)],
    [np.eye(4)],
]

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


def test_correct_eigenvalues_paulis():
    """Test the pauli_eigenvalues function for one qubit"""
    assert np.array_equal(operation_helpers.pauli_eigenvalues(1), np.diag(z_matrix))


def test_correct_eigenvalues_pauli_kronecker_products_two_qubits():
    """Test the pauli_eigenvalues function for two qubits"""
    assert np.array_equal(
        operation_helpers.pauli_eigenvalues(2), np.diag(np.kron(z_matrix, z_matrix))
    )


def test_correct_eigenvalues_pauli_kronecker_products_three_qubits():
    """Test the pauli_eigenvalues function for three qubits"""
    assert np.array_equal(
        operation_helpers.pauli_eigenvalues(3),
        np.diag(np.kron(z_matrix, np.kron(z_matrix, z_matrix))),
    )


def test_ir_matrix_to_ndarray():
    matrix = [[[0, 0], [1, 0]], [[1, 0], [0, 0]]]
    assert np.allclose(ir_matrix_to_ndarray(matrix), np.array([[0, 1], [1, 0]]))


@pytest.mark.parametrize("depth", list(range(1, 6)))
def test_cache_usage(depth):
    """Test that the right number of cachings have been executed after clearing the cache"""
    operation_helpers.pauli_eigenvalues.cache_clear()
    operation_helpers.pauli_eigenvalues(depth)
    assert (
        functools._CacheInfo(depth - 1, depth, 128, depth)
        == operation_helpers.pauli_eigenvalues.cache_info()
    )


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("matrix", invalid_dimension_matrices)
def test_check_matrix_dimensions_invalid_matrix(matrix):
    check_matrix_dimensions(matrix, (0,))


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("matrix", invalid_unitary_matrices)
def test_check_unitary_invalid_matrix(matrix):
    check_unitary(matrix)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("matrix", invalid_hermitian_matrices)
def test_check_hermitian_invalid_matrix(matrix):
    check_hermitian(matrix)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("matrices", invalid_CPTP_matrices)
def test_check_cptp_invalid_matrix(matrices):
    check_cptp(matrices)


@pytest.mark.parametrize("matrices", valid_CPTP_matrices)
def test_check_cptp(matrices):
    check_cptp(matrices)


@pytest.mark.xfail(raises=NotImplementedError)
def test_from_braket_instruction_unsupported_instruction():
    from_braket_instruction(shared_models.DoubleTarget(targets=[4, 3]))
