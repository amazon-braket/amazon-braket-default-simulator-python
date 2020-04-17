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

import functools
import itertools

import numpy as np
import pytest
from braket.default_simulator import operation_helpers
from braket.default_simulator.operation_helpers import (
    check_hermitian,
    check_matrix_dimensions,
    check_unitary,
    ir_matrix_to_ndarray,
)

x_matrix = np.array([[0, 1], [1, 0]])
y_matrix = np.array([[0, -1j], [1j, 0]])
z_matrix = np.array([[1, 0], [0, -1]])
h_matrix = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

standard_observables = [x_matrix, y_matrix, z_matrix, h_matrix]

matrix_pairs = [
    np.kron(x, y) for x, y in list(itertools.product(standard_observables, standard_observables))
]

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


@pytest.mark.parametrize("pauli", standard_observables)
def test_correct_eigenvalues_paulis(pauli):
    """Test the pauli_eigenvalues function for one qubit"""
    assert np.array_equal(operation_helpers.pauli_eigenvalues(1), np.diag(z_matrix))


@pytest.mark.parametrize("pauli_product", matrix_pairs)
def test_correct_eigenvalues_pauli_kronecker_products_two_qubits(pauli_product):
    """Test the pauli_eigenvalues function for two qubits"""
    assert np.array_equal(
        operation_helpers.pauli_eigenvalues(2), np.diag(np.kron(z_matrix, z_matrix))
    )


@pytest.mark.parametrize("pauli_product", matrix_pairs)
def test_correct_eigenvalues_pauli_kronecker_products_three_qubits(pauli_product):
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
