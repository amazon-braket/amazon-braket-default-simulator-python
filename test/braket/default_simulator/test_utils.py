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
from braket.default_simulator import utils

X_MATRIX = np.array([[0, 1], [1, 0]])
Y_MATRIX = np.array([[0, -1j], [1j, 0]])
Z_MATRIX = np.array([[1, 0], [0, -1]])
H_MATRIX = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

STANDARD_OBSERVABLES = [X_MATRIX, Y_MATRIX, Z_MATRIX, H_MATRIX]

MATRIX_PAIRS = [
    np.kron(x, y) for x, y in list(itertools.product(STANDARD_OBSERVABLES, STANDARD_OBSERVABLES))
]


@pytest.mark.parametrize("pauli", STANDARD_OBSERVABLES)
def test_correct_eigenvalues_paulis(pauli):
    """Test the pauli_eigenvalues function for one qubit"""
    assert np.array_equal(utils.pauli_eigenvalues(1), np.diag(Z_MATRIX))


@pytest.mark.parametrize("pauli_product", MATRIX_PAIRS)
def test_correct_eigenvalues_pauli_kronecker_products_two_qubits(pauli_product):
    """Test the pauli_eigenvalues function for two qubits"""
    assert np.array_equal(utils.pauli_eigenvalues(2), np.diag(np.kron(Z_MATRIX, Z_MATRIX)))


@pytest.mark.parametrize("pauli_product", MATRIX_PAIRS)
def test_correct_eigenvalues_pauli_kronecker_products_three_qubits(pauli_product):
    """Test the pauli_eigenvalues function for three qubits"""
    assert np.array_equal(
        utils.pauli_eigenvalues(3), np.diag(np.kron(Z_MATRIX, np.kron(Z_MATRIX, Z_MATRIX)))
    )


@pytest.mark.parametrize("depth", list(range(1, 6)))
def test_cache_usage(depth):
    """Test that the right number of cachings have been executed after clearing the cache"""
    utils.pauli_eigenvalues.cache_clear()
    utils.pauli_eigenvalues(depth)
    assert (
        functools._CacheInfo(depth - 1, depth, 128, depth) == utils.pauli_eigenvalues.cache_info()
    )
