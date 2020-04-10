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
from braket.default_simulator import TensorProduct, gate_operations, observables
from braket.default_simulator.operation_helpers import check_unitary, pauli_eigenvalues

testdata = [
    (observables.Hadamard([13]), [13], pauli_eigenvalues(1), True),
    (observables.PauliX([11]), [11], pauli_eigenvalues(1), True),
    (observables.PauliY([10]), [10], pauli_eigenvalues(1), True),
    (observables.PauliZ([9]), [9], pauli_eigenvalues(1), True),
    (observables.Identity([7]), [7], np.array([1, 1]), False),
    (
        observables.Hermitian([4], np.array([[1, 1 - 1j], [1 + 1j, -1]])),
        [4],
        [-np.sqrt(3), np.sqrt(3)],
        False,
    ),
]

angle = -np.pi / 4
cos_component = np.cos(angle / 2)
sin_component = np.sin(angle / 2)
h_diag = np.array([[cos_component, -sin_component], [sin_component, cos_component]])
x_diag = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
y_diag = np.array([[1, -1j], [1, 1j]]) / np.sqrt(2)


@pytest.mark.parametrize("observable, targets, eigenvalues, is_standard", testdata)
def test_observable_diagonalizing_matrix_unitary(observable, targets, eigenvalues, is_standard):
    if observable.diagonalizing_matrix is not None:
        check_unitary(observable.diagonalizing_matrix)


@pytest.mark.parametrize("observable, targets, eigenvalues, is_standard", testdata)
def test_observable_targets(observable, targets, eigenvalues, is_standard):
    assert observable.targets == targets


@pytest.mark.parametrize("observable, targets, eigenvalues, is_standard", testdata)
def test_observable_eigenvalues(observable, targets, eigenvalues, is_standard):
    assert np.allclose(observable.eigenvalues, eigenvalues)


@pytest.mark.parametrize("observable, targets, eigenvalues, is_standard", testdata)
def test_observable_is_standard(observable, targets, eigenvalues, is_standard):
    assert observable.is_standard == is_standard


def test_observable_known_diagonalization():
    assert np.allclose(
        observables.Hadamard([0]).diagonalizing_matrix, gate_operations.RotY([0], -np.pi / 4).matrix
    )
    assert np.allclose(
        observables.PauliX([0]).diagonalizing_matrix, gate_operations.Hadamard([0]).matrix
    )
    y_diag_expected = np.linalg.multi_dot(
        [
            gate_operations.Hadamard([0]).matrix,
            gate_operations.S([0]).matrix,
            gate_operations.PauliZ([0]).matrix,
        ]
    )
    assert np.allclose(observables.PauliY([0]).diagonalizing_matrix, y_diag_expected)


def test_tensor_product_standard():
    tensor = TensorProduct(
        [
            observables.Hadamard([1]),
            observables.PauliX([3]),
            observables.PauliZ([7]),
            observables.PauliY([4]),
        ]
    )
    assert tensor.targets == [1, 3, 7, 4]
    assert not tensor.is_standard
    assert (tensor.eigenvalues == pauli_eigenvalues(4)).all()

    # Z ignored
    assert (tensor.diagonalizing_matrix == np.kron(np.kron(h_diag, x_diag), y_diag)).all()


def test_tensor_product_nonstandard():
    tensor = TensorProduct(
        [
            observables.Hadamard([1]),
            observables.Identity([5]),
            observables.PauliX([3]),
            observables.PauliZ([7]),
            observables.PauliY([4]),
        ]
    )
    assert tensor.targets == [1, 5, 3, 7, 4]

    eigenvalues = np.array(
        [
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
        ]
    )
    assert (tensor.eigenvalues == eigenvalues).all()

    # Test cached
    assert (tensor.eigenvalues == eigenvalues).all()

    # Both Identity and Z ignored
    assert (tensor.diagonalizing_matrix == np.kron(np.kron(h_diag, x_diag), y_diag)).all()
