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
from braket.default_simulator import TensorProduct, operations
from braket.default_simulator.operation_helpers import pauli_eigenvalues

angle = -np.pi / 4
cos_component = np.cos(angle / 2)
sin_component = np.sin(angle / 2)
h_diag = np.array([[cos_component, -sin_component], [sin_component, cos_component]])
x_diag = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
y_diag = np.array([[1, -1j], [1, 1j]]) / np.sqrt(2)


def test_tensor_product_standard():
    tensor = TensorProduct(
        [
            operations.Hadamard([1]),
            operations.PauliX([3]),
            operations.PauliZ([7]),
            operations.PauliY([4]),
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
            operations.Hadamard([1]),
            operations.Identity([5]),
            operations.PauliX([3]),
            operations.PauliZ([7]),
            operations.PauliY([4]),
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
