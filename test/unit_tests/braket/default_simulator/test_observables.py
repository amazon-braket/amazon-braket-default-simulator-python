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

import numpy as np
import pytest

from braket.default_simulator import gate_operations, observables
from braket.default_simulator.operation_helpers import pauli_eigenvalues

testdata_1q = [
    (observables.Hadamard, [], gate_operations.RotY([0], -np.pi / 4), pauli_eigenvalues(1), True),
    (observables.PauliX, [], gate_operations.Hadamard([0]), pauli_eigenvalues(1), True),
    (
        observables.PauliY,
        [],
        gate_operations.Unitary([0], np.array([[1, -1j], [1, 1j]]) / np.sqrt(2)),
        pauli_eigenvalues(1),
        True,
    ),
    (observables.PauliZ, [], None, pauli_eigenvalues(1), True),
    (observables.Identity, [], None, np.array([1, 1]), False),
    (
        observables.Hermitian,
        [np.array([[1, 1 - 1j], [1 + 1j, -1]])],
        gate_operations.Unitary(
            [0],
            np.array(
                [
                    [-0.45970084 + 0.0j, 0.62796303 - 0.62796303j],
                    [-0.88807383 - 0.0j, -0.32505758 + 0.32505758j],
                ]
            ),
        ),
        [-np.sqrt(3), np.sqrt(3)],
        False,
    ),
]

involutory = [
    observables.Hadamard([0]),
    observables.PauliX([0]),
    observables.PauliY([0]),
    observables.PauliZ([0]),
]

predefined_observables_invalid_targets = [
    (observables.Hadamard, [0, 1]),
    (observables.PauliX, [0, 1]),
    (observables.PauliY, [0, 1]),
    (observables.PauliZ, [0, 1]),
    (observables.Identity, [0, 1]),
]

angle = -np.pi / 4
cos_component = np.cos(angle / 2)
sin_component = np.sin(angle / 2)
h_diag = np.array([[cos_component, -sin_component], [sin_component, cos_component]])
x_diag = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
y_diag = np.array([[1, -1j], [1, 1j]]) / np.sqrt(2)


@pytest.mark.parametrize(
    "obs_class, extra_args, expected_gates, eigenvalues, is_standard", testdata_1q
)
def test_observable_properties_single_qubit(
    obs_class, extra_args, expected_gates, eigenvalues, is_standard
):
    num_qubits = 50
    measured_qubits = (20,)
    observable = obs_class(*extra_args, targets=list(measured_qubits))
    actual_gates = observable.diagonalizing_gates(num_qubits)
    for gate in actual_gates:
        assert np.allclose(gate.matrix, expected_gates.matrix)
    assert observable.measured_qubits == measured_qubits
    if actual_gates:
        assert observable.targets == measured_qubits
    else:
        assert observable.targets == ()
    assert np.allclose(observable.eigenvalues, eigenvalues)
    assert observable.is_standard == is_standard


@pytest.mark.parametrize(
    "obs_class, extra_args, expected_gates, eigenvalues, is_standard", testdata_1q
)
def test_observable_properties_all_qubits(
    obs_class, extra_args, expected_gates, eigenvalues, is_standard
):
    num_qubits = 10  # Smaller number so tests run faster
    observable = obs_class(*extra_args)
    actual_gates = observable.diagonalizing_gates(num_qubits)
    for gate in actual_gates:
        assert np.allclose(gate.matrix, expected_gates.matrix)
    if actual_gates:
        assert len(actual_gates) == num_qubits
    assert not observable.measured_qubits
    assert not observable.targets


@pytest.mark.parametrize("obs", involutory)
def test_involutory_powers(obs):
    for power in range(0, 10, 2):
        assert (obs ** power).__class__ is observables.Identity
    for power in range(1, 11, 2):
        assert (obs ** power).__class__ is obs.__class__


@pytest.mark.xfail(raises=TypeError)
@pytest.mark.parametrize(
    "obs_class, extra_args, expected_gates, eigenvalues, is_standard", testdata_1q
)
def test_power_non_integer(obs_class, extra_args, expected_gates, eigenvalues, is_standard):
    obs_class(*extra_args, targets=[0]) ** np.pi


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("observable, targets", predefined_observables_invalid_targets)
def test_observable_predefined_invalid_targets(observable, targets):
    observable(targets)


@pytest.mark.xfail(raises=ValueError)
def test_hermitian_invalid_none_target():
    observables.Hermitian(np.eye(4), None)


@pytest.mark.xfail(raises=ValueError)
def test_hermitian_multi_qubit_fix_qubit():
    observables.Hermitian(np.eye(4), [0, 1]).fix_qubit(0)


def test_observable_known_diagonalization():
    y_diag_expected = np.linalg.multi_dot(
        [
            gate_operations.Hadamard([0]).matrix,
            gate_operations.S([0]).matrix,
            gate_operations.PauliZ([0]).matrix,
        ]
    )
    assert np.allclose(observables.PauliY([0]).diagonalizing_gates()[0].matrix, y_diag_expected)


def test_tensor_product_standard():
    tensor = observables.TensorProduct(
        [
            observables.Hadamard([1]),
            observables.PauliX([3]),
            observables.PauliZ([7]),
            observables.PauliY([4]),
        ]
    )
    assert tensor.targets == (1, 3, 4)
    assert tensor.measured_qubits == (1, 3, 7, 4)
    assert (tensor.eigenvalues == pauli_eigenvalues(4)).all()
    assert not tensor.is_standard

    actual_gates = [gate.matrix for gate in tensor.diagonalizing_gates()]
    # Z ignored
    assert len(actual_gates) == 3
    assert np.allclose(actual_gates[0], observables.Hadamard([0]).diagonalizing_gates()[0].matrix)
    assert np.allclose(actual_gates[1], observables.PauliX([0]).diagonalizing_gates()[0].matrix)
    assert np.allclose(actual_gates[2], observables.PauliY([0]).diagonalizing_gates()[0].matrix)


def test_tensor_product_nonstandard():
    tensor = observables.TensorProduct(
        [
            observables.Hadamard([1]),
            observables.Identity([5]),
            observables.PauliX([3]),
            observables.PauliZ([7]),
            observables.PauliY([4]),
        ]
    )
    assert tensor.targets == (1, 3, 4)
    assert tensor.measured_qubits == (1, 5, 3, 7, 4)

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

    actual_gates = [gate.matrix for gate in tensor.diagonalizing_gates()]
    # Both Identity and Z ignored
    assert len(actual_gates) == 3
    assert np.allclose(actual_gates[0], observables.Hadamard([0]).diagonalizing_gates()[0].matrix)
    assert np.allclose(actual_gates[1], observables.PauliX([0]).diagonalizing_gates()[0].matrix)
    assert np.allclose(actual_gates[2], observables.PauliY([0]).diagonalizing_gates()[0].matrix)


@pytest.mark.xfail(raises=ValueError)
def test_tensor_product_one_component():
    observables.TensorProduct([observables.Hadamard([2])])


@pytest.mark.xfail(raises=TypeError)
def test_tensor_product_fix_qubit():
    observables.TensorProduct([observables.Hadamard([0]), observables.Hadamard([1])]).fix_qubit(0)
