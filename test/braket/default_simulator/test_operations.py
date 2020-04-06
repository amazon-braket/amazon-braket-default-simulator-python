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

import braket.ir.jaqcd as instruction
import numpy as np
import pytest
from braket.default_simulator import operations
from braket.default_simulator.operation_helpers import check_unitary, pauli_eigenvalues

gate_testdata = [
    (instruction.H(target=13), [13], operations.Hadamard),
    (instruction.X(target=11), [11], operations.PauliX),
    (instruction.Y(target=10), [10], operations.PauliY),
    (instruction.Z(target=9), [9], operations.PauliZ),
    (instruction.CNot(target=9, control=11), [11, 9], operations.CX),
    (instruction.CY(target=10, control=7), [7, 10], operations.CY),
    (instruction.CZ(target=14, control=7), [7, 14], operations.CZ),
    (instruction.T(target=1), [1], operations.T),
    (instruction.S(target=2), [2], operations.S),
    (instruction.PhaseShift(target=2, angle=0.15), [2], operations.PhaseShift),
    (instruction.CPhaseShift(target=2, control=7, angle=0.15), [7, 2], operations.CPhaseShift,),
    (instruction.I(target=4), [4], operations.Identity),
    (instruction.Rx(target=5, angle=0.14), [5], operations.RotX),
    (instruction.Ry(target=6, angle=0.16), [6], operations.RotY),
    (instruction.Rz(target=3, angle=0.17), [3], operations.RotZ),
    (instruction.Swap(targets=[4, 3]), [4, 3], operations.Swap),
    (instruction.ZZ(targets=[2, 1], angle=0.17), [2, 1], operations.ZZ),
    (instruction.YY(targets=[2, 1], angle=0.17), [2, 1], operations.YY),
    (instruction.XX(targets=[2, 1], angle=0.17), [2, 1], operations.XX),
    (
        instruction.Unitary(targets=[4], matrix=[[[0, 0], [0, 1]], [[1, 0], [0, 0]]]),
        [4],
        operations.Unitary,
    ),
]

observable_testdata = [
    (operations.Hadamard([13]), [13], pauli_eigenvalues(1), True),
    (operations.PauliX([11]), [11], pauli_eigenvalues(1), True),
    (operations.PauliY([10]), [10], pauli_eigenvalues(1), True),
    (operations.PauliZ([9]), [9], pauli_eigenvalues(1), True),
    (operations.Identity([7]), [7], np.array([1, 1]), False),
    (
        operations.Hermitian([4], np.array([[1, 1 - 1j], [1 + 1j, -1]])),
        [4],
        [-np.sqrt(3), np.sqrt(3)],
        False,
    ),
]


@pytest.mark.parametrize("instruction, targets, operation_type", gate_testdata)
def test_gate_operation_matrix_is_unitary(instruction, targets, operation_type):
    check_unitary(operations.from_braket_instruction(instruction).matrix)


@pytest.mark.parametrize("instruction, targets, operation_type", gate_testdata)
def test_from_braket_instruction(instruction, targets, operation_type):
    operation_instance = operations.from_braket_instruction(instruction)
    assert isinstance(operation_instance, operation_type)
    assert operation_instance.targets == targets


@pytest.mark.xfail(raises=ValueError)
def test_from_braket_instruction_unsupported_instruction():
    operations.from_braket_instruction(instruction.XY(targets=[0, 1], angle=0.15))


@pytest.mark.parametrize("observable, targets, eigenvalues, is_standard", observable_testdata)
def test_observable_diagonalizing_matrix_unitary(observable, targets, eigenvalues, is_standard):
    if observable.diagonalizing_matrix is not None:
        check_unitary(observable.diagonalizing_matrix)


@pytest.mark.parametrize("observable, targets, eigenvalues, is_standard", observable_testdata)
def test_observable_targets(observable, targets, eigenvalues, is_standard):
    assert observable.targets == targets


@pytest.mark.parametrize("observable, targets, eigenvalues, is_standard", observable_testdata)
def test_observable_eigenvalues(observable, targets, eigenvalues, is_standard):
    assert np.allclose(observable.eigenvalues, eigenvalues)


@pytest.mark.parametrize("observable, targets, eigenvalues, is_standard", observable_testdata)
def test_observable_is_standard(observable, targets, eigenvalues, is_standard):
    assert observable.is_standard == is_standard


def test_observable_known_diagonalization():
    assert np.allclose(
        operations.Hadamard([0]).diagonalizing_matrix, operations.RotY([0], -np.pi / 4).matrix
    )
    assert np.allclose(operations.PauliX([0]).diagonalizing_matrix, operations.Hadamard([0]).matrix)
    y_diag_expected = np.linalg.multi_dot(
        [operations.Hadamard([0]).matrix, operations.S([0]).matrix, operations.PauliZ([0]).matrix]
    )
    assert np.allclose(operations.PauliY([0]).diagonalizing_matrix, y_diag_expected)
