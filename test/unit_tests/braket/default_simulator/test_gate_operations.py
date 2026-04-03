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

import braket.ir.jaqcd as instruction
import numpy as np
import pytest

from braket.default_simulator import gate_operations
from braket.default_simulator.gate_operations import Measure, Reset
from braket.default_simulator.operation_helpers import check_unitary, from_braket_instruction
from braket.default_simulator.simulation_strategies.single_operation_strategy import (
    apply_operations,
)

testdata = [
    (instruction.I(target=4), (4,), gate_operations.Identity),
    (instruction.H(target=13), (13,), gate_operations.Hadamard),
    (instruction.X(target=11), (11,), gate_operations.PauliX),
    (instruction.Y(target=10), (10,), gate_operations.PauliY),
    (instruction.Z(target=9), (9,), gate_operations.PauliZ),
    (instruction.CNot(target=9, control=11), (11, 9), gate_operations.CX),
    (instruction.CV(target=9, control=11), (11, 9), gate_operations.CV),
    (instruction.CY(target=10, control=7), (7, 10), gate_operations.CY),
    (instruction.CZ(target=14, control=7), (7, 14), gate_operations.CZ),
    (instruction.ECR(targets=[4, 3]), (4, 3), gate_operations.ECR),
    (instruction.S(target=2), (2,), gate_operations.S),
    (instruction.Si(target=2), (2,), gate_operations.Si),
    (instruction.T(target=1), (1,), gate_operations.T),
    (instruction.Ti(target=1), (1,), gate_operations.Ti),
    (instruction.V(target=1), (1,), gate_operations.V),
    (instruction.Vi(target=1), (1,), gate_operations.Vi),
    (instruction.PhaseShift(target=2, angle=0.15), (2,), gate_operations.PhaseShift),
    (
        instruction.CPhaseShift(target=2, control=7, angle=0.15),
        (7, 2),
        gate_operations.CPhaseShift,
    ),
    (
        instruction.CPhaseShift00(target=2, control=7, angle=0.15),
        (7, 2),
        gate_operations.CPhaseShift00,
    ),
    (
        instruction.CPhaseShift01(target=2, control=7, angle=0.15),
        (7, 2),
        gate_operations.CPhaseShift01,
    ),
    (
        instruction.CPhaseShift10(target=2, control=7, angle=0.15),
        (7, 2),
        gate_operations.CPhaseShift10,
    ),
    (instruction.Rx(target=5, angle=0.14), (5,), gate_operations.RotX),
    (instruction.Ry(target=6, angle=0.16), (6,), gate_operations.RotY),
    (instruction.Rz(target=3, angle=0.17), (3,), gate_operations.RotZ),
    (instruction.Swap(targets=[4, 3]), (4, 3), gate_operations.Swap),
    (instruction.ISwap(targets=[4, 3]), (4, 3), gate_operations.ISwap),
    (instruction.PSwap(targets=[2, 1], angle=0.17), (2, 1), gate_operations.PSwap),
    (instruction.XY(targets=[2, 1], angle=0.17), (2, 1), gate_operations.XY),
    (instruction.XX(targets=[2, 1], angle=0.17), (2, 1), gate_operations.XX),
    (instruction.YY(targets=[2, 1], angle=0.17), (2, 1), gate_operations.YY),
    (instruction.ZZ(targets=[2, 1], angle=0.17), (2, 1), gate_operations.ZZ),
    (instruction.CCNot(target=9, controls=[13, 11]), (13, 11, 9), gate_operations.CCNot),
    (instruction.CSwap(targets=[9, 7], control=11), (11, 9, 7), gate_operations.CSwap),
    (
        instruction.Unitary(targets=[4], matrix=[[[0, 0], [0, 1]], [[1, 0], [0, 0]]]),
        (4,),
        gate_operations.Unitary,
    ),
]


@pytest.mark.parametrize("ir_instruction, targets, operation_type", testdata)
def test_gate_operation(ir_instruction, targets, operation_type):
    operation_instance = from_braket_instruction(ir_instruction)
    assert isinstance(operation_instance, operation_type)
    assert operation_instance.targets == targets
    check_unitary(operation_instance.matrix)


_s2 = 1 / np.sqrt(2)

measure_testdata = [
    # (operation, input_state, expected_output_state)
    (Measure([0], result=-1), np.array([_s2, _s2], dtype=complex), np.array([_s2, _s2], dtype=complex)),
    (Measure([0], result=0),  np.array([_s2, _s2], dtype=complex), np.array([1.0, 0.0], dtype=complex)),
    (Measure([0], result=1),  np.array([_s2, _s2], dtype=complex), np.array([0.0, 1.0], dtype=complex)),
    # Two-qubit: |00⟩+|01⟩+|10⟩+|11⟩, measure qubit 1 → 0; only |00⟩ and |10⟩ survive
    (Measure([1], result=0),  0.5 * np.ones(4, dtype=complex),     np.array([_s2, 0, _s2, 0], dtype=complex)),
]


@pytest.mark.parametrize("operation, input_state, expected", measure_testdata)
def test_measure_operation(operation, input_state, expected):
    result = operation.apply(input_state.copy())
    np.testing.assert_array_almost_equal(result, expected)


reset_testdata = [
    # (input_state, expected_output_state)
    (np.array([1.0, 0.0], dtype=complex), np.array([1.0, 0.0], dtype=complex)),
    (np.array([0.0, 1.0], dtype=complex), np.array([1.0, 0.0], dtype=complex)),
    (np.array([_s2,  _s2], dtype=complex), np.array([1.0, 0.0], dtype=complex)),
]


@pytest.mark.parametrize("input_state, expected", reset_testdata)
def test_reset_operation(input_state, expected):
    result = Reset([0]).apply(input_state.copy())
    np.testing.assert_array_almost_equal(result, expected)


def test_measure_invalid_result_raises():
    # only 0 and 1 are valid results for _base_matrix; -1 (unset) and anything else should raise
    for bad in (-1, 99):
        with pytest.raises(ValueError, match="Invalid measurement result"):
            Measure([0], result=bad)._base_matrix


def test_measure_multi_qubit_raises():
    with pytest.raises(ValueError, match="single target qubit"):
        Measure([0, 1], result=0).apply(0.5 * np.ones(4, dtype=complex))


def test_measure_zero_norm_state():
    # norm == 0 after projection — should not divide by zero
    m = Measure([0], result=1)
    state = np.array([1.0, 0.0], dtype=complex)  # already in |0⟩, projecting to |1⟩ → zero norm
    result = m.apply(state.copy())
    np.testing.assert_array_almost_equal(result, [0.0, 0.0])


def test_reset_zero_norm_state():
    # norm == 0 after reset — should not divide by zero
    r = Reset([0])
    state = np.zeros(2, dtype=complex)
    result = r.apply(state.copy())
    np.testing.assert_array_almost_equal(result, [0.0, 0.0])


def test_apply_operations_with_measure():
    # exercises the Measure/Reset branch in single_operation_strategy
    state = np.array([_s2, _s2], dtype=complex).reshape(2)
    ops = [Measure([0], result=0)]
    result = apply_operations(state, 1, ops)
    np.testing.assert_array_almost_equal(result.flatten(), [1.0, 0.0])


def test_apply_operations_with_reset():
    state = np.array([0.0, 1.0], dtype=complex).reshape(2)
    ops = [Reset([0])]
    result = apply_operations(state, 1, ops)
    np.testing.assert_array_almost_equal(result.flatten(), [1.0, 0.0])
