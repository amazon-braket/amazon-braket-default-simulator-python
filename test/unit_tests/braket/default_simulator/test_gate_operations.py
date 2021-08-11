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
import pytest

from braket.default_simulator import gate_operations
from braket.default_simulator.operation_helpers import check_unitary, from_braket_instruction

testdata = [
    (instruction.I(target=4), (4,), gate_operations.Identity),
    (instruction.H(target=13), (13,), gate_operations.Hadamard),
    (instruction.X(target=11), (11,), gate_operations.PauliX),
    (instruction.Y(target=10), (10,), gate_operations.PauliY),
    (instruction.Z(target=9), (9,), gate_operations.PauliZ),
    (instruction.CNot(target=9, control=11), (11, 9), gate_operations.CX),
    (instruction.CY(target=10, control=7), (7, 10), gate_operations.CY),
    (instruction.CZ(target=14, control=7), (7, 14), gate_operations.CZ),
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
