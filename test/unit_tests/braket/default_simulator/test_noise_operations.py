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

from braket.default_simulator import noise_operations
from braket.default_simulator.operation_helpers import check_cptp, from_braket_instruction

testdata = [
    (instruction.BitFlip(target=5, probability=0.01), (5,), noise_operations.BitFlip),
    (instruction.PhaseFlip(target=6, probability=0.23), (6,), noise_operations.PhaseFlip),
    (instruction.Depolarizing(target=3, probability=0.45), (3,), noise_operations.Depolarizing),
    (
        instruction.TwoQubitDepolarizing(targets=[3, 4], probability=0.45),
        (
            3,
            4,
        ),
        noise_operations.TwoQubitDepolarizing,
    ),
    (
        instruction.TwoQubitDephasing(targets=[3, 4], probability=0.45),
        (
            3,
            4,
        ),
        noise_operations.TwoQubitDephasing,
    ),
    (
        instruction.AmplitudeDamping(target=3, gamma=0.67),
        (3,),
        noise_operations.AmplitudeDamping,
    ),
    (
        instruction.GeneralizedAmplitudeDamping(target=3, gamma=0.1, probability=0.67),
        (3,),
        noise_operations.GeneralizedAmplitudeDamping,
    ),
    (
        instruction.PauliChannel(target=5, probX=0.1, probY=0.2, probZ=0.3),
        (5,),
        noise_operations.PauliChannel,
    ),
    (instruction.PhaseDamping(target=0, gamma=0.89), (0,), noise_operations.PhaseDamping),
    (
        instruction.Kraus(
            targets=[4],
            matrices=[
                [[[0.8, 0], [0, 0]], [[0, 0], [0.8, 0]]],
                [[[0, 0], [0, 0.6]], [[0.6, 0], [0, 0]]],
            ],
        ),
        (4,),
        noise_operations.Kraus,
    ),
]


@pytest.mark.parametrize("instruction, targets, operation_type", testdata)
def test_gate_operation_matrix_is_CPTP(instruction, targets, operation_type):
    check_cptp(from_braket_instruction(instruction).matrices)


@pytest.mark.parametrize("instruction, targets, operation_type", testdata)
def test_from_braket_instruction(instruction, targets, operation_type):
    operation_instance = from_braket_instruction(instruction)
    assert isinstance(operation_instance, operation_type)
    assert operation_instance.targets == targets
