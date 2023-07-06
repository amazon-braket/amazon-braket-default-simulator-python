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
    (
        instruction.BitFlip(target=5, probability=0.01),
        (5,),
        noise_operations.BitFlip,
        None,
        None,
        0.01,
    ),
    (
        instruction.PhaseFlip(target=6, probability=0.23),
        (6,),
        noise_operations.PhaseFlip,
        None,
        None,
        0.23,
    ),
    (
        instruction.Depolarizing(target=3, probability=0.45),
        (3,),
        noise_operations.Depolarizing,
        None,
        None,
        0.45,
    ),
    (
        instruction.TwoQubitDepolarizing(targets=[3, 4], probability=0.45),
        (
            3,
            4,
        ),
        noise_operations.TwoQubitDepolarizing,
        None,
        None,
        0.45,
    ),
    (
        instruction.TwoQubitDephasing(targets=[3, 4], probability=0.45),
        (
            3,
            4,
        ),
        noise_operations.TwoQubitDephasing,
        None,
        None,
        0.45,
    ),
    (
        instruction.AmplitudeDamping(target=3, gamma=0.67),
        (3,),
        noise_operations.AmplitudeDamping,
        0.67,
        None,
        None,
    ),
    (
        instruction.GeneralizedAmplitudeDamping(target=3, gamma=0.1, probability=0.67),
        (3,),
        noise_operations.GeneralizedAmplitudeDamping,
        0.1,
        None,
        0.67,
    ),
    (
        instruction.PauliChannel(target=5, probX=0.1, probY=0.2, probZ=0.3),
        (5,),
        noise_operations.PauliChannel,
        None,
        [0.1, 0.2, 0.3],
        None,
    ),
    (
        instruction.MultiQubitPauliChannel(
            targets=[5, 6], probabilities={"XX": 0.01, "YY": 0.02, "XZ": 0.03}
        ),
        (5, 6),
        noise_operations.TwoQubitPauliChannel,
        None,
        None,
        None,
    ),
    (
        instruction.PhaseDamping(target=0, gamma=0.89),
        (0,),
        noise_operations.PhaseDamping,
        0.89,
        None,
        None,
    ),
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
        None,
        None,
        None,
    ),
]


@pytest.mark.parametrize(
    "instruction, targets, operation_type, gamma, probabilities, probability", testdata
)
def test_gate_operation_matrix_is_CPTP(
    instruction, targets, operation_type, gamma, probabilities, probability
):
    check_cptp(from_braket_instruction(instruction).matrices)


@pytest.mark.parametrize(
    "instruction, targets, operation_type, gamma, probabilities, probability", testdata
)
def test_from_braket_instruction(
    instruction,
    targets,
    operation_type,
    gamma,
    probabilities,
    probability,
):
    operation_instance = from_braket_instruction(instruction)
    assert isinstance(operation_instance, operation_type)
    assert operation_instance.targets == targets
    if gamma:
        assert operation_instance.gamma == gamma
    if probabilities:
        assert operation_instance.probabilities == probabilities
    if probability:
        assert operation_instance.probability == probability
