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

import cmath
import json
import sys
from collections import Counter, namedtuple

import numpy as np
import pytest
from braket.device_schema.simulators import (
    GateModelSimulatorDeviceCapabilities,
    GateModelSimulatorDeviceParameters,
)
from braket.ir.jaqcd import Program
from braket.task_result import AdditionalMetadata, ResultTypeValue, TaskMetadata

from braket.default_simulator.density_matrix_simulator import DensityMatrixSimulator

CircuitData = namedtuple("CircuitData", "circuit_ir probability_zero")


invalid_ir_result_types = [
    {"type": "statevector"},
    {"type": "amplitude", "states": ["11"]},
]


@pytest.fixture
def noisy_circuit_2_qubit():
    return Program.parse_raw(
        json.dumps(
            {
                "instructions": [
                    {"type": "x", "target": 0},
                    {"type": "x", "target": 1},
                    {"type": "bit_flip", "target": 1, "probability": 0.1},
                ]
            }
        )
    )


@pytest.fixture
def grcs_8_qubit():
    with open("test/resources/grcs_8.json") as circuit_file:
        data = json.load(circuit_file)
        return CircuitData(Program.parse_raw(json.dumps(data["ir"])), data["probability_zero"])


@pytest.fixture
def bell_ir():
    return Program.parse_raw(
        json.dumps(
            {
                "instructions": [
                    {"type": "h", "target": 0},
                    {"type": "cnot", "target": 1, "control": 0},
                ]
            }
        )
    )


@pytest.fixture
def bell_ir_with_result():
    def _bell_ir_with_result(targets=None):
        return Program.parse_raw(
            json.dumps(
                {
                    "instructions": [
                        {"type": "h", "target": 0},
                        {"type": "cnot", "target": 1, "control": 0},
                    ],
                    "results": [{"type": "expectation", "observable": ["x"], "targets": targets}],
                }
            )
        )

    return _bell_ir_with_result


def test_simulator_run_noisy_circuit(noisy_circuit_2_qubit):
    simulator = DensityMatrixSimulator()
    shots_count = 10000
    result = simulator.run(noisy_circuit_2_qubit, qubit_count=2, shots=shots_count)

    assert all([len(measurement) == 2] for measurement in result.measurements)
    assert len(result.measurements) == shots_count
    counter = Counter(["".join(measurement) for measurement in result.measurements])
    assert counter.keys() == {"10", "11"}
    assert 0.0 < counter["10"] / (counter["10"] + counter["11"]) < 0.2
    assert 0.8 < counter["11"] / (counter["10"] + counter["11"]) < 1.0
    assert result.taskMetadata == TaskMetadata(
        id=result.taskMetadata.id, deviceId=DensityMatrixSimulator.DEVICE_ID, shots=shots_count
    )
    assert result.additionalMetadata == AdditionalMetadata(action=noisy_circuit_2_qubit)


@pytest.mark.parametrize("result_type", invalid_ir_result_types)
@pytest.mark.xfail(raises=TypeError)
def test_simulator_run_invalid_ir_result_types(result_type):
    simulator = DensityMatrixSimulator()
    ir = Program.parse_raw(
        json.dumps({"instructions": [{"type": "h", "target": 0}], "results": [result_type]})
    )
    simulator.run(ir, qubit_count=2, shots=100)


def test_simulator_run_bell_pair(bell_ir):
    simulator = DensityMatrixSimulator()
    shots_count = 10000
    result = simulator.run(bell_ir, qubit_count=2, shots=shots_count)

    assert all([len(measurement) == 2] for measurement in result.measurements)
    assert len(result.measurements) == shots_count
    counter = Counter(["".join(measurement) for measurement in result.measurements])
    assert counter.keys() == {"00", "11"}
    assert 0.4 < counter["00"] / (counter["00"] + counter["11"]) < 0.6
    assert 0.4 < counter["11"] / (counter["00"] + counter["11"]) < 0.6
    assert result.taskMetadata == TaskMetadata(
        id=result.taskMetadata.id, deviceId=DensityMatrixSimulator.DEVICE_ID, shots=shots_count
    )
    assert result.additionalMetadata == AdditionalMetadata(action=bell_ir)


@pytest.mark.xfail(raises=ValueError)
def test_simulator_run_no_results_no_shots(bell_ir):
    simulator = DensityMatrixSimulator()
    simulator.run(bell_ir, qubit_count=2, shots=0)


def test_simulator_run_grcs_8(grcs_8_qubit):
    simulator = DensityMatrixSimulator()
    result = simulator.run(grcs_8_qubit.circuit_ir, qubit_count=8, shots=0)
    density_matrix = result.resultTypes[0].value
    assert cmath.isclose(density_matrix[0][0].real, grcs_8_qubit.probability_zero, abs_tol=1e-7)


@pytest.mark.xfail(raises=ValueError)
def test_simulator_run_densitymatrix_shots():
    simulator = DensityMatrixSimulator()
    ir = Program.parse_raw(
        json.dumps(
            {"instructions": [{"type": "h", "target": 0}], "results": [{"type": "densitymatrix"}]}
        )
    )
    simulator.run(ir, qubit_count=2, shots=100)


def test_simulator_run_result_types_shots():
    simulator = DensityMatrixSimulator()
    ir = Program.parse_raw(
        json.dumps(
            {
                "instructions": [
                    {"type": "h", "target": 0},
                    {"type": "cnot", "target": 1, "control": 0},
                ],
                "results": [{"type": "expectation", "observable": ["z"], "targets": [1]}],
            }
        )
    )
    shots_count = 100
    result = simulator.run(ir, qubit_count=2, shots=shots_count)
    assert all([len(measurement) == 2] for measurement in result.measurements)
    assert len(result.measurements) == shots_count
    assert result.measuredQubits == [0, 1]
    assert not result.resultTypes


def test_simulator_run_result_types_shots_basis_rotation_gates():
    simulator = DensityMatrixSimulator()
    ir = Program.parse_raw(
        json.dumps(
            {
                "instructions": [
                    {"type": "h", "target": 0},
                    {"type": "cnot", "target": 1, "control": 0},
                ],
                "basis_rotation_instructions": [{"type": "h", "target": 1}],
                "results": [{"type": "expectation", "observable": ["x"], "targets": [1]}],
            }
        )
    )
    shots_count = 1000
    result = simulator.run(ir, qubit_count=2, shots=shots_count)
    assert all([len(measurement) == 2] for measurement in result.measurements)
    assert len(result.measurements) == shots_count
    assert not result.resultTypes
    assert result.measuredQubits == [0, 1]


@pytest.mark.xfail(raises=ValueError)
def test_simulator_run_result_types_shots_basis_rotation_gates_value_error():
    simulator = DensityMatrixSimulator()
    ir = Program.parse_raw(
        json.dumps(
            {
                "instructions": [
                    {"type": "h", "target": 0},
                    {"type": "cnot", "target": 1, "control": 0},
                ],
                "basis_rotation_instructions": [{"type": "foo", "target": 1}],
                "results": [{"type": "expectation", "observable": ["x"], "targets": [1]}],
            }
        )
    )
    shots_count = 1000
    simulator.run(ir, qubit_count=2, shots=shots_count)


@pytest.mark.parametrize("targets", [(None), ([1]), ([0])])
def test_simulator_bell_pair_result_types(bell_ir_with_result, targets):
    simulator = DensityMatrixSimulator()
    ir = bell_ir_with_result(targets)
    result = simulator.run(ir, qubit_count=2, shots=0)
    assert len(result.resultTypes) == 1
    assert result.resultTypes[0] == ResultTypeValue.construct(
        type=ir.results[0], value=(0 if targets else [0, 0])
    )
    assert result.taskMetadata == TaskMetadata(
        id=result.taskMetadata.id, deviceId=DensityMatrixSimulator.DEVICE_ID, shots=0
    )
    assert result.additionalMetadata == AdditionalMetadata(action=ir)


@pytest.mark.xfail(raises=ValueError)
def test_simulator_fails_samples_0_shots():
    simulator = DensityMatrixSimulator()
    prog = Program.parse_raw(
        json.dumps(
            {
                "instructions": [{"type": "h", "target": 0}],
                "results": [{"type": "sample", "observable": ["x"], "targets": [0]}],
            }
        )
    )
    simulator.run(prog, qubit_count=1, shots=0)


@pytest.mark.parametrize(
    "result_types,expected",
    [
        (
            [
                {"type": "expectation", "observable": ["x"], "targets": [1]},
                {"type": "variance", "observable": ["x"], "targets": [1]},
            ],
            [0, 1],
        ),
        (
            [
                {"type": "expectation", "observable": ["x"]},
                {"type": "variance", "observable": ["x"], "targets": [1]},
            ],
            [[0, 0], 1],
        ),
        (
            [
                {
                    "type": "expectation",
                    "observable": [[[[0, 0], [1, 0]], [[1, 0], [0, 0]]]],
                    "targets": [1],
                },
                {
                    "type": "variance",
                    "observable": [[[[0, 0], [1, 0]], [[1, 0], [0, 0]]]],
                    "targets": [1],
                },
            ],
            [0, 1],
        ),
        (
            [
                {
                    "type": "expectation",
                    "observable": ["x", [[[0, 0], [1, 0]], [[1, 0], [0, 0]]]],
                    "targets": [0, 1],
                },
                {
                    "type": "expectation",
                    "observable": ["x", [[[0, 0], [1, 0]], [[1, 0], [0, 0]]]],
                    "targets": [0, 1],
                },
            ],
            [1, 1],
        ),
        (
            [
                {"type": "variance", "observable": ["x"], "targets": [1]},
                {"type": "expectation", "observable": ["x"]},
                {
                    "type": "expectation",
                    "observable": ["x", [[[0, 0], [1, 0]], [[1, 0], [0, 0]]]],
                    "targets": [0, 1],
                },
            ],
            [1, [0, 0], 1],
        ),
    ],
)
def test_simulator_valid_observables(result_types, expected):
    simulator = DensityMatrixSimulator()
    prog = Program.parse_raw(
        json.dumps(
            {
                "instructions": [
                    {"type": "h", "target": 0},
                    {"type": "cnot", "target": 1, "control": 0},
                ],
                "results": result_types,
            }
        )
    )
    result = simulator.run(prog, qubit_count=2, shots=0)
    for i in range(len(result_types)):
        assert np.allclose(result.resultTypes[i].value, expected[i])


def test_properties():
    simulator = DensityMatrixSimulator()
    observables = ["x", "y", "z", "h", "i", "hermitian"]
    max_shots = sys.maxsize
    qubit_count = 13
    expected_properties = GateModelSimulatorDeviceCapabilities.parse_obj(
        {
            "service": {
                "executionWindows": [
                    {
                        "executionDay": "Everyday",
                        "windowStartHour": "00:00",
                        "windowEndHour": "23:59:59",
                    }
                ],
                "shotsRange": [0, max_shots],
            },
            "action": {
                "braket.ir.jaqcd.program": {
                    "actionType": "braket.ir.jaqcd.program",
                    "version": ["1"],
                    "supportedOperations": [
                        "amplitude_damping",
                        "bit_flip",
                        "ccnot",
                        "cnot",
                        "cphaseshift",
                        "cphaseshift00",
                        "cphaseshift01",
                        "cphaseshift10",
                        "cswap",
                        "cy",
                        "cz",
                        "depolarizing",
                        "generalized_amplitude_damping",
                        "h",
                        "i",
                        "iswap",
                        "kraus",
                        "pauli_channel",
                        "phase_flip",
                        "phase_damping",
                        "phaseshift",
                        "pswap",
                        "rx",
                        "ry",
                        "rz",
                        "s",
                        "si",
                        "swap",
                        "t",
                        "ti",
                        "two_qubit_dephasing",
                        "two_qubit_depolarizing",
                        "unitary",
                        "v",
                        "vi",
                        "x",
                        "xx",
                        "xy",
                        "y",
                        "yy",
                        "z",
                        "zz",
                    ],
                    "supportedResultTypes": [
                        {
                            "name": "Sample",
                            "observables": observables,
                            "minShots": 1,
                            "maxShots": max_shots,
                        },
                        {
                            "name": "Expectation",
                            "observables": observables,
                            "minShots": 0,
                            "maxShots": max_shots,
                        },
                        {
                            "name": "Variance",
                            "observables": observables,
                            "minShots": 0,
                            "maxShots": max_shots,
                        },
                        {"name": "Probability", "minShots": 0, "maxShots": max_shots},
                        {"name": "DensityMatrix", "minShots": 0, "maxShots": 0},
                    ],
                }
            },
            "paradigm": {"qubitCount": qubit_count},
            "deviceParameters": GateModelSimulatorDeviceParameters.schema(),
        }
    )
    assert simulator.properties == expected_properties
