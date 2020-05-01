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

import cmath
import json
from collections import Counter, namedtuple

import pytest
from braket.default_simulator.simulator import DefaultSimulator
from braket.ir.jaqcd import Program

CircuitData = namedtuple("CircuitData", "circuit_ir probability_zero")


@pytest.fixture
def grcs_16_qubit():
    with open("test/resources/grcs_16.json") as circuit_file:
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
                    "results": [
                        {"type": "amplitude", "states": ["11"]},
                        {"type": "expectation", "observable": ["x"], "targets": targets},
                    ],
                }
            )
        )

    return _bell_ir_with_result


@pytest.mark.parametrize("batch_size", [1, 5, 10])
def test_simulator_run_grcs_16(grcs_16_qubit, batch_size):
    simulator = DefaultSimulator()
    result = simulator.run(
        grcs_16_qubit.circuit_ir, qubit_count=16, shots=100, batch_size=batch_size
    )
    state_vector = result["ResultTypes"][0]["Value"]
    assert cmath.isclose(abs(state_vector[0]) ** 2, grcs_16_qubit.probability_zero, abs_tol=1e-7)


@pytest.mark.parametrize("batch_size", [1, 5, 10])
def test_simulator_run_bell_pair(bell_ir, batch_size):
    simulator = DefaultSimulator()
    shots_count = 10000
    result = simulator.run(bell_ir, qubit_count=2, shots=shots_count, batch_size=batch_size)

    assert all([len(measurement) == 2] for measurement in result["Measurements"])
    assert len(result["Measurements"]) == shots_count
    counter = Counter(["".join(measurement) for measurement in result["Measurements"]])
    assert counter.keys() == {"00", "11"}
    assert 0.4 < counter["00"] / (counter["00"] + counter["11"]) < 0.6
    assert 0.4 < counter["11"] / (counter["00"] + counter["11"]) < 0.6
    assert result["TaskMetadata"] == {"Ir": bell_ir.json(), "IrType": "jaqcd", "Shots": shots_count}


@pytest.mark.parametrize("batch_size", [1, 5, 10])
@pytest.mark.parametrize("targets", [(None), ([1]), ([0])])
def test_simulator_bell_pair_result_types(bell_ir_with_result, targets, batch_size):
    simulator = DefaultSimulator()
    result = simulator.run(
        bell_ir_with_result(targets), qubit_count=2, shots=0, batch_size=batch_size
    )
    assert len(result["ResultTypes"]) == 2
    assert result["ResultTypes"] == [
        {"Type": {"type": "amplitude", "states": ["11"]}, "Value": {"11": 1 / 2 ** 0.5}},
        {
            "Type": {"type": "expectation", "observable": ["x"], "targets": targets},
            "Value": 0 if targets else [0, 0],
        },
    ]
    assert result["TaskMetadata"] == {
        "Ir": bell_ir_with_result(targets).json(),
        "IrType": "jaqcd",
        "Shots": 0,
    }


@pytest.mark.xfail(raises=ValueError)
def test_simulator_fails_samples_0_shots():
    simulator = DefaultSimulator()
    prog = Program.parse_raw(
        json.dumps(
            {
                "instructions": [{"type": "h", "target": 0}],
                "results": [{"type": "sample", "observable": ["x"], "targets": [0]}],
            }
        )
    )
    simulator.run(prog, qubit_count=1, shots=0)


@pytest.mark.xfail(raises=ValueError)
def test_simulator_fails_2_obs_no_targets():
    simulator = DefaultSimulator()
    prog = Program.parse_raw(
        json.dumps(
            {
                "instructions": [
                    {"type": "h", "target": 0},
                    {"type": "cnot", "target": 1, "control": 0},
                ],
                "results": [
                    {"type": "expectation", "observable": ["x"]},
                    {"type": "expectation", "observable": ["x"], "targets": [1]},
                ],
            }
        )
    )
    simulator.run(prog, qubit_count=2, shots=100)


@pytest.mark.xfail(raises=ValueError)
def test_simulator_fails_overlapping_targets():
    simulator = DefaultSimulator()
    prog = Program.parse_raw(
        json.dumps(
            {
                "instructions": [
                    {"type": "h", "target": 0},
                    {"type": "cnot", "target": 1, "control": 0},
                ],
                "results": [
                    {"type": "expectation", "observable": ["x"], "targets": [1]},
                    {"type": "expectation", "observable": ["x"], "targets": [1]},
                ],
            }
        )
    )
    simulator.run(prog, qubit_count=2, shots=100)
