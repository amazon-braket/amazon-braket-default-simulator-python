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


def test_simulator_run_grcs_16(grcs_16_qubit):
    simulator = DefaultSimulator()
    result = simulator.run(grcs_16_qubit.circuit_ir, qubit_count=16, shots=100)
    state_vector = result["StateVector"]
    zero_state = "0" * 16
    assert cmath.isclose(
        abs(state_vector[zero_state]) ** 2, grcs_16_qubit.probability_zero, abs_tol=1e-7
    )


def test_simulator_run_bell_pair(bell_ir):
    simulator = DefaultSimulator()
    shots_count = 10000
    result = simulator.run(bell_ir, qubit_count=2, shots=shots_count)
    expected_state_vector = {"00": 0.70710678, "01": 0, "10": 0, "11": 0.70710678}
    assert result["StateVector"].keys() == expected_state_vector.keys()
    assert all(
        [
            cmath.isclose(result["StateVector"][k], expected_state_vector[k], abs_tol=1e-7)
            for k in expected_state_vector.keys()
        ]
    )

    assert all([len(measurement) == 2] for measurement in result["Measurements"])
    assert len(result["Measurements"]) == shots_count
    counter = Counter(["".join(measurement) for measurement in result["Measurements"]])
    assert counter.keys() == {"00", "11"}
    assert 0.4 < counter["00"] / (counter["00"] + counter["11"]) < 0.6
    assert 0.4 < counter["11"] / (counter["00"] + counter["11"]) < 0.6
