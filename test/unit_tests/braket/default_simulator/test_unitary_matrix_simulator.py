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

import json

import numpy as np
import pytest
from braket.device_schema.simulators import (
    GateModelSimulatorDeviceCapabilities,
    GateModelSimulatorDeviceParameters,
)
from braket.ir.jaqcd import Program

from braket.default_simulator import gate_operations
from braket.default_simulator.unitary_matrix_simulator import UnitaryMatrixSimulator


@pytest.fixture
def valid_program():
    return Program.parse_raw(
        json.dumps(
            {
                "instructions": [
                    {"type": "h", "target": 0},
                    {"type": "cnot", "target": 1, "control": 0},
                    {"type": "cnot", "target": 2, "control": 1},
                ],
                "results": [{"type": "unitarymatrix"}],
            }
        )
    )


@pytest.fixture
def valid_unitary_matrix_result():
    return np.dot(
        np.kron(gate_operations.CX([]).matrix, np.eye(2)),
        np.dot(
            np.kron(np.eye(2), gate_operations.CX([]).matrix),
            np.kron(np.eye(4), gate_operations.Hadamard([]).matrix),
        ),
    )


def test_simulator_run(valid_program, valid_unitary_matrix_result):
    simulator = UnitaryMatrixSimulator()
    result = simulator.run(valid_program, qubit_count=3)
    unitary_matrix = result.resultTypes[0].value
    assert np.allclose(unitary_matrix, valid_unitary_matrix_result)


@pytest.mark.xfail(raises=ValueError)
def test_simulator_run_with_shots_biggen_than_zero(valid_program):
    simulator = UnitaryMatrixSimulator()
    simulator.run(valid_program, qubit_count=3, shots=1)


@pytest.mark.xfail(raises=ValueError)
def test_run_simulator_with_program_without_results():
    simulator = UnitaryMatrixSimulator()
    simulator.run(
        Program.parse_raw(
            json.dumps(
                {
                    "instructions": [
                        {"type": "h", "target": 0},
                        {"type": "cnot", "target": 1, "control": 0},
                        {"type": "cnot", "target": 2, "control": 1},
                    ],
                }
            )
        ),
        qubit_count=3,
    )


def test_properties():
    simulator = UnitaryMatrixSimulator()
    max_shots = 0
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
                        "CCNot",
                        "CNot",
                        "CPhaseShift",
                        "CPhaseShift00",
                        "CPhaseShift01",
                        "CPhaseShift10",
                        "CSwap",
                        "CY",
                        "CZ",
                        "H",
                        "I",
                        "ISwap",
                        "PSwap",
                        "PhaseShift",
                        "Rx",
                        "Ry",
                        "Rz",
                        "S",
                        "Si",
                        "Swap",
                        "T",
                        "Ti",
                        "Unitary",
                        "V",
                        "Vi",
                        "X",
                        "XX",
                        "XY",
                        "Y",
                        "YY",
                        "Z",
                        "ZZ",
                    ],
                    "supportedResultTypes": [
                        {"name": "UnitaryMatrix", "minShots": 0, "maxShots": 0},
                    ],
                }
            },
            "paradigm": {"qubitCount": qubit_count},
            "deviceParameters": GateModelSimulatorDeviceParameters.schema(),
        }
    )
    assert simulator.properties == expected_properties
