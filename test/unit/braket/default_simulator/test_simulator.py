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

import numpy as np
import pytest
from braket.default_simulator import observables
from braket.default_simulator.result_types import Expectation, Variance
from braket.default_simulator.simulator import DefaultSimulator
from braket.ir.jaqcd import Program
from braket.simulator import BraketSimulator

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
    result = simulator.run(grcs_16_qubit.circuit_ir, qubit_count=16, shots=0, batch_size=batch_size)
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
    assert result["TaskMetadata"] == {
        "Id": result["TaskMetadata"]["Id"],
        "Ir": bell_ir.json(),
        "IrType": "jaqcd",
        "Shots": shots_count,
    }


@pytest.mark.xfail(raises=ValueError)
def test_simulator_run_no_results_no_shots(bell_ir):
    simulator = DefaultSimulator()
    simulator.run(bell_ir, qubit_count=2, shots=0)


@pytest.mark.xfail(raises=ValueError)
def test_simulator_run_amplitude_shots():
    simulator = DefaultSimulator()
    ir = Program.parse_raw(
        json.dumps(
            {
                "instructions": [{"type": "h", "target": 0}],
                "results": [{"type": "amplitude", "states": ["0"]}],
            }
        )
    )
    simulator.run(ir, qubit_count=2, shots=100)


@pytest.mark.xfail(raises=ValueError)
def test_simulator_run_statevector_shots():
    simulator = DefaultSimulator()
    ir = Program.parse_raw(
        json.dumps(
            {"instructions": [{"type": "h", "target": 0}], "results": [{"type": "statevector"}]}
        )
    )
    simulator.run(ir, qubit_count=2, shots=100)


def test_simulator_run_result_types_shots():
    simulator = DefaultSimulator()
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
    assert all([len(measurement) == 2] for measurement in result["Measurements"])
    assert len(result["Measurements"]) == shots_count
    assert result["MeasuredQubits"] == [0, 1]
    assert "ResultTypes" not in result


def test_simulator_run_result_types_shots_basis_rotation_gates():
    simulator = DefaultSimulator()
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
    assert all([len(measurement) == 2] for measurement in result["Measurements"])
    assert len(result["Measurements"]) == shots_count
    assert "ResultTypes" not in result
    assert result["MeasuredQubits"] == [0, 1]


@pytest.mark.xfail(raises=ValueError)
def test_simulator_run_result_types_shots_basis_rotation_gates_value_error():
    simulator = DefaultSimulator()
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
        "Id": result["TaskMetadata"]["Id"],
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


@pytest.mark.parametrize(
    "result_types,expected_expectation,expected_variance",
    [
        (
            [
                {"type": "expectation", "observable": ["x"], "targets": [1]},
                {"type": "variance", "observable": ["x"], "targets": [1]},
            ],
            0,
            1,
        ),
        (
            [
                {"type": "expectation", "observable": ["x"]},
                {"type": "variance", "observable": ["x"], "targets": [1]},
            ],
            [0, 0],
            1,
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
            0,
            1,
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
            1,
            1,
        ),
    ],
)
def test_simulator_accepts_overlapping_targets_same_observable(
    result_types, expected_expectation, expected_variance
):
    simulator = DefaultSimulator()
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
    expectation = result["ResultTypes"][0]["Value"]
    variance = result["ResultTypes"][1]["Value"]
    assert np.allclose(expectation, expected_expectation)
    assert np.allclose(variance, expected_variance)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "result_types",
    [
        (
            [
                {"type": "expectation", "observable": ["y"]},
                {"type": "variance", "observable": ["x"], "targets": [1]},
            ]
        ),
        (
            [
                {"type": "expectation", "observable": ["y"], "targets": [1]},
                {"type": "variance", "observable": ["x"], "targets": [1]},
            ]
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
                    "observable": [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]]],
                    "targets": [1],
                },
            ]
        ),
        (
            [
                {
                    "type": "expectation",
                    "observable": ["x", [[[0, 0], [1, 0]], [[1, 0], [0, 0]]]],
                    "targets": [0, 1],
                },
                {"type": "variance", "observable": ["y", "x"], "targets": [0, 1]},
            ]
        ),
        (
            [
                {"type": "expectation", "observable": ["i"]},
                {"type": "variance", "observable": ["y"]},
            ]
        ),
    ],
)
def test_simulator_fails_overlapping_targets_different_observable(result_types):
    simulator = DefaultSimulator()
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
    simulator.run(prog, qubit_count=2, shots=0)


@pytest.mark.parametrize(
    "obs1,obs2",
    [
        (observables.PauliX([1]), observables.PauliX(None)),
        (observables.PauliZ([1]), observables.PauliZ(None)),
        (observables.Hermitian(np.eye(2), [1]), observables.Hermitian(np.eye(2), None)),
    ],
)
def test_validate_and_consolidate_observable_result_types_none(obs1, obs2):
    obs_rts = [
        Expectation(obs1),
        Variance(obs2),
    ]
    actual_obs = DefaultSimulator._validate_and_consolidate_observable_result_types(obs_rts, 2)
    assert len(actual_obs) == 1
    assert actual_obs[0].measured_qubits is None


@pytest.mark.parametrize(
    "obs",
    [(observables.PauliX([1])), (observables.PauliZ([1])), (observables.Hermitian(np.eye(2), [1]))],
)
def test_validate_and_consolidate_observable_result_types_same_target(obs):
    obs_rts = [
        Expectation(obs),
        Variance(obs),
    ]
    actual_obs = DefaultSimulator._validate_and_consolidate_observable_result_types(obs_rts, 2)
    assert len(actual_obs) == 1
    assert actual_obs[0].measured_qubits == (1,)


def test_validate_and_consolidate_observable_result_types_tensor_product():
    obs_rts = [
        Expectation(observables.TensorProduct([observables.PauliX([0]), observables.PauliY([1])])),
        Variance(observables.TensorProduct([observables.PauliX([0]), observables.PauliY([1])])),
        Expectation(observables.TensorProduct([observables.PauliX([2]), observables.PauliY([3])])),
    ]
    actual_obs = DefaultSimulator._validate_and_consolidate_observable_result_types(obs_rts, 4)
    assert len(actual_obs) == 2
    assert actual_obs[0].measured_qubits == (0, 1,)
    assert actual_obs[1].measured_qubits == (2, 3,)


@pytest.mark.parametrize(
    "obs1,obs2",
    [
        (observables.PauliX([1]), observables.PauliX([2])),
        (observables.PauliZ([1]), observables.PauliZ([2])),
        (observables.Hermitian(np.eye(2), [1]), observables.Hermitian(np.eye(2), [2])),
    ],
)
def test_validate_and_consolidate_observable_result_types_targets(obs1, obs2):
    obs_rts = [
        Expectation(obs1),
        Expectation(obs2),
    ]
    actual_obs = DefaultSimulator._validate_and_consolidate_observable_result_types(obs_rts, 2)
    assert len(actual_obs) == 2
    assert actual_obs[0].measured_qubits == (1,)
    assert actual_obs[1].measured_qubits == (2,)


def test_default_simulator_instance_braket_simulator():
    assert isinstance(DefaultSimulator(), BraketSimulator)


def test_properties():
    simulator = DefaultSimulator()
    observables = ["X", "Y", "Z", "H", "I", "Hermitian"]
    max_shots = 10000000
    expected_properties = {
        "supportedQuantumOperations": [
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
            {"name": "Sample", "observables": observables, "minShots": 1, "maxShots": max_shots},
            {
                "name": "Expectation",
                "observables": observables,
                "minShots": 0,
                "maxShots": max_shots,
            },
            {"name": "Variance", "observables": observables, "minShots": 0, "maxShots": max_shots},
            {"name": "Probability", "minShots": 0, "maxShots": max_shots},
            {"name": "StateVector", "minShots": 0, "maxShots": 0},
            {"name": "Amplitude", "minShots": 0, "maxShots": 0},
        ],
    }
    assert simulator.properties == expected_properties
