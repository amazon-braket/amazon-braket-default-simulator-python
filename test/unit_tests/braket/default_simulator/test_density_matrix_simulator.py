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
from braket.ir.jaqcd import Expectation
from braket.ir.jaqcd import Program as JaqcdProgram
from braket.ir.openqasm import Program as OpenQASMProgram
from braket.task_result import AdditionalMetadata, TaskMetadata

from braket.default_simulator import DensityMatrixSimulator

CircuitData = namedtuple("CircuitData", "circuit_ir probability_zero")


@pytest.fixture(params=["OpenQASM", "Jaqcd"])
def ir_type(request):
    return request.param


@pytest.fixture
def noisy_circuit_2_qubit():
    return (
        JaqcdProgram.parse_raw(
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
        if ir_type == "Jaqcd"
        else OpenQASMProgram(
            source="""
                OPENQASM 3.0;
                qubit[2] q;

                x q;
                #pragma braket noise bit_flip(.1) q[1]
                """
        )
    )


@pytest.fixture
def grcs_8_qubit(ir_type):
    if ir_type == "Jaqcd":
        with open("test/resources/grcs_8.json") as circuit_file:
            data = json.load(circuit_file)
            return CircuitData(
                JaqcdProgram.parse_raw(json.dumps(data["ir"])), data["probability_zero"]
            )
    return CircuitData(OpenQASMProgram(source="test/resources/grcs_8.qasm"), 0.0007324)


@pytest.fixture
def bell_ir(ir_type):
    return (
        JaqcdProgram.parse_raw(
            json.dumps(
                {
                    "instructions": [
                        {"type": "h", "target": 0},
                        {"type": "cnot", "target": 1, "control": 0},
                    ]
                }
            )
        )
        if ir_type == "Jaqcd"
        else OpenQASMProgram(
            source="""
            OPENQASM 3.0;
            qubit[2] q;

            h q[0];
            cnot q[0], q[1];
            """
        )
    )


def test_simulator_run_noisy_circuit(noisy_circuit_2_qubit, caplog):
    simulator = DensityMatrixSimulator()
    shots_count = 10000
    if isinstance(noisy_circuit_2_qubit, JaqcdProgram):
        result = simulator.run(noisy_circuit_2_qubit, qubit_count=2, shots=shots_count)
    else:
        result = simulator.run(noisy_circuit_2_qubit, shots=shots_count)

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
    assert not caplog.text


def test_simulator_run_bell_pair(bell_ir, caplog):
    simulator = DensityMatrixSimulator()
    shots_count = 10000
    if isinstance(bell_ir, JaqcdProgram):
        result = simulator.run(bell_ir, qubit_count=2, shots=shots_count)
    else:
        result = simulator.run(bell_ir, shots=shots_count)

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
    assert not caplog.text


@pytest.mark.xfail(raises=ValueError)
def test_simulator_run_no_results_no_shots(bell_ir):
    simulator = DensityMatrixSimulator()
    if isinstance(bell_ir, JaqcdProgram):
        simulator.run(bell_ir, qubit_count=2, shots=0)
    else:
        simulator.run(bell_ir, shots=0)


def test_simulator_run_grcs_8(grcs_8_qubit):
    simulator = DensityMatrixSimulator()
    if isinstance(grcs_8_qubit.circuit_ir, JaqcdProgram):
        result = simulator.run(grcs_8_qubit.circuit_ir, qubit_count=8, shots=0)
    else:
        result = simulator.run(grcs_8_qubit.circuit_ir, shots=0)
    density_matrix = result.resultTypes[0].value
    assert cmath.isclose(density_matrix[0][0].real, grcs_8_qubit.probability_zero, abs_tol=1e-7)


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
                "braket.ir.openqasm.program": {
                    "actionType": "braket.ir.openqasm.program",
                    "version": ["1"],
                    "supportedOperations": sorted(
                        [
                            # OpenQASM primitives
                            "U",
                            "GPhase",
                            # builtin Braket gates
                            "ccnot",
                            "cnot",
                            "cphaseshift",
                            "cphaseshift00",
                            "cphaseshift01",
                            "cphaseshift10",
                            "cswap",
                            "cv",
                            "cy",
                            "cz",
                            "ecr",
                            "gpi",
                            "gpi2",
                            "h",
                            "i",
                            "iswap",
                            "ms",
                            "pswap",
                            "phaseshift",
                            "prx",
                            "rx",
                            "ry",
                            "rz",
                            "s",
                            "si",
                            "swap",
                            "t",
                            "ti",
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
                            # noise operations
                            "bit_flip",
                            "phase_flip",
                            "pauli_channel",
                            "depolarizing",
                            "two_qubit_depolarizing",
                            "two_qubit_dephasing",
                            "amplitude_damping",
                            "generalized_amplitude_damping",
                            "phase_damping",
                            "kraus",
                        ]
                    ),
                    "supportedModifiers": [
                        {
                            "name": "ctrl",
                        },
                        {
                            "name": "negctrl",
                        },
                        {
                            "name": "pow",
                            "exponent_types": ["int", "float"],
                        },
                        {
                            "name": "inv",
                        },
                    ],
                    "supportedPragmas": [
                        "braket_unitary_matrix",
                        "braket_result_type_state_vector",
                        "braket_result_type_density_matrix",
                        "braket_result_type_sample",
                        "braket_result_type_expectation",
                        "braket_result_type_variance",
                        "braket_result_type_probability",
                        "braket_result_type_amplitude",
                        "braket_noise_amplitude_damping",
                        "braket_noise_bit_flip",
                        "braket_noise_depolarizing",
                        "braket_noise_kraus",
                        "braket_noise_pauli_channel",
                        "braket_noise_generalized_amplitude_damping",
                        "braket_noise_phase_flip",
                        "braket_noise_phase_damping",
                        "braket_noise_two_qubit_dephasing",
                        "braket_noise_two_qubit_depolarizing",
                    ],
                    "forbiddenPragmas": [
                        "braket_result_type_adjoint_gradient",
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
                    "supportPhysicalQubits": False,
                    "supportsPartialVerbatimBox": False,
                    "requiresContiguousQubitIndices": True,
                    "requiresAllQubitsMeasurement": False,
                    "supportsUnassignedMeasurements": True,
                    "disabledQubitRewiringSupported": False,
                },
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
                        "cv",
                        "cy",
                        "cz",
                        "depolarizing",
                        "ecr",
                        "generalized_amplitude_damping",
                        "h",
                        "i",
                        "iswap",
                        "kraus",
                        "pauli_channel",
                        "two_qubit_pauli_channel",
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
                },
            },
            "paradigm": {"qubitCount": qubit_count},
            "deviceParameters": GateModelSimulatorDeviceParameters.schema(),
        }
    )
    print(expected_properties)
    assert simulator.properties == expected_properties


def test_openqasm_density_matrix_simulator():
    noisy_bell_qasm = """
    qubit[2] qs;

    h qs[0];
    cnot qs[0], qs[1];

    #pragma braket noise bit_flip(.2) qs[1]

    #pragma braket result probability
    """
    device = DensityMatrixSimulator()
    program = OpenQASMProgram(source=noisy_bell_qasm)
    result = device.run(program)
    probabilities = result.resultTypes[0].value
    assert np.allclose(probabilities, [0.4, 0.1, 0.1, 0.4])


invalid_ir_result_types = [
    {"type": "statevector"},
    {"type": "amplitude", "states": ["11"]},
]


@pytest.fixture
def bell_ir_with_result(ir_type):
    def _bell_ir_with_result(targets=None):
        if ir_type == "Jaqcd":
            return JaqcdProgram.parse_raw(
                json.dumps(
                    {
                        "instructions": [
                            {"type": "h", "target": 0},
                            {"type": "cnot", "target": 1, "control": 0},
                        ],
                        "results": [
                            {"type": "expectation", "observable": ["x"], "targets": targets},
                        ],
                    }
                )
            )
        if targets is None:
            observable_string = "x all"
        elif len(targets) == 1:
            observable_string = f"x(q[{targets[0]}])"
        else:
            raise ValueError("bad test")

        return OpenQASMProgram(
            source=f"""
            qubit[2] q;

            h q[0];
            cnot q[0], q[1];

            #pragma braket result expectation {observable_string}
            """
        )

    return _bell_ir_with_result


@pytest.mark.parametrize("result_type", invalid_ir_result_types)
@pytest.mark.xfail(raises=TypeError)
def test_simulator_run_invalid_ir_result_types(result_type):
    simulator = DensityMatrixSimulator()
    ir = JaqcdProgram.parse_raw(
        json.dumps({"instructions": [{"type": "h", "target": 0}], "results": [result_type]})
    )
    simulator.run(ir, qubit_count=2, shots=100)


@pytest.mark.parametrize(
    "result_type",
    (
        "#pragma braket result state_vector",
        "#pragma braket result density_matrix",
        '#pragma braket result amplitude "0"',
    ),
)
def test_simulator_run_invalid_ir_result_types_openqasm(result_type):
    simulator = DensityMatrixSimulator()
    ir = OpenQASMProgram(
        source=f"""
        qubit q;
        h q;
        {result_type}
        """
    )
    with pytest.raises(TypeError):
        simulator.run(ir, qubit_count=2, shots=100)


def test_simulator_run_densitymatrix_shots():
    simulator = DensityMatrixSimulator()
    jaqcd = JaqcdProgram.parse_raw(
        json.dumps(
            {"instructions": [{"type": "h", "target": 0}], "results": [{"type": "densitymatrix"}]}
        )
    )
    qasm = OpenQASMProgram(
        source="""
        qubit q;
        h q;
        #pragma braket result density_matrix
        """
    )
    with pytest.raises(ValueError):
        simulator.run(jaqcd, qubit_count=2, shots=100)
    with pytest.raises(ValueError):
        simulator.run(qasm, shots=100)


def test_simulator_run_result_types_shots(caplog):
    simulator = DensityMatrixSimulator()
    jaqcd = JaqcdProgram.parse_raw(
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
    qasm = OpenQASMProgram(
        source="""
        qubit[2] q;
        h q[0];
        cnot q[0], q[1];
        #pragma braket result expectation z(q[1])
        """
    )
    shots_count = 100
    jaqcd_result = simulator.run(jaqcd, qubit_count=2, shots=shots_count)
    qasm_result = simulator.run(qasm, shots=shots_count)
    for result in jaqcd_result, qasm_result:
        assert all([len(measurement) == 2] for measurement in result.measurements)
        assert len(result.measurements) == shots_count
        assert result.measuredQubits == [0, 1]
    assert not jaqcd_result.resultTypes
    assert not caplog.text


def test_simulator_run_result_types_shots_basis_rotation_gates(caplog):
    simulator = DensityMatrixSimulator()
    jaqcd = JaqcdProgram.parse_raw(
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
    qasm = OpenQASMProgram(
        source="""
            qubit[2] q;
            h q[0];
            cnot q[0], q[1];
            #pragma braket result expectation x(q[1])
            """
    )
    shots_count = 1000
    jaqcd_result = simulator.run(jaqcd, qubit_count=2, shots=shots_count)
    qasm_result = simulator.run(qasm, shots=shots_count)
    for result in jaqcd_result, qasm_result:
        assert all([len(measurement) == 2] for measurement in result.measurements)
        assert len(result.measurements) == shots_count
        assert result.measuredQubits == [0, 1]
    assert not jaqcd_result.resultTypes
    assert not caplog.text


@pytest.mark.xfail(raises=ValueError)
def test_simulator_run_result_types_shots_basis_rotation_gates_value_error():
    simulator = DensityMatrixSimulator()
    ir = JaqcdProgram.parse_raw(
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
def test_simulator_bell_pair_result_types(bell_ir_with_result, targets, caplog):
    simulator = DensityMatrixSimulator()
    ir = bell_ir_with_result(targets)
    if isinstance(ir, JaqcdProgram):
        result = simulator.run(ir, qubit_count=2, shots=0)
    else:
        result = simulator.run(ir, shots=0)
    assert len(result.resultTypes) == 1
    expected_expectation = Expectation(observable=["x"], targets=targets)
    assert result.resultTypes[0].type == expected_expectation
    assert np.allclose(result.resultTypes[0].value, 0 if targets else [0, 0])
    assert result.taskMetadata == TaskMetadata(
        id=result.taskMetadata.id, deviceId=DensityMatrixSimulator.DEVICE_ID, shots=0
    )
    assert result.additionalMetadata == AdditionalMetadata(action=ir)
    assert not caplog.text


def test_simulator_fails_samples_0_shots():
    simulator = DensityMatrixSimulator()
    jaqcd = JaqcdProgram.parse_raw(
        json.dumps(
            {
                "instructions": [{"type": "h", "target": 0}],
                "results": [{"type": "sample", "observable": ["x"], "targets": [0]}],
            }
        )
    )
    qasm = OpenQASMProgram(
        source="""
            qubit q;
            h q;
            #pragma braket result sample x(q)
            """
    )
    with pytest.raises(ValueError):
        simulator.run(jaqcd, qubit_count=1, shots=0)
    with pytest.raises(ValueError):
        simulator.run(qasm, shots=0)


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
    prog = JaqcdProgram.parse_raw(
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


@pytest.mark.parametrize(
    "result_types,expected",
    [
        (
            """
            #pragma braket result expectation x(q[1])
            #pragma braket result variance x(q[1])
            """,
            [0, 1],
        ),
        (
            """
            #pragma braket result expectation x all
            #pragma braket result variance x(q[1])
            """,
            [[0, 0], 1],
        ),
        (
            """
            #pragma braket result expectation hermitian([[0, 1], [1, 0]]) q[1]
            #pragma braket result variance hermitian([[0, 1], [1, 0]]) q[1]
            """,
            [0, 1],
        ),
        (
            """
            #pragma braket result expectation x(q[0]) @ hermitian([[0, 1], [1, 0]]) q[1]
            #pragma braket result expectation x(q[0]) @ hermitian([[0, 1], [1, 0]]) q[1]
            """,
            [1, 1],
        ),
        (
            """
            #pragma braket result variance x(q[1])
            #pragma braket result expectation x all
            #pragma braket result expectation x(q[0]) @ hermitian([[0, 1], [1, 0]]) q[1]
            """,
            [1, [0, 0], 1],
        ),
    ],
)
def test_simulator_valid_observables_qasm(result_types, expected, caplog):
    simulator = DensityMatrixSimulator()
    prog = OpenQASMProgram(
        source=f"""
        qubit[2] q;
        h q[0];
        cnot q[0], q[1];
        {result_types}
        """
    )
    result = simulator.run(prog, shots=0)
    for i in range(len(result_types.split("\n")) - 2):
        assert np.allclose(result.resultTypes[i].value, expected[i])
    assert not caplog.text


def test_adjoint_gradient_pragma_dm1():
    simulator = DensityMatrixSimulator()
    prog = OpenQASMProgram(
        source="""
        input float alpha;
        input float beta;
        qubit[1] q;
        h q[0];
        #pragma braket result adjoint_gradient h(q[0]) alpha, beta
        """,
        inputs={"alpha": 0.2, "beta": 0.3},
    )
    ag_not_supported = "Result type adjoint_gradient is not supported."

    with pytest.raises(TypeError, match=ag_not_supported):
        simulator.run(prog, shots=0)


def test_measure_targets():
    qasm = """
    qubit[2] q;
    bit[1] b;
    h q[0];
    cnot q[0], q[1];
    b[0] = measure q[0];
    """
    simulator = DensityMatrixSimulator()
    result = simulator.run(OpenQASMProgram(source=qasm), shots=1000)
    measurements = np.array(result.measurements, dtype=int)
    assert 400 < np.sum(measurements, axis=0)[0] < 600
    assert len(measurements[0]) == 1
    assert result.measuredQubits == [0]


def test_measure_no_gates():
    qasm = """
    bit[4] b;
    qubit[4] q;
    b[0] = measure q[0];
    b[1] = measure q[1];
    b[2] = measure q[2];
    b[3] = measure q[3];
    """
    simulator = DensityMatrixSimulator()
    result = simulator.run(OpenQASMProgram(source=qasm), shots=1000)
    measurements = np.array(result.measurements, dtype=int)
    assert np.all(measurements == np.zeros((1000, 4)))
    assert result.measuredQubits == [0, 1, 2, 3]


def test_measure_with_qubits_not_used():
    qasm = """
    bit[4] b;
    qubit[4] q;
    h q[0];
    cnot q[0], q[1];
    b = measure q;
    """
    simulator = DensityMatrixSimulator()
    result = simulator.run(OpenQASMProgram(source=qasm), shots=1000)
    measurements = np.array(result.measurements, dtype=int)
    assert 400 < np.sum(measurements, axis=0)[0] < 600
    assert 400 < np.sum(measurements, axis=0)[1] < 600
    assert np.sum(measurements, axis=0)[2] == 0
    assert np.sum(measurements, axis=0)[3] == 0
    assert len(measurements[0]) == 4
    assert result.measuredQubits == [0, 1, 2, 3]
