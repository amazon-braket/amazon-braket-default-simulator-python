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
from uuid import UUID
from collections import Counter, namedtuple
from unittest.mock import patch

import numpy as np
import pytest
from braket.device_schema.simulators import (
    GateModelSimulatorDeviceCapabilities,
    GateModelSimulatorDeviceParameters,
)
from braket.ir.jaqcd import Expectation
from braket.ir.jaqcd import Program as JaqcdProgram
from braket.ir.openqasm import Program as OpenQASMProgram
from braket.ir.openqasm.program_set_v1 import ProgramSet
from braket.ir.openqasm.program_v1 import Program
from braket.task_result import AdditionalMetadata, TaskMetadata
from braket.task_result.program_set_executable_result_v1 import (
    ProgramSetExecutableResult,
    ProgramSetExecutableResultMetadata,
)
from braket.task_result.program_set_task_metadata_v1 import ProgramMetadata, ProgramSetTaskMetadata

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
def noncontiguous_jaqcd():
    with open("test/resources/noncontiguous_jaqcd.json") as jaqcd_definition:
        data = json.load(jaqcd_definition)
        return json.dumps(data)


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


@pytest.fixture(scope="module")
def simulator():
    return DensityMatrixSimulator()


def test_simulator_run_noisy_circuit(noisy_circuit_2_qubit, caplog, simulator):
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


def test_simulator_run_bell_pair(bell_ir, caplog, simulator):
    shots_count = 10000
    if isinstance(bell_ir, JaqcdProgram):
        # Ignore qubit_count
        result = simulator.run(bell_ir, shots=shots_count)
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
def test_simulator_run_no_results_no_shots(bell_ir, simulator):
    if isinstance(bell_ir, JaqcdProgram):
        simulator.run(bell_ir, qubit_count=2, shots=0)
    else:
        simulator.run(bell_ir, shots=0)


def test_simulator_run_grcs_8(grcs_8_qubit, simulator):
    if isinstance(grcs_8_qubit.circuit_ir, JaqcdProgram):
        result = simulator.run(grcs_8_qubit.circuit_ir, qubit_count=8, shots=0)
    else:
        result = simulator.run(grcs_8_qubit.circuit_ir, shots=0)
    density_matrix = result.resultTypes[0].value
    assert cmath.isclose(density_matrix[0][0].real, grcs_8_qubit.probability_zero, abs_tol=1e-7)


def test_properties(simulator):
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
                    "requiresContiguousQubitIndices": False,
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
                "braket.ir.openqasm.program_set": {
                    "actionType": "braket.ir.openqasm.program_set",
                    "version": ["1"],
                    "maximumExecutables": 100,
                    "maximumTotalShots": 200_000,
                },
            },
            "paradigm": {"qubitCount": qubit_count},
            "deviceParameters": GateModelSimulatorDeviceParameters.schema(),
        }
    )
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


def test_ghz_0():
    qasm = """
    qubit[4] q;
    h q[0];
    cnot q[0], q[1];
    cnot q[0], q[2];
    ctrl @ x q[0], q[3];
    #pragma braket result probability
    """
    simulator = DensityMatrixSimulator()
    result = simulator.run(OpenQASMProgram(source=qasm))
    probs = result.resultTypes[0].value
    assert np.allclose(probs, np.array([0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5]))


def test_gphase():
    qasm = """
    qubit[2] qs;

    int[8] two = 2;

    gate x a { U(π, 0, π) a; }
    gate cx c, a { ctrl @ x c, a; }
    gate phase c, a {
        gphase(π/2);
        pow(1) @ ctrl(two) @ gphase(π) c, a;
    }
    gate h a { U(π/2, 0, π) a; }

    inv @ U(π/2, 0, π) qs[0];
    cx qs[0], qs[1];
    phase qs[0], qs[1];

    gphase(π);
    inv @ gphase(π / 2);
    negctrl @ ctrl @ gphase(2 * π) qs[0], qs[1];

    #pragma braket result density_matrix
    """
    simulator = DensityMatrixSimulator()
    result = simulator.run(OpenQASMProgram(source=qasm))
    dm = result.resultTypes[0].value
    assert np.allclose(
        dm, np.array([[0.5, 0, 0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [-0.5, 0, 0, 0.5]])
    )


@pytest.mark.parametrize("result_type", invalid_ir_result_types)
@pytest.mark.xfail(raises=TypeError)
def test_simulator_run_invalid_ir_result_types(result_type, simulator):
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
def test_simulator_run_invalid_ir_result_types_openqasm(result_type, simulator):
    ir = OpenQASMProgram(
        source=f"""
        qubit q;
        h q;
        {result_type}
        """
    )
    with pytest.raises(TypeError):
        simulator.run(ir, qubit_count=2, shots=100)


def test_simulator_run_densitymatrix_shots(simulator):
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


def test_simulator_run_result_types_shots(caplog, simulator):
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


def test_simulator_run_result_types_shots_basis_rotation_gates(caplog, simulator):
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
def test_simulator_run_result_types_shots_basis_rotation_gates_value_error(simulator):
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
def test_simulator_bell_pair_result_types(bell_ir_with_result, targets, caplog, simulator):
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


def test_simulator_fails_samples_0_shots(simulator):
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
def test_simulator_valid_observables(result_types, expected, simulator):
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
def test_simulator_valid_observables_qasm(result_types, expected, caplog, simulator):
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


def test_adjoint_gradient_pragma_dm1(simulator):
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


def test_measure_targets(simulator):
    qasm = """
    qubit[2] q;
    bit[1] b;
    h q[0];
    cnot q[0], q[1];
    b[0] = measure q[0];
    """
    result = simulator.run(OpenQASMProgram(source=qasm), shots=1000)
    measurements = np.array(result.measurements, dtype=int)
    assert 400 < np.sum(measurements, axis=0)[0] < 600
    assert len(measurements[0]) == 1
    assert result.measuredQubits == [0]


def test_measure_no_gates(simulator):
    qasm = """
    bit[4] b;
    qubit[4] q;
    b[0] = measure q[0];
    b[1] = measure q[1];
    b[2] = measure q[2];
    b[3] = measure q[3];
    """
    result = simulator.run(OpenQASMProgram(source=qasm), shots=1000)
    measurements = np.array(result.measurements, dtype=int)
    assert np.all(measurements == np.zeros((1000, 4)))
    assert result.measuredQubits == [0, 1, 2, 3]


def test_measure_with_qubits_not_used(simulator):
    qasm = """
    bit[5] b;
    qubit[5] q;
    h q[1];
    cnot q[1], q[3];
    b = measure q;
    """
    result = simulator.run(OpenQASMProgram(source=qasm), shots=1000)
    measurements = np.array(result.measurements, dtype=int)
    assert 400 < np.sum(measurements, axis=0)[1] < 600
    assert 400 < np.sum(measurements, axis=0)[3] < 600
    assert np.sum(measurements, axis=0)[0] == 0
    assert np.sum(measurements, axis=0)[2] == 0
    assert np.sum(measurements, axis=0)[4] == 0
    assert len(measurements[0]) == 5
    assert result.measuredQubits == [0, 1, 2, 3, 4]


def test_noncontiguous_qubits_jaqcd(noncontiguous_jaqcd):
    prg = JaqcdProgram.parse_raw(noncontiguous_jaqcd)
    result = DensityMatrixSimulator().run(prg, qubit_count=2, shots=1)

    assert result.measuredQubits == [0, 1]
    assert result.measurements in ([["0", "0"]], [["1", "1"]])


@pytest.mark.parametrize("qasm_file_name", ["noncontiguous_virtual", "noncontiguous_physical"])
def test_noncontiguous_qubits_openqasm(qasm_file_name, simulator):
    shots = 1000
    result = simulator.run(
        OpenQASMProgram(source=f"test/resources/{qasm_file_name}.qasm"), shots=shots
    )

    assert result.measuredQubits == [2, 8]
    measurements = np.array(result.measurements, dtype=int)
    assert measurements.shape == (shots, 2)
    assert all(
        (np.allclose(measurement, [0, 0]) or np.allclose(measurement, [1, 1]))
        for measurement in measurements
    )


def test_run_multiple(simulator):
    payloads = [
        OpenQASMProgram(
            source=f"""
            OPENQASM 3.0;
            bit[1] b;
            qubit[1] q;
            {gate} q[0];
            #pragma braket result density_matrix
            """
        )
        for gate in ["h", "z", "x"]
    ]
    results = simulator.run_multiple(payloads, shots=0)
    assert np.allclose(results[0].resultTypes[0].value, np.array([[0.5, 0.5], [0.5, 0.5]]))
    assert np.allclose(results[1].resultTypes[0].value, np.array([[1, 0], [0, 0]]))
    assert np.allclose(results[2].resultTypes[0].value, np.array([[0, 0], [0, 1]]))


@patch("uuid.uuid4")
def test_run_program_set_dm(mock_uuid):
    qasm_all_one = """
    OPENQASM 3.0;
    bit[2] b;
    qubit[2] q;
    x q;
    #pragma braket noise bit_flip(0.0) q[0]
    b = measure q;
    """
    qasm_all_zero = """
    OPENQASM 3.0;
    bit[2] b;
    qubit[2] q;
    z q;
    #pragma braket noise bit_flip(0.0) q[0]
    b = measure q;
    """
    shots = 10
    patched_id = UUID("12345678-1234-4567-abcd-1234567890ab")

    mock_uuid.return_value = patched_id
    prog1 = Program(source=qasm_all_one)
    prog2 = Program(source=qasm_all_zero)
    program_set = ProgramSet(programs=[prog1, prog2])
    result = DensityMatrixSimulator().run(program_set, shots=shots)

    expected_metadata = ProgramSetTaskMetadata(
        id=str(patched_id),
        requestedShots=shots,
        successfulShots=shots,
        totalFailedExecutables=0,
        deviceId="braket_dm",
        programMetadata=[
            ProgramMetadata(executables=[ProgramSetExecutableResultMetadata()]),
            ProgramMetadata(executables=[ProgramSetExecutableResultMetadata()]),
        ],
    )
    expected_program_0_executable_results = ProgramSetExecutableResult(
        inputsIndex=0,
        measurements=[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
        measuredQubits=[0, 1],
    )
    expected_program_1_executable_results = ProgramSetExecutableResult(
        inputsIndex=0,
        measurements=[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        measuredQubits=[0, 1],
    )
    assert result.programResults[0].executableResults[0] == expected_program_0_executable_results
    assert result.programResults[1].executableResults[0] == expected_program_1_executable_results
    assert result.taskMetadata == expected_metadata


def test_verbatim_box_with_noise(simulator):
    """Test that a verbatim circuit with noise pragmas inside the box runs successfully.
    Without the fix, this raises QASM3ParsingError: pragmas must be global."""
    source = "\n".join(
        [
            "OPENQASM 3.0;",
            "bit[2] b;",
            "#pragma braket verbatim",
            "box{",
            "h $0;",
            "#pragma braket noise depolarizing(0.01) $0",
            "cnot $0, $1;",
            "#pragma braket noise depolarizing(0.01) $0",
            "#pragma braket noise depolarizing(0.01) $1",
            "}",
            "b[0] = measure $0;",
            "b[1] = measure $1;",
        ]
    )
    result = simulator.run(OpenQASMProgram(source=source, inputs={}), shots=100)
    assert len(result.measurements) == 100
    assert result.measuredQubits == [0, 1]


def test_multiple_verbatim_boxes_with_noise(simulator):
    """Test that multiple verbatim boxes with noise pragmas run successfully,
    using user-written OpenQASM with whitespace variations."""
    source = (
        "OPENQASM 3.0;\n"
        "bit[2] b;\n"
        "#pragma braket verbatim\n"
        "box {\n"
        "  h $0;\n"
        "  #pragma braket noise depolarizing(0.01) $0\n"
        "}\n"
        "#pragma braket verbatim\n"
        "box {\n"
        "  cnot $0, $1;\n"
        "  #pragma braket noise depolarizing(0.01) $0\n"
        "  #pragma braket noise depolarizing(0.01) $1\n"
        "}\n"
        "b[0] = measure $0;\n"
        "b[1] = measure $1;"
    )
    result = simulator.run(OpenQASMProgram(source=source, inputs={}), shots=100)
    assert len(result.measurements) == 100
    assert result.measuredQubits == [0, 1]
    result = simulator.run(OpenQASMProgram(source=source, inputs={}), shots=100)
    assert len(result.measurements) == 100
    assert result.measuredQubits == [0, 1]


def test_noise_without_verbatim_box(simulator):
    """Test that a noisy circuit without verbatim box still works through run_openqasm."""
    source = "\n".join(
        [
            "OPENQASM 3.0;",
            "bit[2] b;",
            "h $0;",
            "#pragma braket noise depolarizing(0.01) $0",
            "cnot $0, $1;",
            "b[0] = measure $0;",
            "b[1] = measure $1;",
        ]
    )
    result = simulator.run(OpenQASMProgram(source=source, inputs={}), shots=100)
    assert len(result.measurements) == 100
    assert result.measuredQubits == [0, 1]


def test_remove_verbatim_box_with_nested_braces():
    """Test that _remove_verbatim_box correctly handles nested braces."""
    source = "\n".join(
        [
            "OPENQASM 3.0;",
            "#pragma braket verbatim",
            "box{",
            "if (b == 1) {",
            "x $0;",
            "}",
            "}",
            "b[0] = measure $0;",
        ]
    )
    result = DensityMatrixSimulator._remove_verbatim_box(source)
    assert "#pragma braket verbatim" not in result
    assert "box{" not in result
    assert "if (b == 1) {" in result
    assert "x $0;" in result


class TestDensityMatrixSimulatorBranchedRun:
    """End-to-end coverage for the Kraus-native branched density-matrix run.

    These tests exercise the ``DensityMatrixSimulator`` overrides wired in for the
    Kraus-native MCM path: ``create_program_context`` (which returns a
    ``DensityMatrixProgramContext``) and ``_run_branched`` (which forms
    ``ρ_total = Σ ρ_sub`` and draws all shots in a single pass). They assert that a
    branched program returns measurements in the same format and column ordering as
    the single-path density-matrix output, reports ``measured_qubits`` as the original
    physical identifiers, and produces exactly the requested number of shots.
    """

    def test_create_program_context_returns_density_matrix_context(self):
        """``create_program_context`` returns a ``DensityMatrixProgramContext`` so the
        Kraus-native MCM path is used (Req 13.1)."""
        from braket.default_simulator.openqasm.density_matrix_program_context import (
            DensityMatrixProgramContext,
        )

        context = DensityMatrixSimulator().create_program_context()
        assert isinstance(context, DensityMatrixProgramContext)

    def test_branched_run_shot_count_format_and_measured_qubits(self):
        """A branched Bell-pair MCM program returns the requested shot count, a
        two-column measurement matrix, original physical ``measuredQubits``, and only
        the physically reachable outcomes (Req 2.2, 2.3, 2.4)."""
        qasm = """
        OPENQASM 3.0;
        bit[2] b;
        qubit[2] q;
        h q[0];
        cnot q[0], q[1];
        b[0] = measure q[0];
        if (b[0] == 1) {
            x q[1];
        }
        b[1] = measure q[1];
        """
        shots = 2000
        result = DensityMatrixSimulator().run(OpenQASMProgram(source=qasm), shots=shots)

        # Requested shot count drawn in a single final pass.
        assert len(result.measurements) == shots
        assert result.taskMetadata.shots == shots
        # Same format/column ordering as single-path DM output: one column per
        # recorded measurement, reported against original physical identifiers.
        assert result.measuredQubits == [0, 1]
        assert all(len(m) == 2 for m in result.measurements)
        assert all(bit in ("0", "1") for m in result.measurements for bit in m)

        # q[1] is Bell-correlated with q[0] then flipped iff b[0]==1, so it is always
        # |0>; only "00" and "10" are reachable (~50/50).
        counts = Counter("".join(m) for m in result.measurements)
        assert set(counts) <= {"00", "10"}
        for key in ("00", "10"):
            assert 0.4 < counts.get(key, 0) / shots < 0.6

    def test_branched_run_reports_noncontiguous_physical_identifiers(self):
        """A branched program touching only sparse, high-index qubits reports the
        original physical qubit identifiers and selects exactly those columns,
        consistent with current measured-qubit selection behavior (Req 2.4, 8.3)."""
        qasm = """
        OPENQASM 3.0;
        bit[2] b;
        qubit[18] q;
        h q[13];
        b[0] = measure q[13];
        if (b[0] == 1) {
            prx(3.141592653589793, 0.0) q[17];
        }
        b[0] = measure q[13];
        b[1] = measure q[17];
        """
        shots = 2000
        result = DensityMatrixSimulator().run(OpenQASMProgram(source=qasm), shots=shots)

        assert len(result.measurements) == shots
        # Original physical identifiers, not the contiguous matrix axes.
        assert result.measuredQubits == [13, 17]
        assert all(len(m) == 2 for m in result.measurements)

        # q[17] flips iff b[0]==1, so the two recorded columns are perfectly
        # correlated: only "00" and "11" are reachable (~50/50).
        counts = Counter("".join(m) for m in result.measurements)
        assert set(counts) <= {"00", "11"}
        for key in ("00", "11"):
            assert 0.4 < counts.get(key, 0) / shots < 0.6

    def test_branched_run_uses_kraus_native_path(self):
        """Parsing a measurement-conditioned program with shots > 0 transitions the
        density-matrix context to branched mode, routing the run through the
        overridden ``_run_branched`` rather than the single-path aggregation."""
        qasm = """
        OPENQASM 3.0;
        bit[2] b;
        qubit[2] q;
        h q[0];
        b[0] = measure q[0];
        if (b[0] == 1) {
            x q[1];
        }
        b[1] = measure q[1];
        """
        simulator = DensityMatrixSimulator()
        context = simulator._parse_program_with_shots(OpenQASMProgram(source=qasm), 100)
        assert context.is_branched
        # ρ_total is the exact mixed state with unit trace.
        rho_total = context.total_density_matrix()
        assert rho_total is not None
        assert np.isclose(np.real(np.trace(rho_total)), 1.0)
