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
import re
import sys
from collections import Counter, namedtuple

import numpy as np
import pytest
from braket.device_schema.simulators import (
    GateModelSimulatorDeviceCapabilities,
    GateModelSimulatorDeviceParameters,
)
from braket.ir.jaqcd import Amplitude, DensityMatrix, Expectation, Probability
from braket.ir.jaqcd import Program as JaqcdProgram
from braket.ir.jaqcd import StateVector, Variance
from braket.ir.openqasm import Program as OpenQASMProgram
from braket.task_result import AdditionalMetadata, TaskMetadata

from braket.default_simulator import DefaultSimulator, StateVectorSimulator, observables

CircuitData = namedtuple("CircuitData", "circuit_ir probability_zero")


@pytest.fixture(params=["OpenQASM", "Jaqcd"])
def ir_type(request):
    return request.param


@pytest.fixture
def grcs_16_qubit(ir_type):
    if ir_type == "Jaqcd":
        with open("test/resources/grcs_16.json") as circuit_file:
            data = json.load(circuit_file)
            return CircuitData(
                JaqcdProgram.parse_raw(json.dumps(data["ir"])), data["probability_zero"]
            )
    return CircuitData(OpenQASMProgram(source="test/resources/grcs_16.qasm"), 0.0000062)


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


@pytest.mark.parametrize("batch_size", [1, 5, 10])
def test_simulator_run_grcs_16(grcs_16_qubit, batch_size):
    simulator = StateVectorSimulator()
    if isinstance(grcs_16_qubit.circuit_ir, JaqcdProgram):
        result = simulator.run(
            grcs_16_qubit.circuit_ir, qubit_count=16, shots=0, batch_size=batch_size
        )
    else:
        result = simulator.run(grcs_16_qubit.circuit_ir, shots=0, batch_size=batch_size)
    state_vector = result.resultTypes[0].value
    assert cmath.isclose(abs(state_vector[0]) ** 2, grcs_16_qubit.probability_zero, abs_tol=1e-7)


@pytest.mark.parametrize("batch_size", [1, 5, 10])
def test_simulator_run_bell_pair(bell_ir, batch_size, caplog):
    simulator = StateVectorSimulator()
    shots_count = 10000
    if isinstance(bell_ir, JaqcdProgram):
        result = simulator.run(bell_ir, qubit_count=2, shots=shots_count, batch_size=batch_size)
    else:
        result = simulator.run(bell_ir, shots=shots_count, batch_size=batch_size)

    assert all([len(measurement) == 2] for measurement in result.measurements)
    assert len(result.measurements) == shots_count
    counter = Counter(["".join(measurement) for measurement in result.measurements])
    assert counter.keys() == {"00", "11"}
    assert 0.4 < counter["00"] / (counter["00"] + counter["11"]) < 0.6
    assert 0.4 < counter["11"] / (counter["00"] + counter["11"]) < 0.6
    assert result.taskMetadata == TaskMetadata(
        id=result.taskMetadata.id, deviceId=StateVectorSimulator.DEVICE_ID, shots=shots_count
    )
    assert result.additionalMetadata == AdditionalMetadata(action=bell_ir)
    assert not caplog.text


def test_properties():
    simulator = StateVectorSimulator()
    observables = ["x", "y", "z", "h", "i", "hermitian"]
    max_shots = sys.maxsize
    qubit_count = 26
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
                    "supportedOperations": [
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
                    ],
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
                    ],
                    "forbiddenPragmas": [
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
                        {"name": "StateVector", "minShots": 0, "maxShots": 0},
                        {"name": "DensityMatrix", "minShots": 0, "maxShots": 0},
                        {"name": "Amplitude", "minShots": 0, "maxShots": 0},
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
                        "h",
                        "i",
                        "iswap",
                        "pswap",
                        "phaseshift",
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
                        {"name": "StateVector", "minShots": 0, "maxShots": 0},
                        {"name": "DensityMatrix", "minShots": 0, "maxShots": 0},
                        {"name": "Amplitude", "minShots": 0, "maxShots": 0},
                    ],
                },
            },
            "paradigm": {"qubitCount": qubit_count},
            "deviceParameters": GateModelSimulatorDeviceParameters.schema(),
        }
    )
    assert simulator.properties == expected_properties


def test_alias():
    assert StateVectorSimulator().properties == DefaultSimulator().properties


@pytest.fixture
def sv_adder():
    return """
    OPENQASM 3;

    input uint[4] a_in;
    input uint[4] b_in;

    gate majority a, b, c {
        cnot c, b;
        cnot c, a;
        ccnot a, b, c;
    }

    gate unmaj a, b, c {
        ccnot a, b, c;
        cnot c, a;
        cnot a, b;
    }

    qubit cin;
    qubit[4] a;
    qubit[4] b;
    qubit cout;

    // set input states
    for int[8] i in [0: 3] {
      if(bool(a_in[i])) x a[i];
      if(bool(b_in[i])) x b[i];
    }

    // add a to b, storing result in b
    majority cin, b[3], a[3];
    for int[8] i in [3: -1: 1] { majority a[i], b[i - 1], a[i - 1]; }
    cnot a[0], cout;
    for int[8] i in [1: 3] { unmaj a[i], b[i - 1], a[i - 1]; }
    unmaj cin, b[3], a[3];

    // todo: subtle bug when trying to get a result type for both at once
    #pragma braket result probability cout, b
    #pragma braket result probability cout
    #pragma braket result probability b
    """


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

    #pragma braket result amplitude '00', '01', '10', '11'
    """
    simulator = StateVectorSimulator()
    result = simulator.run(OpenQASMProgram(source=qasm))
    sv = [result.resultTypes[0].value[state] for state in ("00", "01", "10", "11")]
    assert np.allclose(sv, [-1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])


def test_adder(sv_adder):
    simulator = StateVectorSimulator()
    inputs = {"a_in": 7, "b_in": 3}
    result = simulator.run(OpenQASMProgram(source=sv_adder, inputs=inputs), shots=100)
    assert result.resultTypes[0] == Probability(targets=[9, 5, 6, 7, 8])


def test_adder_analytic(sv_adder):
    simulator = StateVectorSimulator()
    inputs = {"a_in": 7, "b_in": 3}
    result = simulator.run(OpenQASMProgram(source=sv_adder, inputs=inputs))
    expected_probs = np.zeros(2**5)
    expected_probs[10] = 1
    probs = np.outer(result.resultTypes[1].value, result.resultTypes[2].value).flatten()
    assert np.allclose(probs, expected_probs)


def test_result_types_analytic():
    simulator = StateVectorSimulator()
    qasm = """
    qubit[3] q;
    bit[3] c;

    h q[0];
    cnot q[0], q[1];
    cnot q[1], q[2];
    x q[2];

    // {{ 001: .5, 110: .5 }}

    #pragma braket result state_vector
    #pragma braket result probability
    #pragma braket result probability all
    #pragma braket result probability q
    #pragma braket result probability q[0]
    #pragma braket result probability q[0:1]
    #pragma braket result probability q[{0, 2, 1}]
    #pragma braket result amplitude "001", "110"
    #pragma braket result density_matrix
    #pragma braket result density_matrix q
    #pragma braket result density_matrix q[0]
    #pragma braket result density_matrix q[0:1]
    #pragma braket result density_matrix q[0], q[1]
    #pragma braket result density_matrix q[{0, 2, 1}]
    #pragma braket result expectation z(q[0])
    #pragma braket result expectation x all
    #pragma braket result variance x(q[0]) @ z(q[2]) @ h(q[1])
    #pragma braket result expectation hermitian([[0, -1im], [0 + 1im, 0]]) q[0]
    """
    program = OpenQASMProgram(source=qasm)
    result = simulator.run(program, shots=0)

    result_types = result.resultTypes

    assert result_types[0].type == StateVector()
    assert result_types[1].type == Probability()
    assert result_types[2].type == Probability()
    assert result_types[3].type == Probability(targets=(0, 1, 2))
    assert result_types[4].type == Probability(targets=(0,))
    assert result_types[5].type == Probability(targets=(0, 1))
    assert result_types[6].type == Probability(targets=(0, 2, 1))
    assert result_types[7].type == Amplitude(states=("001", "110"))
    assert result_types[8].type == DensityMatrix()
    assert result_types[9].type == DensityMatrix(targets=(0, 1, 2))
    assert result_types[10].type == DensityMatrix(targets=(0,))
    assert result_types[11].type == DensityMatrix(targets=(0, 1))
    assert result_types[12].type == DensityMatrix(targets=(0, 1))
    assert result_types[13].type == DensityMatrix(targets=(0, 2, 1))
    assert result_types[14].type == Expectation(observable=("z",), targets=(0,))
    assert result_types[15].type == Expectation(observable=("x",))
    assert result_types[16].type == Variance(observable=("x", "z", "h"), targets=(0, 2, 1))
    assert result_types[17].type == Expectation(
        observable=([[[0, 0], [0, -1]], [[0, 1], [0, 0]]],),
        targets=(0,),
    )

    assert np.allclose(
        result_types[0].value,
        [0, 1 / np.sqrt(2), 0, 0, 0, 0, 1 / np.sqrt(2), 0],
    )
    assert np.allclose(
        result_types[1].value,
        [0, 0.5, 0, 0, 0, 0, 0.5, 0],
    )
    assert np.allclose(
        result_types[2].value,
        [0, 0.5, 0, 0, 0, 0, 0.5, 0],
    )
    assert np.allclose(
        result_types[3].value,
        [0, 0.5, 0, 0, 0, 0, 0.5, 0],
    )
    assert np.allclose(
        result_types[4].value,
        [0.5, 0.5],
    )
    assert np.allclose(
        result_types[5].value,
        [0.5, 0, 0, 0.5],
    )
    assert np.allclose(
        result_types[6].value,
        [0, 0, 0.5, 0, 0, 0.5, 0, 0],
    )
    assert np.isclose(result_types[7].value["001"], 1 / np.sqrt(2))
    assert np.isclose(result_types[7].value["110"], 1 / np.sqrt(2))
    assert np.allclose(
        result_types[8].value,
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0.5, 0, 0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0.5, 0, 0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    )
    assert np.allclose(
        result_types[9].value,
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0.5, 0, 0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0.5, 0, 0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    )
    assert np.allclose(
        result_types[10].value,
        np.eye(2) * 0.5,
    )
    assert np.allclose(
        result_types[11].value,
        [[0.5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]],
    )
    assert np.allclose(
        result_types[12].value,
        [[0.5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]],
    )
    assert np.allclose(
        result_types[13].value,
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0, 0.5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0, 0.5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    )
    assert np.allclose(result_types[14].value, 0)
    assert np.allclose(result_types[14].value, 0)
    assert np.allclose(result_types[16].value, 1)
    assert np.allclose(result_types[17].value, 0)


def test_invalid_standard_observable_target():
    qasm = """
    qubit[2] qs;
    #pragma braket result variance x(qs)
    """
    simulator = StateVectorSimulator()
    program = OpenQASMProgram(source=qasm)

    must_be_one_qubit = "Standard observable target must be exactly 1 qubit."

    with pytest.raises(ValueError, match=must_be_one_qubit):
        simulator.run(program, shots=0)


@pytest.mark.parametrize("shots", (0, 10))
def test_invalid_hermitian_target(shots):
    qasm = """
    OPENQASM 3.0;
    qubit[3] q;
    i q;
    #pragma braket result expectation hermitian([[-6+0im, 2+1im, -3+0im, -5+2im], [2-1im, 0im, 2-1im, -5+4im], [-3+0im, 2+1im, 0im, -4+3im], [-5-2im, -5-4im, -4-3im, -6+0im]]) q[0] # noqa: E501
    """
    simulator = StateVectorSimulator()
    program = OpenQASMProgram(source=qasm)

    invalid_observable = re.escape(
        "Invalid observable specified: ["
        "[[-6.0, 0.0], [2.0, 1.0], [-3.0, 0.0], [-5.0, 2.0]], "
        "[[2.0, -1.0], [0.0, 0.0], [2.0, -1.0], [-5.0, 4.0]], "
        "[[-3.0, 0.0], [2.0, 1.0], [0.0, 0.0], [-4.0, 3.0]], "
        "[[-5.0, -2.0], [-5.0, -4.0], [-4.0, -3.0], [-6.0, 0.0]]"
        "], targets: [0]"
    )

    with pytest.raises(ValueError, match=invalid_observable):
        simulator.run(program, shots=shots)


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
                            {"type": "amplitude", "states": ["11"]},
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

            #pragma braket result amplitude "11"
            #pragma braket result expectation {observable_string}
            """
        )

    return _bell_ir_with_result


@pytest.fixture
def circuit_noise(ir_type):
    if ir_type == "Jaqcd":
        return JaqcdProgram.parse_raw(
            json.dumps(
                {
                    "instructions": [
                        {"type": "h", "target": 0},
                        {"type": "cnot", "target": 1, "control": 0},
                        {"type": "bit_flip", "target": 0, "probability": 0.15},
                    ]
                }
            )
        )
    else:
        return OpenQASMProgram(
            source="""
            OPENQASM 3.0;
            qubit[2] q;
            h q[0];
            cnot q[0], q[1];
            #pragma braket noise bit_flip(.15) q[0]
            """
        )


def test_simulator_identity(caplog):
    simulator = StateVectorSimulator()
    shots_count = 1000
    programs = (
        JaqcdProgram.parse_raw(
            json.dumps({"instructions": [{"type": "i", "target": 0}, {"type": "i", "target": 1}]})
        ),
        OpenQASMProgram(
            source="""
            qubit[2] q;
            i q;
            """
        ),
    )
    for program in programs:
        if isinstance(program, JaqcdProgram):
            result = simulator.run(
                program,
                qubit_count=2,
                shots=shots_count,
            )
        else:
            result = simulator.run(
                program,
                shots=shots_count,
            )
        counter = Counter(["".join(measurement) for measurement in result.measurements])
        assert counter.keys() == {"00"}
        assert counter["00"] == shots_count
    assert not caplog.text


def test_simulator_instructions_not_supported(circuit_noise):
    simulator = StateVectorSimulator()
    no_noise = re.escape(
        "Noise instructions are not supported by the state vector simulator (by default). "
        'You need to use the density matrix simulator: LocalSimulator("braket_dm").'
    )
    with pytest.raises(TypeError, match=no_noise):
        if isinstance(circuit_noise, JaqcdProgram):
            simulator.run(circuit_noise, qubit_count=2, shots=0)
        else:
            simulator.run(circuit_noise, shots=0)


@pytest.mark.xfail(raises=ValueError)
def test_simulator_run_no_results_no_shots(bell_ir):
    simulator = StateVectorSimulator()
    if isinstance(bell_ir, JaqcdProgram):
        simulator.run(bell_ir, qubit_count=2, shots=0)
    else:
        simulator.run(bell_ir, shots=0)


def test_simulator_run_amplitude_shots():
    simulator = StateVectorSimulator()
    jaqcd = JaqcdProgram.parse_raw(
        json.dumps(
            {
                "instructions": [{"type": "h", "target": 0}],
                "results": [{"type": "amplitude", "states": ["00"]}],
            }
        )
    )
    qasm = OpenQASMProgram(
        source="""
        qubit q;
        h q;
        #pragma braket result amplitude "00"
        """
    )
    with pytest.raises(ValueError):
        simulator.run(jaqcd, qubit_count=2, shots=100)
    with pytest.raises(ValueError):
        simulator.run(qasm, shots=100)


def test_simulator_run_amplitude_no_shots_invalid_states():
    simulator = StateVectorSimulator()
    jaqcd = JaqcdProgram.parse_raw(
        json.dumps(
            {
                "instructions": [{"type": "h", "target": 0}],
                "results": [{"type": "amplitude", "states": ["0"]}],
            }
        )
    )
    qasm = OpenQASMProgram(
        source="""
        qubit[2] q;
        h q[0];
        i q[1];
        #pragma braket result amplitude "0"
        """
    )
    with pytest.raises(ValueError):
        simulator.run(jaqcd, qubit_count=2, shots=0)
    with pytest.raises(ValueError):
        simulator.run(qasm, shots=0)


def test_simulator_run_statevector_shots():
    simulator = StateVectorSimulator()
    jaqcd = JaqcdProgram.parse_raw(
        json.dumps(
            {"instructions": [{"type": "h", "target": 0}], "results": [{"type": "statevector"}]}
        )
    )
    qasm = OpenQASMProgram(
        source="""
        qubit q;
        h q;
        #pragma braket result state_vector
        """
    )
    with pytest.raises(ValueError):
        simulator.run(jaqcd, qubit_count=2, shots=100)
    with pytest.raises(ValueError):
        simulator.run(qasm, shots=100)


def test_simulator_run_result_types_shots(caplog):
    simulator = StateVectorSimulator()
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
        qubit[2] qs;
        h qs[0];
        cnot qs[0], qs[1];
        #pragma braket result expectation x(qs[1])
        """
    )
    shots_count = 100
    jaqcd_result = simulator.run(jaqcd, qubit_count=2, shots=shots_count)
    qasm_result = simulator.run(qasm, shots=shots_count)
    for result in jaqcd_result, qasm_result:
        assert all([len(measurement) == 2] for measurement in result.measurements)
        assert len(result.measurements) == shots_count
        assert result.measuredQubits == [0, 1]
    # qasm_result.resultTypes carries info back to the BDK to calculate results
    assert not jaqcd_result.resultTypes
    assert not caplog.text


def test_simulator_run_result_types_shots_basis_rotation_gates(caplog):
    simulator = StateVectorSimulator()
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
    # not a valid computation path for openqasm, since basis rotation instructions
    # are calculated from the result types during simulation
    simulator = StateVectorSimulator()
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


@pytest.mark.parametrize(
    "ir, qubit_count",
    [
        (
            JaqcdProgram.parse_raw(
                json.dumps(
                    {
                        "instructions": [{"type": "z", "target": 2}],
                        "basis_rotation_instructions": [],
                        "results": [],
                    }
                )
            ),
            1,
        ),
        (
            JaqcdProgram.parse_raw(
                json.dumps(
                    {
                        "instructions": [{"type": "h", "target": 0}],
                        "basis_rotation_instructions": [{"type": "z", "target": 3}],
                        "results": [],
                    }
                )
            ),
            2,
        ),
    ],
)
@pytest.mark.xfail(raises=ValueError)
def test_simulator_run_non_contiguous_qubits(ir, qubit_count):
    # not relevant for openqasm, since it handles qubit allocation
    simulator = StateVectorSimulator()
    shots_count = 1000
    simulator.run(ir, qubit_count=qubit_count, shots=shots_count)


@pytest.mark.parametrize(
    "ir, qubit_count",
    [
        (
            JaqcdProgram.parse_raw(
                json.dumps(
                    {
                        "results": [{"targets": [2], "type": "expectation", "observable": ["z"]}],
                        "basis_rotation_instructions": [],
                        "instructions": [{"type": "z", "target": 0}],
                    }
                )
            ),
            1,
        ),
        (
            JaqcdProgram.parse_raw(
                json.dumps(
                    {
                        "results": [{"targets": [2], "type": "expectation", "observable": ["z"]}],
                        "basis_rotation_instructions": [],
                        "instructions": [{"type": "z", "target": 0}, {"type": "z", "target": 1}],
                    }
                )
            ),
            2,
        ),
        (
            OpenQASMProgram(
                source="""
                qubit[2] q;
                z q;
                #pragma braket result expectation z(q[2])
                """
            ),
            None,
        ),
    ],
)
def test_simulator_run_observable_references_invalid_qubit(ir, qubit_count):
    simulator = StateVectorSimulator()
    shots_count = 0
    if isinstance(ir, JaqcdProgram):
        with pytest.raises(ValueError):
            simulator.run(ir, qubit_count=qubit_count, shots=shots_count)
    else:
        # index error since you're indexing from a logical qubit
        with pytest.raises(IndexError):
            simulator.run(ir, shots=shots_count)


@pytest.mark.parametrize("batch_size", [1, 5, 10])
@pytest.mark.parametrize("targets", [(None), ([1]), ([0])])
def test_simulator_bell_pair_result_types(bell_ir_with_result, targets, batch_size, caplog):
    simulator = StateVectorSimulator()
    ir = bell_ir_with_result(targets)
    if isinstance(ir, JaqcdProgram):
        result = simulator.run(ir, qubit_count=2, shots=0, batch_size=batch_size)
    else:
        result = simulator.run(ir, shots=0, batch_size=batch_size)
    assert len(result.resultTypes) == 2
    assert result.resultTypes[0].type == Amplitude(states=["11"])
    assert np.isclose(result.resultTypes[0].value["11"], 1 / np.sqrt(2))
    expected_expectation = Expectation(observable=["x"], targets=targets)
    assert result.resultTypes[1].type == expected_expectation
    assert np.allclose(result.resultTypes[1].value, 0 if targets else [0, 0])
    assert result.taskMetadata == TaskMetadata(
        id=result.taskMetadata.id, deviceId=StateVectorSimulator.DEVICE_ID, shots=0
    )
    assert result.additionalMetadata == AdditionalMetadata(action=ir)
    assert not caplog.text


def test_simulator_fails_samples_0_shots():
    simulator = StateVectorSimulator()
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
    simulator = StateVectorSimulator()
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
    simulator = StateVectorSimulator()
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


def test_observable_hash_tensor_product():
    matrix = np.eye(4)
    obs = observables.TensorProduct(
        [observables.PauliX([0]), observables.Hermitian(matrix, [1, 2]), observables.PauliY([1])]
    )
    hash_dict = StateVectorSimulator._observable_hash(obs)
    matrix_hash = hash_dict[1]
    assert hash_dict == {0: "PauliX", 1: matrix_hash, 2: matrix_hash, 3: "PauliY"}


def test_basis_rotation(caplog):
    qasm = """
    qubit q;
    qubit[2] qs;
    i q;
    h qs;
    #pragma braket result expectation x(q[0])
    #pragma braket result expectation x(qs[0]) @ i(qs[1])
    #pragma braket result variance x(q[0])
    """
    simulator = StateVectorSimulator()
    result = simulator.run(OpenQASMProgram(source=qasm), shots=1000)
    measurements = np.array(result.measurements, dtype=int)
    assert 400 < np.sum(measurements, axis=0)[0] < 600
    assert np.sum(measurements, axis=0)[1] == 0
    assert 400 < np.sum(measurements, axis=0)[2] < 600
    assert not caplog.text


def test_basis_rotation_all(caplog):
    qasm = """
    qubit q;
    qubit[2] qs;
    h q;
    h qs;
    #pragma braket result variance x all
    """
    simulator = StateVectorSimulator()
    result = simulator.run(OpenQASMProgram(source=qasm), shots=1000)
    measurements = np.array(result.measurements, dtype=int)
    assert np.array_equal(measurements, np.zeros([1000, 3]))


@pytest.mark.parametrize(
    "qasm, error_string",
    (
        (
            """
        qubit[2] q;
        i q;
        #pragma braket result expectation x(q[0])
        // # noqa: E501
        #pragma braket result expectation hermitian([[-6+0im, 2+1im, -3+0im, -5+2im], [2-1im, 0im, 2-1im, -5+4im], [-3+0im, 2+1im, 0im, -4+3im], [-5-2im, -5-4im, -4-3im, -6+0im]]) q[0:1]
        """,
            "Qubit part of incompatible results targets",
        ),
        (
            """
        qubit[2] q;
        i q;
        // # noqa: E501
        // # noqa: E501
        #pragma braket result expectation hermitian([[-6+0im, 2+1im, -3+0im, -5+2im], [2-1im, 0im, 2-1im, -5+4im], [-3+0im, 2+1im, 0im, -4+3im], [-5-2im, -5-4im, -4-3im, -6+0im]]) q[0:1]
        // # noqa: E501
        #pragma braket result expectation hermitian([[-5+0im, 2+1im, -3+0im, -5+2im], [2-1im, 0im, 2-1im, -5+4im], [-3+0im, 2+1im, 0im, -4+3im], [-5-2im, -5-4im, -4-3im, -6+0im]]) q[0:1]
        """,
            "Conflicting result types applied to a single qubit",
        ),
        (
            """
        qubit[2] q;
        i q;
        #pragma braket result expectation x(q[0])
        #pragma braket result expectation z(q[0]) @ x(q[1])
        """,
            "Conflicting result types applied to a single qubit",
        ),
    ),
)
def test_partially_overlapping_basis_rotation(qasm, error_string):
    with pytest.raises(ValueError, match=error_string):
        simulator = StateVectorSimulator()
        simulator.run(OpenQASMProgram(source=qasm), shots=1000)


def test_sample(caplog):
    qasm = """
    qubit[2] qs;
    i qs;
    #pragma braket result sample x(qs[0]) @ i(qs[1])
    """
    simulator = StateVectorSimulator()
    result = simulator.run(OpenQASMProgram(source=qasm), shots=1000)
    measurements = np.array(result.measurements, dtype=int)
    assert 400 < np.sum(measurements, axis=0)[0] < 600
    assert np.sum(measurements, axis=0)[1] == 0
    assert not caplog.text


def test_adjoint_gradient_pragma_sv1():
    simulator = StateVectorSimulator()
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


def test_missing_input():
    qasm = """
    input int[8] in_int;
    int[8] doubled;

    doubled = in_int * 2;
    qubit q;
    rx(doubled) q;
    """
    simulator = StateVectorSimulator()
    missing_input = "Missing input variable 'in_int'."
    with pytest.raises(NameError, match=missing_input):
        simulator.run(OpenQASMProgram(source=qasm), shots=1000)


def test_measure_targets():
    qasm = """
    qubit[2] q;
    bit[1] b;
    h q[0];
    cnot q[0], q[1];
    b[0] = measure q[0];
    """
    simulator = StateVectorSimulator()
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
    simulator = StateVectorSimulator()
    result = simulator.run(OpenQASMProgram(source=qasm), shots=1000)
    measurements = np.array(result.measurements, dtype=int)
    assert np.all(measurements == np.zeros((1000, 4)))
    # assert np.sum(measurements, axis=0)[2] == 0
    # assert len(measurements[0]) == 4
    assert result.measuredQubits == [0, 1, 2, 3]


def test_measure_with_qubits_not_used():
    qasm = """
    bit[4] b;
    qubit[4] q;
    h q[0];
    cnot q[0], q[1];
    b = measure q;
    """
    simulator = StateVectorSimulator()
    result = simulator.run(OpenQASMProgram(source=qasm), shots=1000)
    measurements = np.array(result.measurements, dtype=int)
    assert 400 < np.sum(measurements, axis=0)[0] < 600
    assert 400 < np.sum(measurements, axis=0)[1] < 600
    assert np.sum(measurements, axis=0)[2] == 0
    assert np.sum(measurements, axis=0)[3] == 0
    assert len(measurements[0]) == 4
    assert result.measuredQubits == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "operation, state_vector",
    [
        ["rx(π) q[0];", [0, -1j]],
        ["rx(pi) q[0];", [0, -1j]],
        ["rx(ℇ) q[0];", [0.21007866, -0.97768449j]],
        ["rx(euler) q[0];", [0.21007866, -0.97768449j]],
        ["rx(τ) q[0];", [-1, 0]],
        ["rx(tau) q[0];", [-1, 0]],
        ["rx(pi + pi) q[0];", [-1, 0]],
        ["rx(pi - pi) q[0];", [1, 0]],
        ["rx(-pi + pi) q[0];", [1, 0]],
        ["rx(pi * 2) q[0];", [-1, 0]],
        ["rx(pi / 2) q[0];", [0.70710678, -0.70710678j]],
        ["rx(-pi / 2) q[0];", [0.70710678, 0.70710678j]],
        ["rx(-pi) q[0];", [0, 1j]],
        ["rx(pi + 2 * pi) q[0];", [0, 1j]],
        ["rx(pi + pi / 2) q[0];", [-0.70710678, -0.70710678j]],
        ["rx((pi / 4) + (pi / 2) / 2) q[0];", [0.70710678, -0.70710678j]],
        ["rx(0) q[0];", [1, 0]],
        ["rx(0 + 0) q[0];", [1, 0]],
        ["rx((1.1 + 2.04) / 2) q[0];", [0.70738827, -0.70682518j]],
        ["rx((6 - 2.86) * 0.5) q[0];", [0.70738827, -0.70682518j]],
        ["rx(pi ** 2) q[0];", [0.22058404, 0.97536797j]],
    ],
)
def test_rotation_parameter_expressions(operation, state_vector):
    qasm = f"""
    OPENQASM 3.0;
    bit[1] b;
    qubit[1] q;
    {operation}
    #pragma braket result state_vector
    """
    simulator = StateVectorSimulator()
    result = simulator.run(OpenQASMProgram(source=qasm), shots=0)
    assert result.resultTypes[0].type == StateVector()
    assert np.allclose(result.resultTypes[0].value, np.array(state_vector))


def test_run_multiple():
    payloads = [
        OpenQASMProgram(
            source=f"""
            OPENQASM 3.0;
            bit[2] b;
            qubit[2] q;
            {gates[0]} q[0];
            {gates[1]} q[1];
            b = measure q;
            """
        )
        for gates in [("x", "z"), ("z", "x"), ("x", "x")]
    ]
    args = [[2], [5], [10]]
    kwargs = [{"shots": 3}, {"shots": 6}, {"shots": 9}]
    expected_measurements = [[1, 0], [0, 1], [1, 1]]
    simulator = StateVectorSimulator()
    for result, payload_args, expected in zip(
        simulator.run_multiple(payloads, args=args), args, expected_measurements
    ):
        measurements = np.array(result.measurements, dtype=int)
        print(measurements)
        print(expected)
        assert len(measurements) == payload_args[0]
        assert all(np.alltrue(expected == actual) for actual in measurements)
    for result, payload_kwargs, expected in zip(
        simulator.run_multiple(payloads, kwargs=kwargs), kwargs, expected_measurements
    ):
        measurements = np.array(result.measurements, dtype=int)
        assert len(measurements) == payload_kwargs["shots"]
        assert all(np.alltrue(expected == actual) for actual in measurements)


def test_run_multiple_wrong_num_args():
    payload = OpenQASMProgram(
        source="""
            OPENQASM 3.0;
            bit[1] b;
            qubit[1] q;
            h q[0];
            b = measure q;
            """
    )
    args = [[2], [5], [10], [15]]
    simulator = StateVectorSimulator()
    with pytest.raises(ValueError):
        simulator.run_multiple([payload] * 3, args=args)


def test_run_multiple_wrong_num_kwargs():
    payload = OpenQASMProgram(
        source="""
            OPENQASM 3.0;
            bit[1] b;
            qubit[1] q;
            h q[0];
            b = measure q;
            """
    )
    kwargs = [{"shots": 3}, {"shots": 6}]
    simulator = StateVectorSimulator()
    with pytest.raises(ValueError):
        simulator.run_multiple([payload] * 3, kwargs=kwargs)
