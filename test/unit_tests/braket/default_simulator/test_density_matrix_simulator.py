import cmath
import json
import sys
from collections import Counter, namedtuple

import pytest
from braket.device_schema.simulators import (
    GateModelSimulatorDeviceCapabilities,
    GateModelSimulatorDeviceParameters,
)
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
    assert simulator.properties == expected_properties
