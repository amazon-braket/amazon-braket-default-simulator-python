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

import itertools
import json
import random

import braket.ir.jaqcd as jaqcd
import numpy as np
import pytest

from braket.default_simulator import StateVectorSimulator

results_data = [
    ([jaqcd.Expectation(observable=["x"])]),
    ([jaqcd.Probability(targets=[0, 1])]),
    ([jaqcd.Probability(), jaqcd.Variance(observable=["x"])]),
    ([jaqcd.Variance(observable=["z"], targets=[0])]),
]


@pytest.fixture
def generate_continuous_gates_circuit():
    def _generate_circuit(num_qubits, num_layers, results):
        instructions = []
        for layer in range(num_layers):
            for qubit in range(num_qubits):
                instructions.extend(
                    [
                        jaqcd.Rx(target=qubit, angle=0.15),
                        jaqcd.Ry(target=qubit, angle=0.16),
                        jaqcd.Rz(target=qubit, angle=0.17),
                    ]
                )
                if qubit > 0:
                    instructions.extend([jaqcd.CZ(control=0, target=qubit)])
        return jaqcd.Program(instructions=instructions, results=results)

    return _generate_circuit


@pytest.fixture
def generate_qft_circuit():
    def _qft_operations(qubit_count):
        qft_ops = []
        for target_qubit in range(qubit_count):
            angle = np.pi / 2
            qft_ops.append(jaqcd.H(target=target_qubit))
            for control_qubit in range(target_qubit + 1, qubit_count):
                qft_ops.append(
                    jaqcd.CPhaseShift(control=control_qubit, target=target_qubit, angle=angle)
                )
                angle /= 2

        amplitudes = [
            "".join([str(random.randint(0, 1)) for _ in range(qubit_count)])
            for _ in range(2 ** (qubit_count // 2))
        ]
        return jaqcd.Program(
            instructions=qft_ops,
            results=[
                jaqcd.StateVector(),
                jaqcd.Amplitude(states=amplitudes),
                jaqcd.Expectation(observable=["x"]),
            ],
        )

    return _qft_operations


@pytest.fixture
def grcs_circuit_16():
    with open("resources/grcs_16.json") as circuit_file:
        data = json.load(circuit_file)
        return jaqcd.Program.parse_raw(json.dumps(data["ir"]))


def test_grcs_simulation(benchmark, grcs_circuit_16):
    device = StateVectorSimulator()
    benchmark(device.run, grcs_circuit_16, 16, shots=0)


@pytest.mark.parametrize("nqubits", range(4, 20, 4))
def test_qft(benchmark, generate_qft_circuit, nqubits):
    circuit = generate_qft_circuit(nqubits)
    device = StateVectorSimulator()
    benchmark(device.run, circuit, nqubits, shots=0)


@pytest.mark.parametrize("nqubits,nlayers", itertools.product(range(2, 20, 4), range(4, 22, 8)))
def test_layered_continuous_gates_circuit(
    benchmark, generate_continuous_gates_circuit, nqubits, nlayers
):
    circuit = generate_continuous_gates_circuit(nqubits, nlayers, [jaqcd.StateVector()])
    device = StateVectorSimulator()
    benchmark(device.run, circuit, nqubits, shots=0)


@pytest.mark.parametrize("results", results_data)
def test_layered_continuous_gates_circuit_result_types(
    benchmark, generate_continuous_gates_circuit, results
):
    nqubits = 12
    nlayers = 15
    shots = 0
    circuit = generate_continuous_gates_circuit(nqubits, nlayers, results)
    device = StateVectorSimulator()
    benchmark(device.run, circuit, nqubits, shots=shots)
