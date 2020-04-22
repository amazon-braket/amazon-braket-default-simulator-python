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

import itertools

import pytest
import numpy as np
from braket.default_simulator.simulator import DefaultSimulator
import braket.ir.jaqcd as jaqcd


N_QUBITS = list(range(4, 6))
N_LAYERS = range(8, 9)


@pytest.fixture
def generate_layered_discrete_gates_circuit():
    def _generate_circuit(num_qubits, num_layers):
        instructions = []
        for layer in range(num_layers):
            for qubit in range(num_qubits):
                instructions.extend([jaqcd.H(target=qubit), jaqcd.X(target=qubit)])
                if qubit > 0:
                    instructions.extend([jaqcd.CNot(control=0, target=qubit)])
        return jaqcd.Program(instructions=instructions)
    return _generate_circuit


@pytest.fixture
def generate_layered_continuous_gates_circuit():
    def _generate_circuit(num_qubits, num_layers):
        instructions = []
        for layer in range(num_layers):
            for qubit in range(num_qubits):
                instructions.extend([jaqcd.Rx(target=qubit, angle=0.15), jaqcd.Ry(target=qubit, angle=0.16), jaqcd.Rz(target=qubit, angle=0.17)])
                if qubit > 0:
                    instructions.extend([jaqcd.CZ(control=0, target=qubit)])
        return jaqcd.Program(instructions=instructions)
    return _generate_circuit


@pytest.fixture
def generate_qft_circuit():
    def _qft_operations(qubit_count):
        qft_ops = []
        for target_qubit in range(qubit_count):
            angle = np.pi / 2
            qft_ops.append(jaqcd.H(target=target_qubit))
            for control_qubit in range(target_qubit + 1, qubit_count):
                qft_ops.append(jaqcd.CPhaseShift(control=control_qubit, target=target_qubit, angle=angle))
                angle /= 2
        return jaqcd.Program(instructions=qft_ops, results=[jaqcd.StateVector()])
    return _qft_operations


#@pytest.mark.parametrize("nqubits,nlayers", itertools.product(N_QUBITS, N_LAYERS))
@pytest.mark.parametrize("nqubits", N_QUBITS)
def test_braket_performance(benchmark, generate_qft_circuit, nqubits):
    circuit = generate_qft_circuit(nqubits)
    device = DefaultSimulator()
    benchmark(device.run, circuit, nqubits, shots=0)