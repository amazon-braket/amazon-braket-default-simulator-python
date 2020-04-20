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
from braket.circuits import Circuit, ResultType
from braket.devices import LocalSimulator

import numpy as np

# TODO: Pass these in as pytest options in conftest
N_QUBITS = range(18, 21, 2)
N_LAYERS = range(20, 60, 20)


@pytest.fixture
def generate_circuit():
    def _generate_circuit(num_qubits: int, num_layers: int):
        circuit = Circuit()
        for layer in range(num_layers):
            circuit = circuit.add(hayden_preskill_generator(num_qubits, 3 * num_qubits))

        circuit = circuit.add_result_type(ResultType.StateVector())
        return circuit

    return _generate_circuit


def hayden_preskill_generator(qubits: int, numgates: int):
    """Yields the circuit elements for the scrambling unitary.
    Generates a circuit with numgates gates by laying down a
    random gate at each time step.  Gates are chosen from single
    qubit unitary rotations by a random angle, Hadamard, or a
    controlled-Z between a random pair of qubits."""
    circ = Circuit()
    for i in range(numgates):
        circ.add(np.random.choice(
            [
                Circuit().rx(np.random.choice(qubits, 1, replace=True), np.random.ranf()),
                Circuit().ry(np.random.choice(qubits, 1, replace=True), np.random.ranf()),
                Circuit().rz(np.random.choice(qubits, 1, replace=True), np.random.ranf()),
                Circuit().h(np.random.choice(qubits, 1, replace=True)),
                Circuit().cz(*np.random.choice(qubits, 2, replace=False))
            ],
            1,
            replace=True,
            p=[1/8, 1/8, 1/8, 1/8, 1/2])[0])
    return circ


@pytest.mark.parametrize('nqubits,nlayers', itertools.product(N_QUBITS, N_LAYERS))
def test_braket_performance(benchmark, generate_circuit, nqubits, nlayers, partition_size):
    benchmark.group = "braket"
    circuit = generate_circuit(nqubits, nlayers)
    device = LocalSimulator()
    benchmark(device.run, circuit, shots=1, partition_size=partition_size)
