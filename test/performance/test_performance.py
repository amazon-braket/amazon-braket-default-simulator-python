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
from braket.default_simulator.simulator import DefaultSimulator
from braket.ir.jaqcd import CNot, H, Program, X

# TODO: Pass these in as pytest options in conftest
N_QUBITS = range(17, 19)
N_LAYERS = range(25, 28)


@pytest.fixture
def generate_circuit():
    def _generate_circuit(num_qubits: int, num_layers: int):
        instructions = []
        for layer in range(num_layers):
            instructions.extend([H(target=0), X(target=0)])
            for qubit in range(1, num_qubits):
                instructions.extend(
                    [H(target=qubit), X(target=qubit), CNot(control=0, target=qubit)]
                )

        return Program(instructions=instructions)

    return _generate_circuit


@pytest.mark.parametrize("nqubits,nlayers", itertools.product(N_QUBITS, N_LAYERS))
def test_braket_performance(benchmark, generate_circuit, nqubits, nlayers, partition_size):
    benchmark.group = "braket"
    circuit = generate_circuit(nqubits, nlayers)
    device = DefaultSimulator()
    benchmark(device.run, circuit, nqubits, shots=1, partition_size=partition_size)
