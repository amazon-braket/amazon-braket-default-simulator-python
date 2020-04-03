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

from collections import Counter

import braket.default_simulator.gate_operations as operation
import numpy as np
import pytest
from braket.default_simulator.simulation import StateVectorSimulation

testdata = [
    ([operation.Hadamard([0])], 1, [0.70710678, 0.70710678], [0.5, 0.5]),
    ([operation.PauliX([0])], 1, [0, 1], [0, 1]),
    ([operation.PauliY([0])], 1, [0, 1j], [0, 1]),
    ([operation.PauliX([0]), operation.PauliZ([0])], 1, [0, -1], [0, 1]),
    ([operation.PauliX([0]), operation.CX([0, 1])], 2, [0, 0, 0, 1], [0, 0, 0, 1]),
    ([operation.PauliX([0]), operation.CY([0, 1])], 2, [0, 0, 0, 1j], [0, 0, 0, 1]),
    ([operation.PauliX([0]), operation.CZ([0, 1])], 2, [0, 0, 1, 0], [0, 0, 1, 0]),
    ([operation.PauliX([0]), operation.Swap([0, 1])], 2, [0, 1, 0, 0], [0, 1, 0, 0]),
    (
        [operation.PauliX([0]), operation.Swap([0, 2])],
        3,
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
    ),
    ([operation.PauliX([0]), operation.T([0])], 1, [0, 0.70710678 + 0.70710678j], [0, 1],),
    ([operation.PauliX([0]), operation.S([0])], 1, [0, 1j], [0, 1]),
    ([operation.Identity([0])], 1, [1, 0], [1, 0]),
    ([operation.Unitary([0], [[0, 1], [1, 0]])], 1, [0, 1], [0, 1]),
    (
        [operation.PauliX([0]), operation.PhaseShift([0], 0.15)],
        1,
        [0, 0.98877108 + 0.14943813j],
        [0, 1],
    ),
    (
        [operation.PauliX([0]), operation.PauliX([1]), operation.CPhaseShift([0, 1], 0.15)],
        2,
        [0, 0, 0, 0.98877108 + 0.14943813j],
        [0, 0, 0, 1],
    ),
    ([operation.RotX([0], 0.15)], 1, [0.99718882, -0.07492971j], [0.99438554, 0.00561446],),
    (
        [operation.PauliX([0]), operation.RotY([0], 0.15)],
        1,
        [-0.07492971, 0.99718882],
        [0.00561446, 0.99438554],
    ),
    ([operation.ZZ([0, 1], 0.15)], 2, [0.99718882 + 0.07492971j, 0, 0, 0], [1, 0, 0, 0],),
    (
        [operation.YY([0, 1], 0.15)],
        2,
        [0.98877108, 0, 0, 0.14943813j],
        [0.97766824, 0, 0, 0.02233176],
    ),
    (
        [operation.XX([0, 1], 0.15)],
        2,
        [0.70710678, 0, 0, -0.10566872 - 0.69916673j],
        [0.5, 0, 0, 0.5],
    ),
]


@pytest.fixture
def qft_circuit_operations():
    def _qft_operations(qubit_count):
        qft_ops = []
        for target_qubit in range(qubit_count):
            angle = np.pi / 2
            qft_ops.append(operation.Hadamard([target_qubit]))
            for control_qubit in range(target_qubit + 1, qubit_count):
                qft_ops.append(operation.CPhaseShift([control_qubit, target_qubit], angle))
                angle /= 2
        return qft_ops

    return _qft_operations


@pytest.mark.parametrize(
    "instructions, qubit_count, state_vector, probability_amplitudes", testdata
)
def test_simulation_simple_circuits(
    instructions, qubit_count, state_vector, probability_amplitudes
):
    simulation = StateVectorSimulation(qubit_count)
    simulation.evolve(instructions)
    assert np.allclose(state_vector, simulation.state_vector)
    assert np.allclose(probability_amplitudes, simulation.probability_amplitudes)


def test_simulation_qft_circuit(qft_circuit_operations):
    qubit_count = 16
    simulation = StateVectorSimulation(qubit_count)
    operations = qft_circuit_operations(qubit_count)
    simulation.evolve(operations)
    assert np.allclose(
        simulation.probability_amplitudes, [1 / (2 ** qubit_count)] * (2 ** qubit_count)
    )


def test_simulation_retrieve_samples():
    simulation = StateVectorSimulation(2)
    simulation.evolve([operation.Hadamard([0]), operation.CX([0, 1])])
    counter = Counter(simulation.retrieve_samples(10000))
    assert simulation.qubit_count == 2
    assert counter.keys() == {0, 3}
    assert 0.4 < counter[0] / (counter[0] + counter[3]) < 0.6
    assert 0.4 < counter[3] / (counter[0] + counter[3]) < 0.6
    assert counter[0] + counter[3] == 10000
