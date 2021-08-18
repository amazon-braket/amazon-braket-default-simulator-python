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

from collections import Counter

import numpy as np
import pytest

from braket.default_simulator import gate_operations, noise_operations, observables
from braket.default_simulator.density_matrix_simulation import DensityMatrixSimulation

sx = np.matrix([[0, 1], [1, 0]])
si = np.matrix([[1, 0], [0, 1]])
matrix_4q = np.kron(np.kron(sx, si), np.kron(si, si))
matrix_5q = np.kron(sx, np.kron(np.kron(sx, si), np.kron(si, si)))
density_matrix_4q = np.zeros((16, 16))
density_matrix_4q[8][8] = 1
density_matrix_5q = np.zeros((32, 32))
density_matrix_5q[24][24] = 1

evolve_testdata = [
    (
        [gate_operations.PauliX([0]), noise_operations.BitFlip([0], 0.1)],
        1,
        [[0.1, 0.0], [0.0, 0.9]],
        [0.1, 0.9],
    ),
    (
        [gate_operations.Hadamard([0]), noise_operations.PhaseFlip([0], 0.1)],
        1,
        [[0.5, 0.4], [0.4, 0.5]],
        [0.5, 0.5],
    ),
    (
        [gate_operations.PauliX([0]), noise_operations.Depolarizing([0], 0.3)],
        1,
        [[0.2, 0.0], [0.0, 0.8]],
        [0.2, 0.8],
    ),
    (
        [gate_operations.PauliX([0]), noise_operations.AmplitudeDamping([0], 0.15)],
        1,
        [[0.15, 0.0], [0.0, 0.85]],
        [0.15, 0.85],
    ),
    (
        [
            gate_operations.Hadamard([0]),
            noise_operations.PhaseDamping([0], 0.36),
            gate_operations.Hadamard([0]),
        ],
        1,
        [[0.9, 0.0], [0.0, 0.1]],
        [0.9, 0.1],
    ),
    (
        [
            gate_operations.PauliX([0]),
            noise_operations.Kraus([0], [[[0.8, 0], [0, 0.8]], [[0, 0.6], [0.6, 0]]]),
        ],
        1,
        [[0.36, 0.0], [0.0, 0.64]],
        [0.36, 0.64],
    ),
    (
        [
            gate_operations.Unitary((0, 1, 2, 3), matrix_4q),
        ],
        4,
        density_matrix_4q,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ),
    (
        [
            noise_operations.Kraus([0, 1, 2, 3, 4], [matrix_5q]),
        ],
        5,
        density_matrix_5q,
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    ),
]

apply_observables_testdata = [
    ([observables.PauliX([0])], [gate_operations.Hadamard([0])], 1),
    ([observables.PauliZ([0])], [], 1),
    ([observables.Identity([0])], [], 1),
    (
        [observables.PauliX([0]), observables.PauliZ([3]), observables.Hadamard([2])],
        [gate_operations.Hadamard([0]), gate_operations.RotY([2], -np.pi / 4)],
        5,
    ),
    (
        [
            observables.TensorProduct(
                [
                    observables.PauliX([0]),
                    observables.PauliZ([3]),
                    observables.Hadamard([2]),
                    observables.Identity([1]),
                ]
            )
        ],
        [gate_operations.Hadamard([0]), gate_operations.RotY([2], -np.pi / 4)],
        5,
    ),
    ([observables.PauliX()], [gate_operations.Hadamard([0]), gate_operations.Hadamard([1])], 2),
    ([observables.PauliZ()], [], 2),
    ([observables.Identity()], [], 2),
    ([observables.TensorProduct([observables.Identity([2]), observables.PauliZ([0])])], [], 3),
    (
        [observables.TensorProduct([observables.PauliX([2]), observables.PauliZ([0])])],
        [gate_operations.Hadamard([2])],
        3,
    ),
]


@pytest.fixture
def qft_circuit_operations():
    def _qft_operations(qubit_count):
        qft_ops = []
        for target_qubit in range(qubit_count):
            angle = np.pi / 2
            qft_ops.append(gate_operations.Hadamard([target_qubit]))
            for control_qubit in range(target_qubit + 1, qubit_count):
                qft_ops.append(gate_operations.CPhaseShift([control_qubit, target_qubit], angle))
                angle /= 2
        return qft_ops

    return _qft_operations


@pytest.mark.parametrize(
    "instructions, qubit_count, density_matrix, probability_amplitudes", evolve_testdata
)
def test_simulation_simple_circuits(
    instructions, qubit_count, density_matrix, probability_amplitudes
):
    simulation = DensityMatrixSimulation(qubit_count, 0)
    simulation.evolve(instructions)
    assert np.allclose(density_matrix, simulation.density_matrix)
    assert np.allclose(probability_amplitudes, simulation.probabilities)


@pytest.mark.parametrize("obs, equivalent_gates, qubit_count", apply_observables_testdata)
def test_apply_observables(obs, equivalent_gates, qubit_count):
    sim_observables = DensityMatrixSimulation(qubit_count, 0)
    sim_observables.apply_observables(obs)
    sim_gates = DensityMatrixSimulation(qubit_count, 0)
    sim_gates.evolve(equivalent_gates)
    assert np.allclose(sim_observables.state_with_observables, sim_gates.density_matrix)


@pytest.mark.xfail(raises=RuntimeError)
def test_apply_observables_fails_second_call():
    simulation = DensityMatrixSimulation(4, 0)
    simulation.apply_observables([observables.PauliX([0])])
    simulation.apply_observables([observables.PauliX([0])])


@pytest.mark.xfail(raises=RuntimeError)
def test_state_with_observables_fails_before_applying():
    DensityMatrixSimulation(4, 0).state_with_observables


def test_simulation_qft_circuit(qft_circuit_operations):
    qubit_count = 6
    simulation = DensityMatrixSimulation(qubit_count, 0)
    operations = qft_circuit_operations(qubit_count)
    simulation.evolve(operations)
    assert np.allclose(simulation.probabilities, [1 / (2 ** qubit_count)] * (2 ** qubit_count))


def test_simulation_retrieve_samples():
    simulation = DensityMatrixSimulation(2, 10000)
    simulation.evolve([gate_operations.Hadamard([0]), gate_operations.CX([0, 1])])
    counter = Counter(simulation.retrieve_samples())
    assert simulation.qubit_count == 2
    assert counter.keys() == {0, 3}
    assert 0.4 < counter[0] / (counter[0] + counter[3]) < 0.6
    assert 0.4 < counter[3] / (counter[0] + counter[3]) < 0.6
    assert counter[0] + counter[3] == 10000
