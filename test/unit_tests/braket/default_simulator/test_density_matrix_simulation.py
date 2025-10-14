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

sx = np.array([[0, 1], [1, 0]], dtype=complex)
si = np.array([[1, 0], [0, 1]], dtype=complex)
matrix_2q = np.kron(sx, si).reshape(4, 4)
matrix_4q = np.kron(np.kron(sx, si), np.kron(si, si))
matrix_5q = np.kron(sx, np.kron(np.kron(sx, si), np.kron(si, si)))
density_matrix_4q = np.zeros((16, 16), dtype=complex)
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
        [
            gate_operations.CX([0, 1]),
            noise_operations.Kraus(
                [0, 1],
                [
                    [
                        [1.0, 0, 0, 0],
                        [0, 1.0, 0, 0],
                        [0, 0, 1.0, 0],
                        [0, 0, 0, np.exp(1j * np.pi / 3)],
                    ],
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                ],
            ),
        ],
        2,
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [1, 0, 0, 0],
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
    (
        [gate_operations.Hadamard([0]), noise_operations.TwoQubitDepolarizing([0, 1], 0.0)],
        2,
        [
            [0.5, 0, 0.5, 0],
            [0, 0, 0, 0],
            [0.5, 0, 0.5, 0],
            [0, 0, 0, 0],
        ],
        [0.5, 0, 0.5, 0],
    ),
    # Test cases with non-contiguous qubits - simple cases
    (
        [gate_operations.PauliX([2])],
        3,
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [0, 1, 0, 0, 0, 0, 0, 0],
    ),
    # Test case with unordered qubits - CX with control on higher index
    (
        [gate_operations.CX([2, 0])],
        3,
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [1, 0, 0, 0, 0, 0, 0, 0],
    ),
]

kraus_branch_specific_testdata = [
    (
        [
            noise_operations.Kraus(
                [0],
                [
                    np.eye(2),
                ],
            ),
        ],
        1,
    ),
    (
        [
            gate_operations.Hadamard([0]),
            noise_operations.Kraus(
                [0, 1],
                [
                    np.sqrt(0.7) * np.eye(4),
                    np.sqrt(0.3)
                    * np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
                ],
            ),
        ],
        2,
    ),
    (
        [
            gate_operations.Hadamard([0]),
            noise_operations.Kraus(
                [0, 1, 2, 3, 4],
                [
                    np.sqrt(0.8) * np.eye(32),
                    np.sqrt(0.2) * np.roll(np.eye(32), 1, axis=0),
                ],
            ),
        ],
        5,
    ),
    (
        [
            noise_operations.Kraus(
                [0, 1, 2, 3, 4],
                [
                    np.eye(32),
                ],
            ),
        ],
        5,
    ),
    (
        [
            noise_operations.Kraus(
                [0, 1, 2, 3, 4],
                [
                    np.sqrt(0.9) * np.eye(32),
                    np.sqrt(0.1) * np.diag([1, -1] + [1] * 30),
                ],
            ),
        ],
        5,
    ),
    # Test case with non-contiguous qubits in Kraus operation - simple identity
    (
        [
            noise_operations.Kraus(
                [0, 2],
                [
                    np.eye(4),
                ],
            ),
        ],
        3,
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
    # Test cases with non-contiguous qubits in observables - simple case
    (
        [observables.PauliX([2])],
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


@pytest.mark.parametrize("instructions, qubit_count", kraus_branch_specific_testdata)
def test_kraus_specific_branches(instructions, qubit_count):
    simulation = DensityMatrixSimulation(qubit_count, 0)
    simulation.evolve(instructions)

    assert np.allclose(np.trace(simulation.density_matrix), 1.0, atol=1e-10)
    assert np.all(np.linalg.eigvals(simulation.density_matrix) >= -1e-10)


def test_superoperator_no_swap():
    simulation = DensityMatrixSimulation(1, 0)

    instructions = [noise_operations.Kraus([0], [np.eye(2)])]

    simulation.evolve(instructions)

    expected = np.array([[1, 0], [0, 0]])
    assert np.allclose(simulation.density_matrix, expected, atol=1e-10)


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
    assert np.allclose(simulation.probabilities, [1 / (2**qubit_count)] * (2**qubit_count))


def test_simulation_retrieve_samples():
    simulation = DensityMatrixSimulation(2, 10000)
    simulation.evolve([gate_operations.Hadamard([0]), gate_operations.CX([0, 1])])
    counter = Counter(simulation.retrieve_samples())
    assert simulation.qubit_count == 2
    assert counter.keys() == {0, 3}
    assert 0.4 < counter[0] / (counter[0] + counter[3]) < 0.6
    assert 0.4 < counter[3] / (counter[0] + counter[3]) < 0.6
    assert counter[0] + counter[3] == 10000


@pytest.mark.parametrize(
    "qubit_count, operations, shots, expected_outcomes, tolerance",
    [
        (
            2,
            [gate_operations.Hadamard([0]), gate_operations.CX([0, 1])],
            10000,
            {0: 0.5, 3: 0.5},
            0.1,
        ),
        (
            1,
            [
                gate_operations.PauliX([0]),
                gate_operations.PauliY([0]),
                gate_operations.PauliX([0]),
                gate_operations.PauliY([0]),
            ],
            5000,
            {0: 1.0},
            0.05,
        ),
        (
            3,
            [gate_operations.Hadamard([i]) for i in range(3)]
            + [gate_operations.CX([0, 1]), gate_operations.CX([1, 2])],
            8000,
            {0: 0.5, 7: 0.5},
            0.4,
        ),
        (
            1,
            [
                gate_operations.RotX([0], np.pi / 4),
                gate_operations.RotY([0], np.pi / 3),
                gate_operations.RotZ([0], np.pi / 2),
            ],
            5000,
            {0: 0.5, 1: 0.5},
            0.2,
        ),
        (
            2,
            [
                gate_operations.Hadamard([0]),
                gate_operations.Hadamard([1]),
                gate_operations.S([0]),
                gate_operations.T([1]),
                noise_operations.BitFlip([0], 0.05),
            ],
            5000,
            {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
            0.15,  # Wider tolerance due to noise
        ),
    ],
)
def test_parameterized_simulation(qubit_count, operations, shots, expected_outcomes, tolerance):
    """Test the quantum simulation with various circuits and optimizations."""
    simulation = DensityMatrixSimulation(qubit_count, shots)

    simulation.evolve(operations)

    counter = Counter(simulation.retrieve_samples())
    total_shots = sum(counter.values())

    assert simulation.qubit_count == qubit_count
    assert total_shots == shots

    for outcome, expected_prob in expected_outcomes.items():
        observed_prob = counter.get(outcome, 0) / total_shots
        assert abs(observed_prob - expected_prob) < tolerance


def test_custom_density_matrix():
    """Test simulation with a custom initial density matrix."""
    qubit_count = 4
    shots = 3000

    simulation = DensityMatrixSimulation(qubit_count, shots)
    simulation._density_matrix = density_matrix_4q.copy()

    simulation.evolve([gate_operations.PauliX([0])])

    counter = Counter(simulation.retrieve_samples())
    assert counter.get(0, 0) / shots > 0.9


def test_toffoli_gate():
    """Test Toffoli (CCNot) gate behavior in density matrix simulation."""
    qubit_count = 3
    shots = 5000

    simulation = DensityMatrixSimulation(qubit_count, shots)
    simulation.evolve([gate_operations.CCNot([0, 1, 2])])

    expected_density_matrix = np.zeros((8, 8), dtype=complex)
    expected_density_matrix[0, 0] = 1.0

    assert np.allclose(simulation.density_matrix, expected_density_matrix)
    assert np.allclose(simulation.probabilities, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    simulation = DensityMatrixSimulation(qubit_count, shots)
    simulation.evolve([gate_operations.PauliX([0]), gate_operations.PauliX([1])])
    simulation.evolve([gate_operations.CCNot([0, 1, 2])])

    expected_density_matrix = np.zeros((8, 8), dtype=complex)
    expected_density_matrix[7, 7] = 1.0

    assert np.allclose(simulation.density_matrix, expected_density_matrix)
    assert np.allclose(simulation.probabilities, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    simulation = DensityMatrixSimulation(qubit_count, shots)
    simulation.evolve([gate_operations.Hadamard([0]), gate_operations.PauliX([1])])
    simulation.evolve([gate_operations.CCNot([0, 1, 2])])

    expected_probabilities = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5]
    assert np.allclose(simulation.probabilities, expected_probabilities, atol=1e-10)

    counter = Counter(simulation.retrieve_samples())
    total_shots = sum(counter.values())
    assert total_shots == shots

    assert set(counter.keys()).issubset({2, 7})
    if 2 in counter and 7 in counter:
        prob_2 = counter[2] / total_shots
        prob_7 = counter[7] / total_shots
        assert abs(prob_2 - 0.5) < 0.1
        assert abs(prob_7 - 0.5) < 0.1
