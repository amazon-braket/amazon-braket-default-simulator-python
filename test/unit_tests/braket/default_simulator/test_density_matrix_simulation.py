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
from braket.default_simulator.linalg_utils import partial_trace

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


# ---------------------------------------------------------------------------
# Kraus-native MCM helpers: project_unnormalized and expand_with_ancilla
# ---------------------------------------------------------------------------


def _density_matrix_after(qubit_count, operations):
    """Helper: evolve |0...0> under the given operations and return the density matrix."""
    simulation = DensityMatrixSimulation(qubit_count, 0)
    simulation.evolve(operations)
    return simulation.density_matrix


def test_project_unnormalized_superposition_traces_are_analytic_probabilities():
    """H|0> -> measure: both outcomes have probability 0.5 (Req 1.2, 15.1)."""
    rho = _density_matrix_after(1, [gate_operations.Hadamard([0])])

    rho0, p0 = DensityMatrixSimulation.project_unnormalized(rho, 1, 0, 0)
    rho1, p1 = DensityMatrixSimulation.project_unnormalized(rho, 1, 0, 1)

    assert np.isclose(p0, 0.5, atol=1e-12)
    assert np.isclose(p1, 0.5, atol=1e-12)

    # P_b rho P_b is the unnormalized projected matrix (trace == probability).
    expected_rho0 = np.array([[0.5, 0], [0, 0]], dtype=complex)
    expected_rho1 = np.array([[0, 0], [0, 0.5]], dtype=complex)
    assert np.allclose(rho0, expected_rho0, atol=1e-12)
    assert np.allclose(rho1, expected_rho1, atol=1e-12)
    # Trace of the returned matrix equals the reported probability (no renormalization).
    assert np.isclose(np.real(np.trace(rho0)), p0, atol=1e-12)
    assert np.isclose(np.real(np.trace(rho1)), p1, atol=1e-12)


def test_project_unnormalized_bell_partner_is_deterministic():
    """For a Bell pair, measuring one qubit's partner outcome is deterministic given
    the first measurement (Req 1.2, 15.1).

    Project qubit 0 of the Bell state onto |0>; the resulting (unnormalized) matrix
    then has the partner qubit 1 deterministically in |0> as well.
    """
    bell = _density_matrix_after(2, [gate_operations.Hadamard([0]), gate_operations.CX([0, 1])])

    # Probabilities of measuring qubit 0.
    rho_q0_0, p0 = DensityMatrixSimulation.project_unnormalized(bell, 2, 0, 0)
    rho_q0_1, p1 = DensityMatrixSimulation.project_unnormalized(bell, 2, 0, 1)
    assert np.isclose(p0, 0.5, atol=1e-12)
    assert np.isclose(p1, 0.5, atol=1e-12)

    # Given qubit 0 == 0, measuring qubit 1 is deterministic: P(q1==1) == 0.
    _, p1_given_q0_0_is_one = DensityMatrixSimulation.project_unnormalized(rho_q0_0, 2, 1, 1)
    _, p1_given_q0_0_is_zero = DensityMatrixSimulation.project_unnormalized(rho_q0_0, 2, 1, 0)
    assert np.isclose(p1_given_q0_0_is_one, 0.0, atol=1e-12)
    assert np.isclose(p1_given_q0_0_is_zero, 0.5, atol=1e-12)

    # Given qubit 0 == 1, measuring qubit 1 is deterministically 1.
    _, p1_given_q0_1_is_one = DensityMatrixSimulation.project_unnormalized(rho_q0_1, 2, 1, 1)
    _, p1_given_q0_1_is_zero = DensityMatrixSimulation.project_unnormalized(rho_q0_1, 2, 1, 0)
    assert np.isclose(p1_given_q0_1_is_one, 0.5, atol=1e-12)
    assert np.isclose(p1_given_q0_1_is_zero, 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "qubit_count, operations, qubit_axis",
    [
        (1, [gate_operations.Hadamard([0])], 0),
        (1, [gate_operations.RotX([0], np.pi / 3)], 0),
        (2, [gate_operations.Hadamard([0]), gate_operations.CX([0, 1])], 0),
        (2, [gate_operations.Hadamard([0]), gate_operations.CX([0, 1])], 1),
        (3, [gate_operations.Hadamard([0]), gate_operations.Hadamard([1])], 2),
    ],
)
def test_project_unnormalized_split_conserves_parent_trace(qubit_count, operations, qubit_axis):
    """p0 + p1 == trace(rho) within tolerance (Req 15.4)."""
    rho = _density_matrix_after(qubit_count, operations)
    parent_trace = float(np.real(np.trace(rho)))

    _, p0 = DensityMatrixSimulation.project_unnormalized(rho, qubit_count, qubit_axis, 0)
    _, p1 = DensityMatrixSimulation.project_unnormalized(rho, qubit_count, qubit_axis, 1)

    assert np.isclose(p0 + p1, parent_trace, atol=1e-12)


def test_project_unnormalized_does_not_mutate_input():
    """project_unnormalized must not modify the input density matrix."""
    rho = _density_matrix_after(1, [gate_operations.Hadamard([0])])
    rho_copy = rho.copy()
    DensityMatrixSimulation.project_unnormalized(rho, 1, 0, 0)
    assert np.allclose(rho, rho_copy, atol=1e-15)


def test_expand_with_ancilla_appends_zero_and_preserves_marginals():
    """rho ⊗ |0><0| keeps existing-qubit marginals and adds a |0> qubit (Req 8.1)."""
    rho = _density_matrix_after(1, [gate_operations.Hadamard([0])])

    expanded = DensityMatrixSimulation.expand_with_ancilla(rho, 1)

    assert expanded.shape == (4, 4)
    # Trace is preserved.
    assert np.isclose(np.real(np.trace(expanded)), np.real(np.trace(rho)), atol=1e-12)

    # The new (least-significant) qubit is in |0>: reduced density matrix == |0><0|.
    reshaped = np.reshape(expanded, [2] * 2 * 2)
    new_qubit_marginal = partial_trace(reshaped, [1])
    assert np.allclose(new_qubit_marginal, np.array([[1, 0], [0, 0]], dtype=complex), atol=1e-12)

    # The existing qubit's marginal is unchanged.
    existing_qubit_marginal = partial_trace(reshaped, [0])
    assert np.allclose(existing_qubit_marginal, rho, atol=1e-12)


def test_expand_with_ancilla_matches_explicit_evolution():
    """Expanding a 1-qubit state with an ancilla equals evolving the same gate on a
    2-qubit |00> register (the ancilla axis is least-significant)."""
    rho_1q = _density_matrix_after(1, [gate_operations.Hadamard([0])])
    expanded = DensityMatrixSimulation.expand_with_ancilla(rho_1q, 1)

    rho_2q = _density_matrix_after(2, [gate_operations.Hadamard([0])])
    assert np.allclose(expanded, rho_2q, atol=1e-12)


def test_expand_with_ancilla_multiple_qubits():
    """Appending multiple ancillas grows the dimension by 2 per qubit and adds |0...0>."""
    rho = _density_matrix_after(1, [gate_operations.Hadamard([0])])

    expanded = DensityMatrixSimulation.expand_with_ancilla(rho, 2)

    assert expanded.shape == (8, 8)
    assert np.isclose(np.real(np.trace(expanded)), np.real(np.trace(rho)), atol=1e-12)
    # Both ancillas in |0>.
    reshaped = np.reshape(expanded, [2] * 2 * 3)
    assert np.allclose(partial_trace(reshaped, [0]), rho, atol=1e-12)
    assert np.allclose(
        partial_trace(reshaped, [1]), np.array([[1, 0], [0, 0]], dtype=complex), atol=1e-12
    )
    assert np.allclose(
        partial_trace(reshaped, [2]), np.array([[1, 0], [0, 0]], dtype=complex), atol=1e-12
    )


@pytest.mark.parametrize("num_new", [0, -1])
def test_expand_with_ancilla_noop_for_nonpositive(num_new):
    """expand_with_ancilla returns rho unchanged when num_new <= 0."""
    rho = _density_matrix_after(1, [gate_operations.Hadamard([0])])
    result = DensityMatrixSimulation.expand_with_ancilla(rho, num_new)
    assert result is rho
