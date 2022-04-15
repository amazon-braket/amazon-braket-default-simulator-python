from unittest.mock import patch

import numpy as np
import pytest

from braket.default_simulator.linalg_utils import controlled_unitary
from braket.default_simulator.openqasm.quantum_simulation import QuantumSimulation


def test_add_qubits():
    quantum_simulation = QuantumSimulation()
    assert np.array_equal(
        quantum_simulation.state_vector,
        np.array([], dtype=complex),
    )
    quantum_simulation.add_qubits(1)
    assert np.array_equal(
        quantum_simulation.state_vector,
        np.array([1, 0], dtype=complex),
    )
    quantum_simulation.add_qubits(2)
    assert np.array_equal(
        quantum_simulation.state_vector,
        np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex),
    )


@pytest.mark.parametrize(
    "qubits, outcome, state_vector",
    (
        (
            [0, 2],
            [1, 0],
            [0, 0, 0, 0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0],
        ),
        (
            [0, 2],
            [1, 1],
            [0, 0, 0, 0, 0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
        ),
        (
            [1, 2],
            [1, 1],
            [0, 0, 0, 1 / np.sqrt(2), 0, 0, 0, 1 / np.sqrt(2)],
        ),
        (
            1,
            [1],
            [0, 0, 1 / 2, 1 / 2, 0, 0, 1 / 2, 1 / 2],
        ),
        (
            1,
            [0],
            [1 / 2, 1 / 2, 0, 0, 1 / 2, 1 / 2, 0, 0],
        ),
    ),
)
@patch(
    "braket.default_simulator.openqasm.quantum_simulation.QuantumSimulation._sample_quantum_state"
)
def test_measure_qubits(mock_sample, qubits, outcome, state_vector):
    mock_sample.return_value = outcome
    quantum_simulation = QuantumSimulation()
    quantum_simulation.add_qubits(3)

    h = QuantumSimulation.generate_u(np.pi / 2, 0, np.pi)
    for q in range(quantum_simulation.num_qubits):
        quantum_simulation.execute_unitary(h, q)

    quantum_simulation.measure_qubits(qubits)
    assert np.allclose(quantum_simulation.state_vector, state_vector)


@pytest.mark.parametrize(
    "qubits, result",
    (
        (0, [1 / 2, 1 / 2, 1 / 2, 1 / 2, 0, 0, 0, 0]),
        (1, [1 / 2, 1 / 2, 0, 0, 1 / 2, 1 / 2, 0, 0]),
        ([0, 1], [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0, 0, 0, 0, 0]),
        ([0, 2], [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0, 0, 0, 0, 0]),
        (range(3), [1, 0, 0, 0, 0, 0, 0, 0]),
        (None, [1, 0, 0, 0, 0, 0, 0, 0]),
    ),
)
def test_reset(qubits, result):
    quantum_simulation = QuantumSimulation()
    quantum_simulation.add_qubits(3)

    h = QuantumSimulation.generate_u(np.pi / 2, 0, np.pi)
    for q in range(quantum_simulation.num_qubits):
        quantum_simulation.execute_unitary(h, q)

    quantum_simulation.reset_qubits(qubits)
    assert np.allclose(quantum_simulation.state_vector, result)


@pytest.mark.parametrize(
    "qubits, result",
    (
        (
            ([0], [1 / np.sqrt(2), 0, 0, 0, 1 / np.sqrt(2), 0, 0, 0]),
            ([1], [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0, 0, 0, 0, 0]),
            ([0, 1], [1 / 2, 0, 1 / 2, 0, 1 / 2, 0, 1 / 2, 0]),
            (range(2), [1 / 2, 0, 1 / 2, 0, 1 / 2, 0, 1 / 2, 0]),
        )
    ),
)
def test_execute_unitary(qubits, result):
    quantum_simulation = QuantumSimulation()
    quantum_simulation.add_qubits(3)

    h = QuantumSimulation.generate_u(np.pi / 2, 0, np.pi)

    for qubit in qubits:
        quantum_simulation.execute_unitary(h, qubit)
    assert np.allclose(quantum_simulation.state_vector, result)


def test_control():
    quantum_simulation = QuantumSimulation()
    quantum_simulation.add_qubits(3)

    x = QuantumSimulation.generate_u(np.pi, 0, np.pi)
    h = QuantumSimulation.generate_u(np.pi / 2, 0, np.pi)
    ch = controlled_unitary(h)
    nch = controlled_unitary(h, neg=True)

    quantum_simulation.execute_unitary(x, 0)
    quantum_simulation.execute_unitary(ch, (0, 1))
    quantum_simulation.execute_unitary(nch, (0, 2))
    assert np.allclose(
        quantum_simulation.state_vector, [0, 0, 0, 0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0]
    )


def test_sample():
    quantum_simulation = QuantumSimulation()
    quantum_simulation.add_qubits(5)

    x = QuantumSimulation.generate_u(np.pi, 0, np.pi)
    cx = controlled_unitary(x)
    h = QuantumSimulation.generate_u(np.pi / 2, 0, np.pi)

    # bell on 2, 4
    quantum_simulation.execute_unitary(h, 4)
    quantum_simulation.execute_unitary(cx, (4, 2))

    for _ in range(100):
        sample = quantum_simulation._sample_quantum_state((4, 3, 2))
        assert np.array_equal(sample, [False, False, False]) or np.array_equal(
            sample, [True, False, True]
        )


def test_state():
    quantum_simulation = QuantumSimulation()
    quantum_simulation.add_qubits(3)

    assert np.array_equal(
        quantum_simulation.state_vector,
        [1, 0, 0, 0, 0, 0, 0, 0],
    )
    assert np.array_equal(
        quantum_simulation.state_tensor,
        [[[1, 0], [0, 0]], [[0, 0], [0, 0]]],
    )
