import numpy as np
import pytest

from braket.default_simulator.linalg_utils import controlled_unitary
from braket.default_simulator.openqasm.quantum_simulator import QuantumSimulator


def test_add_qubits():
    quantum_simulator = QuantumSimulator()
    assert np.array_equal(
        quantum_simulator.state_vector,
        np.array([], dtype=complex),
    )
    quantum_simulator.add_qubits(1)
    assert np.array_equal(
        quantum_simulator.state_vector,
        np.array([1, 0], dtype=complex),
    )
    quantum_simulator.add_qubits(2)
    assert np.array_equal(
        quantum_simulator.state_vector,
        np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex),
    )


@pytest.mark.parametrize(
    "qubits, measurement, state_vector",
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
            [0, 2],
            1,
            [0, 0, 0, 0, 0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
        ),
        (
            1,
            1,
            [0, 0, 1 / 2, 1 / 2, 0, 0, 1 / 2, 1 / 2],
        ),
        (
            1,
            0,
            [1 / 2, 1 / 2, 0, 0, 1 / 2, 1 / 2, 0, 0],
        ),
    ),
)
def test_measure_qubits(qubits, measurement, state_vector):
    quantum_simulator = QuantumSimulator()
    quantum_simulator.add_qubits(3)
    # hadamard all hack
    quantum_simulator._state_tensor[:] = 1 / np.sqrt(8)
    quantum_simulator.measure_qubits(qubits, measurement)
    assert np.allclose(quantum_simulator.state_vector, state_vector)


@pytest.mark.parametrize(
    "qubits, result",
    (
        (0, [1 / 2, 1 / 2, 1 / 2, 1 / 2, 0, 0, 0, 0]),
        (1, [1 / 2, 1 / 2, 0, 0, 1 / 2, 1 / 2, 0, 0]),
        ([0, 1], [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0, 0, 0, 0, 0]),
        ([0, 2], [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0, 0, 0, 0, 0]),
        (range(3), [1, 0, 0, 0, 0, 0, 0, 0]),
    ),
)
def test_reset(qubits, result):
    quantum_simulator = QuantumSimulator()
    quantum_simulator.add_qubits(3)
    # hadamard all hack
    quantum_simulator._state_tensor[:] = 1 / np.sqrt(8)
    quantum_simulator.reset_qubits(qubits)
    assert np.allclose(quantum_simulator.state_vector, result)


@pytest.mark.parametrize(
    "qubits, result",
    (
        (
            ([0], [1 / np.sqrt(2), 0, 0, 0, 1 / np.sqrt(2), 0, 0, 0]),
            ([1], [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0, 0, 0, 0, 0]),
            ([0, 1], [1 / 2, 0, 1 / 2, 0, 1 / 2, 0, 1 / 2, 0]),
        )
    ),
)
def test_execute_unitary(qubits, result):
    quantum_simulator = QuantumSimulator()
    quantum_simulator.add_qubits(3)

    hadamard = QuantumSimulator.generate_u(np.pi / 2, 0, np.pi)

    for qubit in qubits:
        quantum_simulator.execute_unitary(hadamard, qubit)
    assert np.allclose(quantum_simulator.state_vector, result)


def test_control():
    quantum_simulator = QuantumSimulator()
    quantum_simulator.add_qubits(3)

    x = QuantumSimulator.generate_u(np.pi, 0, np.pi)
    hadamard = QuantumSimulator.generate_u(np.pi / 2, 0, np.pi)
    ch = controlled_unitary(hadamard)
    nch = controlled_unitary(hadamard, neg=True)

    quantum_simulator.execute_unitary(x, 0)
    quantum_simulator.execute_unitary(ch, (0, 1))
    quantum_simulator.execute_unitary(nch, (0, 2))
    assert np.allclose(
        quantum_simulator.state_vector, [0, 0, 0, 0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0]
    )
