import numpy as np
import pytest

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
    "qubit, mask",
    (
        (0, "01010101"),
        (1, "00110011"),
        (2, "00001111"),
    ),
)
def test_get_qubit_mask(qubit, mask):
    quantum_simulator = QuantumSimulator()
    quantum_simulator.add_qubits(3)
    mask = [x == "1" for x in mask]
    assert np.array_equal(quantum_simulator._get_qubit_mask(qubit), mask)


@pytest.mark.parametrize(
    "qubits, measurement, state_vector",
    (
        (
            [0, 2],
            [1, 0],
            [0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0, 0, 0, 0],
        ),
        (
            [0, 2],
            [1, 1],
            [0, 0, 0, 0, 0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
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
    quantum_simulator._state_vector[:] = 1 / np.sqrt(8)
    quantum_simulator.measure_qubits(qubits, measurement)
    assert np.allclose(quantum_simulator.state_vector, state_vector)


@pytest.mark.parametrize(
    "qubits, result",
    (
        (0, [1 / 2, 0, 1 / 2, 0, 1 / 2, 0, 1 / 2, 0]),
        (1, [1 / 2, 1 / 2, 0, 0, 1 / 2, 1 / 2, 0, 0]),
        ([0, 1], [1 / np.sqrt(2), 0, 0, 0, 1 / np.sqrt(2), 0, 0, 0]),
        ([0, 2], [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0, 0, 0, 0, 0]),
        (range(3), [1, 0, 0, 0, 0, 0, 0, 0]),
    ),
)
def test_reset(qubits, result):
    quantum_simulator = QuantumSimulator()
    quantum_simulator.add_qubits(3)
    # hadamard all hack
    quantum_simulator._state_vector[:] = 1 / np.sqrt(8)
    quantum_simulator.reset_qubits(qubits)
    assert np.allclose(quantum_simulator.state_vector, result)


def test_execute_u():
    quantum_simulator = QuantumSimulator()
    quantum_simulator.add_qubits(3)
    quantum_simulator.execute_u(0, np.pi / 2, 0, np.pi)
    print(quantum_simulator.state_vector)
