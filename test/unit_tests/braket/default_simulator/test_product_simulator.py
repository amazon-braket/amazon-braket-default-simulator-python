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

import numpy as np
import pytest

from braket.default_simulator.product_simulator import ProductStateSimulator


def hadamard_matrix():
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def x_matrix():
    return np.array([[0, 1], [1, 0]], dtype=complex)


def y_matrix():
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def z_matrix():
    return np.array([[1, 0], [0, -1]], dtype=complex)


def rx_matrix(angle):
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry_matrix(angle):
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz_matrix(angle):
    return np.array([[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]], dtype=complex)


def phase_shift_matrix(angle):
    return np.array([[1, 0], [0, np.exp(1j * angle)]], dtype=complex)


class TestProductStateInitialization:
    def test_default_initialization(self):
        sim = ProductStateSimulator(3)
        assert sim.n_qubits == 3
        for i in range(3):
            state = sim.get_qubit_state(i)
            assert np.allclose(state, [1, 0], atol=1e-7)

    def test_basis_state_initialization(self):
        sim = ProductStateSimulator(3)
        sim.initialize(5)
        assert np.isclose(sim.get_amplitude(5), 1.0, atol=1e-7)
        assert np.isclose(sim.get_amplitude(0), 0.0, atol=1e-7)

    def test_custom_product_state(self):
        sim = ProductStateSimulator(2)
        plus_state = np.array([1, 1], dtype=complex) / np.sqrt(2)
        minus_state = np.array([1, -1], dtype=complex) / np.sqrt(2)
        sim.initialize_from_product([plus_state, minus_state])

        state0 = sim.get_qubit_state(0)
        state1 = sim.get_qubit_state(1)
        assert np.allclose(state0, plus_state, atol=1e-7)
        assert np.allclose(state1, minus_state, atol=1e-7)


class TestProductStateGates:
    def test_hadamard(self):
        sim = ProductStateSimulator(1)
        sim.apply_gate(hadamard_matrix(), 0)
        state = sim.get_qubit_state(0)
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        assert np.allclose(state, expected, atol=1e-7)

    def test_pauli_x(self):
        sim = ProductStateSimulator(1)
        sim.apply_gate(x_matrix(), 0)
        state = sim.get_qubit_state(0)
        assert np.allclose(state, [0, 1], atol=1e-7)

    def test_pauli_y(self):
        sim = ProductStateSimulator(1)
        sim.apply_gate(y_matrix(), 0)
        state = sim.get_qubit_state(0)
        assert np.allclose(state, [0, 1j], atol=1e-7)

    def test_pauli_z(self):
        sim = ProductStateSimulator(1)
        sim.apply_gate(hadamard_matrix(), 0)
        sim.apply_gate(z_matrix(), 0)
        state = sim.get_qubit_state(0)
        expected = np.array([1, -1], dtype=complex) / np.sqrt(2)
        assert np.allclose(state, expected, atol=1e-7)

    def test_rotation_gates(self):
        sim = ProductStateSimulator(1)
        sim.apply_gate(rx_matrix(np.pi / 2), 0)
        state = sim.get_qubit_state(0)
        expected = np.array([np.cos(np.pi / 4), -1j * np.sin(np.pi / 4)], dtype=complex)
        assert np.allclose(state, expected, atol=1e-7)

    def test_phase_gates(self):
        sim = ProductStateSimulator(1)
        sim.apply_gate(x_matrix(), 0)
        sim.apply_gate(phase_shift_matrix(np.pi / 2), 0)
        state = sim.get_qubit_state(0)
        assert np.allclose(state, [0, 1j], atol=1e-7)

    def test_gate_sequence(self):
        sim = ProductStateSimulator(1)
        sim.apply_gate(hadamard_matrix(), 0)
        sim.apply_gate(z_matrix(), 0)
        sim.apply_gate(hadamard_matrix(), 0)
        state = sim.get_qubit_state(0)
        assert np.allclose(state, [0, 1], atol=1e-7)


class TestProductStateAmplitudes:
    def test_computational_basis_amplitude(self):
        sim = ProductStateSimulator(2)
        sim.initialize(2)
        assert np.isclose(sim.get_amplitude(2), 1.0, atol=1e-7)
        assert np.isclose(sim.get_amplitude(0), 0.0, atol=1e-7)
        assert np.isclose(sim.get_amplitude(1), 0.0, atol=1e-7)
        assert np.isclose(sim.get_amplitude(3), 0.0, atol=1e-7)

    def test_superposition_amplitudes(self):
        sim = ProductStateSimulator(2)
        sim.apply_gate(hadamard_matrix(), 0)
        sim.apply_gate(hadamard_matrix(), 1)

        for i in range(4):
            assert np.isclose(np.abs(sim.get_amplitude(i)), 0.5, atol=1e-7)

    def test_amplitude_normalization(self):
        sim = ProductStateSimulator(3)
        sim.apply_gate(hadamard_matrix(), 0)
        sim.apply_gate(hadamard_matrix(), 1)
        sim.apply_gate(hadamard_matrix(), 2)

        total_prob = sum(np.abs(sim.get_amplitude(i)) ** 2 for i in range(8))
        assert np.isclose(total_prob, 1.0, atol=1e-7)


class TestProductStateSampling:
    def test_deterministic_state(self):
        sim = ProductStateSimulator(2)
        sim.initialize(3)
        results = sim.sample(100)
        assert results == {"11": 100}

    def test_uniform_superposition_distribution(self):
        sim = ProductStateSimulator(2)
        sim.apply_gate(hadamard_matrix(), 0)
        sim.apply_gate(hadamard_matrix(), 1)

        results = sim.sample(10000)
        for bitstring in ["00", "01", "10", "11"]:
            count = results.get(bitstring, 0)
            assert 2000 < count < 3000

    def test_biased_distribution(self):
        sim = ProductStateSimulator(1)
        sim.apply_gate(ry_matrix(np.pi / 3), 0)

        results = sim.sample(10000)
        prob_0 = np.cos(np.pi / 6) ** 2
        expected_0 = int(prob_0 * 10000)
        actual_0 = results.get("0", 0)
        assert abs(actual_0 - expected_0) < 500

    def test_sample_count_accuracy(self):
        sim = ProductStateSimulator(2)
        sim.apply_gate(hadamard_matrix(), 0)
        results = sim.sample(1000)
        total = sum(results.values())
        assert total == 1000

    def test_sample_array_returns_numpy(self):
        sim = ProductStateSimulator(2)
        sim.apply_gate(hadamard_matrix(), 0)
        sim.apply_gate(hadamard_matrix(), 1)
        results = sim.sample_array(1000)
        assert isinstance(results, np.ndarray)
        assert len(results) == 1000
        assert results.dtype == np.int64

    def test_sample_array_deterministic(self):
        sim = ProductStateSimulator(2)
        sim.initialize(3)
        results = sim.sample_array(100)
        assert np.all(results == 3)


class TestProductStateBlochVector:
    def test_zero_state_bloch(self):
        sim = ProductStateSimulator(1)
        x, y, z = sim.get_bloch_vector(0)
        assert np.isclose(x, 0, atol=1e-7)
        assert np.isclose(y, 0, atol=1e-7)
        assert np.isclose(z, 1, atol=1e-7)

    def test_one_state_bloch(self):
        sim = ProductStateSimulator(1)
        sim.apply_gate(x_matrix(), 0)
        x, y, z = sim.get_bloch_vector(0)
        assert np.isclose(x, 0, atol=1e-7)
        assert np.isclose(y, 0, atol=1e-7)
        assert np.isclose(z, -1, atol=1e-7)

    def test_plus_state_bloch(self):
        sim = ProductStateSimulator(1)
        sim.apply_gate(hadamard_matrix(), 0)
        x, y, z = sim.get_bloch_vector(0)
        assert np.isclose(x, 1, atol=1e-7)
        assert np.isclose(y, 0, atol=1e-7)
        assert np.isclose(z, 0, atol=1e-7)

    def test_arbitrary_state_bloch(self):
        sim = ProductStateSimulator(1)
        sim.apply_gate(ry_matrix(np.pi / 4), 0)
        x, y, z = sim.get_bloch_vector(0)
        norm = np.sqrt(x**2 + y**2 + z**2)
        assert np.isclose(norm, 1, atol=1e-7)


class TestProductSimulatorCorrectness:
    def test_matches_full_simulator_small(self):
        sim = ProductStateSimulator(2)
        sim.apply_gate(hadamard_matrix(), 0)
        sim.apply_gate(x_matrix(), 1)

        state = sim.get_state_vector()
        expected = np.kron(np.array([1, 1]) / np.sqrt(2), np.array([0, 1]))
        assert np.allclose(state, expected, atol=1e-7)

    def test_matches_full_simulator_medium(self):
        sim = ProductStateSimulator(4)
        for i in range(4):
            sim.apply_gate(hadamard_matrix(), i)

        state = sim.get_state_vector()
        expected = np.ones(16) / 4
        assert np.allclose(state, expected, atol=1e-7)

    def test_random_circuits(self):
        np.random.seed(42)
        sim = ProductStateSimulator(3)

        gates = [hadamard_matrix(), x_matrix(), y_matrix(), z_matrix()]
        for _ in range(10):
            gate = gates[np.random.randint(4)]
            qubit = np.random.randint(3)
            sim.apply_gate(gate, qubit)

        state = sim.get_state_vector()
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-7)


class TestProductSimulatorErrors:
    def test_invalid_gate_shape(self):
        sim = ProductStateSimulator(2)
        with pytest.raises(ValueError, match="Only single-qubit gates"):
            sim.apply_gate(np.eye(4), 0)

    def test_qubit_out_of_range_negative(self):
        sim = ProductStateSimulator(2)
        with pytest.raises(ValueError, match="out of range"):
            sim.apply_gate(hadamard_matrix(), -1)

    def test_qubit_out_of_range_high(self):
        sim = ProductStateSimulator(2)
        with pytest.raises(ValueError, match="out of range"):
            sim.apply_gate(hadamard_matrix(), 5)

    def test_invalid_product_state_count(self):
        sim = ProductStateSimulator(2)
        with pytest.raises(ValueError, match="Expected 2 states"):
            sim.initialize_from_product([np.array([1, 0])])

    def test_invalid_product_state_shape(self):
        sim = ProductStateSimulator(2)
        with pytest.raises(ValueError, match="must have shape"):
            sim.initialize_from_product([np.array([1, 0, 0]), np.array([1, 0])])

    def test_get_qubit_state_out_of_range(self):
        sim = ProductStateSimulator(2)
        with pytest.raises(ValueError, match="out of range"):
            sim.get_qubit_state(5)


class TestProductSimulatorCopy:
    def test_copy_independence(self):
        sim = ProductStateSimulator(2)
        sim.apply_gate(hadamard_matrix(), 0)
        sim_copy = sim.copy()
        sim.apply_gate(x_matrix(), 1)
        state_orig = sim.get_qubit_state(1)
        state_copy = sim_copy.get_qubit_state(1)
        assert not np.allclose(state_orig, state_copy)

    def test_copy_preserves_state(self):
        sim = ProductStateSimulator(2)
        sim.apply_gate(hadamard_matrix(), 0)
        sim.apply_gate(x_matrix(), 1)
        sim_copy = sim.copy()
        assert np.allclose(sim.get_state_vector(), sim_copy.get_state_vector())


class TestProductSimulatorProbabilities:
    def test_get_probabilities(self):
        sim = ProductStateSimulator(2)
        sim.apply_gate(hadamard_matrix(), 0)
        probs = sim.get_probabilities()
        assert len(probs) == 4
        assert np.isclose(probs[0], 0.5, atol=1e-7)
        assert np.isclose(probs[1], 0.0, atol=1e-7)
        assert np.isclose(probs[2], 0.5, atol=1e-7)
        assert np.isclose(probs[3], 0.0, atol=1e-7)

    def test_get_probabilities_uniform(self):
        sim = ProductStateSimulator(2)
        sim.apply_gate(hadamard_matrix(), 0)
        sim.apply_gate(hadamard_matrix(), 1)
        probs = sim.get_probabilities()
        assert len(probs) == 4
        for p in probs:
            assert np.isclose(p, 0.25, atol=1e-7)


class TestProductSimulatorApplyOperations:
    def test_apply_operations_single_gate(self):
        from braket.default_simulator.gate_operations import Hadamard

        sim = ProductStateSimulator(2)
        ops = [Hadamard([0])]
        sim.apply_operations(ops)
        state = sim.get_qubit_state(0)
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        assert np.allclose(state, expected, atol=1e-7)

    def test_apply_operations_multiple_gates(self):
        from braket.default_simulator.gate_operations import Hadamard, PauliX

        sim = ProductStateSimulator(2)
        ops = [Hadamard([0]), PauliX([1])]
        sim.apply_operations(ops)
        state0 = sim.get_qubit_state(0)
        state1 = sim.get_qubit_state(1)
        assert np.allclose(state0, np.array([1, 1]) / np.sqrt(2), atol=1e-7)
        assert np.allclose(state1, np.array([0, 1]), atol=1e-7)

    def test_apply_operations_rejects_two_qubit_gate(self):
        from braket.default_simulator.gate_operations import CX

        sim = ProductStateSimulator(2)
        ops = [CX([0, 1])]
        with pytest.raises(ValueError, match="single-qubit gates"):
            sim.apply_operations(ops)


class TestProductSimulatorNumbaThreshold:
    def test_amplitude_above_threshold(self):
        n_qubits = 14
        sim = ProductStateSimulator(n_qubits)
        sim.initialize(0)
        for i in range(n_qubits):
            sim.apply_gate(hadamard_matrix(), i)
        amp = sim.get_amplitude(0)
        expected = 1.0 / (2 ** (n_qubits / 2))
        assert np.isclose(np.abs(amp), expected, atol=1e-7)

    def test_amplitude_at_threshold(self):
        n_qubits = 12
        sim = ProductStateSimulator(n_qubits)
        for i in range(n_qubits):
            sim.apply_gate(hadamard_matrix(), i)
        amp = sim.get_amplitude(0)
        expected = 1.0 / (2 ** (n_qubits / 2))
        assert np.isclose(np.abs(amp), expected, atol=1e-7)

    def test_state_vector_above_threshold(self):
        n_qubits = 14
        sim = ProductStateSimulator(n_qubits)
        for i in range(n_qubits):
            sim.apply_gate(hadamard_matrix(), i)
        state = sim.get_state_vector()
        assert len(state) == 2**n_qubits
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)
        expected_amp = 1.0 / (2 ** (n_qubits / 2))
        assert np.allclose(np.abs(state), expected_amp, atol=1e-7)

    def test_state_vector_at_threshold(self):
        n_qubits = 12
        sim = ProductStateSimulator(n_qubits)
        for i in range(n_qubits):
            sim.apply_gate(hadamard_matrix(), i)
        state = sim.get_state_vector()
        assert len(state) == 2**n_qubits
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)

    def test_state_vector_too_large(self):
        sim = ProductStateSimulator(21)
        with pytest.raises(ValueError, match="too large"):
            sim.get_state_vector()

    def test_basis_state_amplitude_above_threshold(self):
        n_qubits = 14
        sim = ProductStateSimulator(n_qubits)
        basis = 5461
        sim.initialize(basis)
        assert np.isclose(sim.get_amplitude(basis), 1.0, atol=1e-7)
        assert np.isclose(sim.get_amplitude(0), 0.0, atol=1e-7)


class TestProductSimulatorQFT:
    def test_apply_qft_normalization(self):
        sim = ProductStateSimulator(4)
        sim.initialize(0)
        sim.apply_qft()
        state = sim.get_state_vector()
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)

    def test_apply_qft_uniform_probabilities(self):
        sim = ProductStateSimulator(4)
        sim.initialize(0)
        sim.apply_qft()
        state = sim.get_state_vector()
        probs = np.abs(state) ** 2
        expected = np.full(16, 1.0 / 16)
        np.testing.assert_allclose(probs, expected, atol=1e-7)

    def test_apply_qft_with_basis_state(self):
        sim = ProductStateSimulator(3)
        sim.initialize(5)
        sim.apply_qft()
        state = sim.get_state_vector()
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)

    def test_apply_inverse_qft_normalization(self):
        sim = ProductStateSimulator(4)
        sim.initialize(3)
        sim.apply_inverse_qft()
        state = sim.get_state_vector()
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)

    def test_partial_qft(self):
        sim = ProductStateSimulator(4)
        sim.initialize(0)
        sim.apply_qft(start_qubit=1, end_qubit=3)
        state = sim.get_state_vector()
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)

    def test_qft_sampling(self):
        sim = ProductStateSimulator(4)
        sim.initialize(0)
        sim.apply_qft()
        samples = sim.sample(1000)
        total = sum(samples.values())
        assert total == 1000


class TestProductSimulatorCoverageGaps:
    def test_inverse_qft_default_end_qubit(self):
        """Test apply_inverse_qft with default end_qubit (None)."""
        sim = ProductStateSimulator(4)
        sim.initialize(0)
        sim.apply_qft()
        sim.apply_inverse_qft()  # end_qubit defaults to None -> n_qubits
        state = sim.get_state_vector()
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)

    def test_get_probabilities_numba_path(self):
        """Test get_probabilities with >= 12 qubits to trigger Numba path."""
        n_qubits = 12
        sim = ProductStateSimulator(n_qubits)
        for i in range(n_qubits):
            sim.apply_gate(hadamard_matrix(), i)
        probs = sim.get_probabilities()
        assert len(probs) == 2**n_qubits
        assert np.isclose(np.sum(probs), 1.0, atol=1e-7)
        expected = 1.0 / (2**n_qubits)
        assert np.allclose(probs, expected, atol=1e-7)


class TestProductSimulatorInverseQFTBranch:
    """Test branch coverage for apply_inverse_qft."""

    def test_inverse_qft_explicit_end_qubit(self):
        """Test apply_inverse_qft with explicit end_qubit parameter (line 179->181)."""
        sim = ProductStateSimulator(4)
        sim.initialize(0)
        sim.apply_qft(start_qubit=0, end_qubit=4)
        # Apply inverse QFT with explicit end_qubit (not None)
        sim.apply_inverse_qft(start_qubit=0, end_qubit=4)
        state = sim.get_state_vector()
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)

    def test_inverse_qft_partial_explicit_end(self):
        """Test partial inverse QFT with explicit end_qubit."""
        sim = ProductStateSimulator(6)
        sim.initialize(0)
        sim.apply_qft(start_qubit=1, end_qubit=4)
        # Apply inverse QFT with explicit end_qubit
        sim.apply_inverse_qft(start_qubit=1, end_qubit=4)
        state = sim.get_state_vector()
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)
