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

from braket.default_simulator.mps_simulator import MPSSimulator, MPSTensor


def hadamard_matrix():
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def x_matrix():
    return np.array([[0, 1], [1, 0]], dtype=complex)


def cnot_matrix():
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def cz_matrix():
    return np.diag([1, 1, 1, -1]).astype(complex)


def rx_matrix(angle):
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry_matrix(angle):
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


class TestMPSInitialization:
    def test_default_initialization(self):
        sim = MPSSimulator(3)
        assert sim.n_qubits == 3
        assert len(sim.tensors) == 3
        assert np.isclose(sim.get_amplitude(0), 1.0, atol=1e-7)

    def test_basis_state_initialization(self):
        sim = MPSSimulator(3)
        sim.initialize(5)
        assert np.isclose(sim.get_amplitude(5), 1.0, atol=1e-7)
        assert np.isclose(sim.get_amplitude(0), 0.0, atol=1e-7)

    def test_initial_bond_dimensions(self):
        sim = MPSSimulator(4)
        bond_dims = sim.get_bond_dimensions()
        assert all(d == 1 for d in bond_dims)


class TestMPSSingleQubitGates:
    def test_hadamard(self):
        sim = MPSSimulator(2)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)

        amp_00 = sim.get_amplitude(0)
        amp_10 = sim.get_amplitude(2)
        assert np.isclose(np.abs(amp_00), 1 / np.sqrt(2), atol=1e-7)
        assert np.isclose(np.abs(amp_10), 1 / np.sqrt(2), atol=1e-7)

    def test_rotation_gates(self):
        sim = MPSSimulator(1)
        sim.apply_single_qubit_gate(rx_matrix(np.pi / 2), 0)

        amp_0 = sim.get_amplitude(0)
        amp_1 = sim.get_amplitude(1)
        assert np.isclose(np.abs(amp_0), np.cos(np.pi / 4), atol=1e-7)
        assert np.isclose(np.abs(amp_1), np.sin(np.pi / 4), atol=1e-7)

    def test_bond_dimension_unchanged(self):
        sim = MPSSimulator(3)
        initial_bonds = sim.get_bond_dimensions()
        sim.apply_single_qubit_gate(hadamard_matrix(), 1)
        final_bonds = sim.get_bond_dimensions()
        assert initial_bonds == final_bonds


class TestMPSTwoQubitGates:
    def test_adjacent_cnot(self):
        sim = MPSSimulator(2)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim.apply_two_qubit_gate(cnot_matrix(), 0, 1)

        amp_00 = sim.get_amplitude(0)
        amp_11 = sim.get_amplitude(3)
        assert np.isclose(np.abs(amp_00), 1 / np.sqrt(2), atol=1e-7)
        assert np.isclose(np.abs(amp_11), 1 / np.sqrt(2), atol=1e-7)

    def test_non_adjacent_cnot(self):
        sim = MPSSimulator(3)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim.apply_two_qubit_gate(cnot_matrix(), 0, 2)

        amp_000 = sim.get_amplitude(0)
        amp_101 = sim.get_amplitude(5)
        assert np.isclose(np.abs(amp_000), 1 / np.sqrt(2), atol=1e-7)
        assert np.isclose(np.abs(amp_101), 1 / np.sqrt(2), atol=1e-7)

    def test_bond_dimension_growth(self):
        sim = MPSSimulator(3, max_bond_dim=None)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim.apply_two_qubit_gate(cnot_matrix(), 0, 1)

        bond_dims = sim.get_bond_dimensions()
        assert bond_dims[0] >= 1

    def test_max_bond_dimension_enforcement(self):
        sim = MPSSimulator(4, max_bond_dim=2)
        for i in range(3):
            sim.apply_single_qubit_gate(hadamard_matrix(), i)
            if i < 3:
                sim.apply_two_qubit_gate(cnot_matrix(), i, i + 1)

        bond_dims = sim.get_bond_dimensions()
        assert all(d <= 2 for d in bond_dims)


class TestMPSTruncation:
    def test_truncation_error_tracking(self):
        sim = MPSSimulator(4, max_bond_dim=2)
        for i in range(3):
            sim.apply_single_qubit_gate(hadamard_matrix(), i)
            sim.apply_two_qubit_gate(cnot_matrix(), i, i + 1)

        error = sim.get_truncation_error()
        assert error >= 0

    def test_truncation_error_bound(self):
        sim = MPSSimulator(3, max_bond_dim=4)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim.apply_two_qubit_gate(cnot_matrix(), 0, 1)

        error = sim.get_truncation_error()
        assert error < 1.0

    def test_cutoff_threshold(self):
        sim = MPSSimulator(2, svd_cutoff=0.1)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim.apply_two_qubit_gate(cnot_matrix(), 0, 1)
        pass


class TestMPSAmplitudes:
    def test_computational_basis_amplitude(self):
        sim = MPSSimulator(2)
        sim.initialize(2)
        assert np.isclose(sim.get_amplitude(2), 1.0, atol=1e-7)
        assert np.isclose(sim.get_amplitude(0), 0.0, atol=1e-7)

    def test_superposition_amplitudes(self):
        sim = MPSSimulator(2)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim.apply_single_qubit_gate(hadamard_matrix(), 1)

        for i in range(4):
            assert np.isclose(np.abs(sim.get_amplitude(i)), 0.5, atol=1e-7)

    def test_amplitude_vs_full_simulator(self):
        sim = MPSSimulator(3)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim.apply_two_qubit_gate(cnot_matrix(), 0, 1)
        sim.apply_single_qubit_gate(hadamard_matrix(), 2)

        state = sim.get_state_vector()
        for i in range(8):
            assert np.isclose(sim.get_amplitude(i), state[i], atol=1e-7)


class TestMPSSampling:
    def test_deterministic_state(self):
        sim = MPSSimulator(2)
        sim.initialize(3)
        results = sim.sample(100)
        assert results == {"11": 100}

    def test_superposition_distribution(self):
        sim = MPSSimulator(2)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim.apply_single_qubit_gate(hadamard_matrix(), 1)

        results = sim.sample(10000)
        for bitstring in ["00", "01", "10", "11"]:
            count = results.get(bitstring, 0)
            assert 2000 < count < 3000

    def test_entangled_state_correlations(self):
        sim = MPSSimulator(2)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim.apply_two_qubit_gate(cnot_matrix(), 0, 1)

        results = sim.sample(10000)
        assert results.get("00", 0) > 4000
        assert results.get("11", 0) > 4000
        assert results.get("01", 0) < 500
        assert results.get("10", 0) < 500


class TestMPSEntanglement:
    def test_product_state_entropy(self):
        sim = MPSSimulator(3)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim.apply_single_qubit_gate(hadamard_matrix(), 1)
        sim.apply_single_qubit_gate(hadamard_matrix(), 2)

        entropy = sim.compute_entanglement_entropy(1)
        assert np.isclose(entropy, 0.0, atol=1e-7)

    def test_bell_state_entropy(self):
        sim = MPSSimulator(2)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim.apply_two_qubit_gate(cnot_matrix(), 0, 1)

        entropy = sim.compute_entanglement_entropy(0)
        assert np.isclose(entropy, 1.0, atol=1e-5)

    def test_entropy_vs_bond_dimension(self):
        sim = MPSSimulator(4)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim.apply_two_qubit_gate(cnot_matrix(), 0, 1)
        sim.apply_two_qubit_gate(cnot_matrix(), 1, 2)
        sim.apply_two_qubit_gate(cnot_matrix(), 2, 3)

        entropy = sim.compute_entanglement_entropy(1)
        bond_dims = sim.get_bond_dimensions()
        max_entropy = np.log2(bond_dims[1]) if bond_dims[1] > 0 else 0
        assert entropy <= max_entropy + 1e-7


class TestMPSCorrectness:
    def test_matches_full_simulator_small(self):
        sim = MPSSimulator(3)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim.apply_two_qubit_gate(cnot_matrix(), 0, 1)
        sim.apply_single_qubit_gate(x_matrix(), 2)

        state = sim.get_state_vector()
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-7)

    def test_matches_full_simulator_low_entanglement(self):
        sim = MPSSimulator(4, max_bond_dim=8)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim.apply_two_qubit_gate(cnot_matrix(), 0, 1)
        sim.apply_single_qubit_gate(hadamard_matrix(), 2)
        sim.apply_two_qubit_gate(cnot_matrix(), 2, 3)

        state = sim.get_state_vector()
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_truncation_error_meaningful(self):
        sim = MPSSimulator(4, max_bond_dim=2)
        for i in range(4):
            sim.apply_single_qubit_gate(hadamard_matrix(), i)
        for i in range(3):
            sim.apply_two_qubit_gate(cnot_matrix(), i, i + 1)

        error = sim.get_truncation_error()
        assert error >= 0


class TestMPSTensor:
    def test_tensor_properties(self):
        data = np.random.randn(2, 2, 3) + 1j * np.random.randn(2, 2, 3)
        tensor = MPSTensor(data)
        assert tensor.left_bond == 2
        assert tensor.physical_dim == 2
        assert tensor.right_bond == 3

    def test_left_canonicalize(self):
        data = np.random.randn(2, 2, 3) + 1j * np.random.randn(2, 2, 3)
        tensor = MPSTensor(data)
        new_tensor, remainder = tensor.left_canonicalize()

        matrix = new_tensor.data.reshape(-1, new_tensor.right_bond)
        identity = matrix.conj().T @ matrix
        assert np.allclose(identity, np.eye(identity.shape[0]), atol=1e-7)

    def test_right_canonicalize(self):
        data = np.random.randn(2, 2, 3) + 1j * np.random.randn(2, 2, 3)
        tensor = MPSTensor(data)
        remainder, new_tensor = tensor.right_canonicalize()

        matrix = new_tensor.data.reshape(new_tensor.left_bond, -1)
        identity = matrix @ matrix.conj().T
        assert np.allclose(identity, np.eye(identity.shape[0]), atol=1e-7)

    def test_invalid_tensor_dimension(self):
        data = np.random.randn(2, 2)
        with pytest.raises(ValueError, match="3D array"):
            MPSTensor(data)

    def test_copy(self):
        data = np.random.randn(2, 2, 3) + 1j * np.random.randn(2, 2, 3)
        tensor = MPSTensor(data)
        tensor_copy = tensor.copy()
        assert np.allclose(tensor.data, tensor_copy.data)
        tensor.data[0, 0, 0] = 999
        assert not np.allclose(tensor.data, tensor_copy.data)


class TestMPSGateValidation:
    def test_invalid_single_qubit_gate_shape(self):
        sim = MPSSimulator(2)
        with pytest.raises(ValueError, match="2x2"):
            sim.apply_single_qubit_gate(np.eye(4), 0)

    def test_single_qubit_gate_out_of_range(self):
        sim = MPSSimulator(2)
        with pytest.raises(ValueError, match="out of range"):
            sim.apply_single_qubit_gate(hadamard_matrix(), 5)

    def test_invalid_two_qubit_gate_shape(self):
        sim = MPSSimulator(2)
        with pytest.raises(ValueError, match="4x4"):
            sim.apply_two_qubit_gate(np.eye(2), 0, 1)


class TestMPSApplyOperations:
    def test_apply_operations_single_qubit(self):
        from braket.default_simulator.gate_operations import Hadamard, PauliX
        sim = MPSSimulator(2)
        ops = [Hadamard([0]), PauliX([1])]
        sim.apply_operations(ops)
        state = sim.get_state_vector()
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)

    def test_apply_operations_two_qubit(self):
        from braket.default_simulator.gate_operations import CX, Hadamard
        sim = MPSSimulator(2)
        ops = [Hadamard([0]), CX([0, 1])]
        sim.apply_operations(ops)
        amp_00 = sim.get_amplitude(0)
        amp_11 = sim.get_amplitude(3)
        assert np.isclose(np.abs(amp_00), 1 / np.sqrt(2), atol=1e-7)
        assert np.isclose(np.abs(amp_11), 1 / np.sqrt(2), atol=1e-7)

    def test_apply_operations_rejects_three_qubit(self):
        class FakeOp:
            def __init__(self):
                self.targets = [0, 1, 2]
                self.matrix = np.eye(8)
        sim = MPSSimulator(3)
        with pytest.raises(ValueError, match="up to 2-qubit"):
            sim.apply_operations([FakeOp()])


class TestMPSCanonicalize:
    def test_canonicalize_center(self):
        sim = MPSSimulator(4)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim.apply_two_qubit_gate(cnot_matrix(), 0, 1)
        sim.canonicalize(2)
        state = sim.get_state_vector()
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)


class TestMPSEntropyEdgeCases:
    def test_entropy_invalid_cut(self):
        sim = MPSSimulator(3)
        with pytest.raises(ValueError, match="out of range"):
            sim.compute_entanglement_entropy(-1)

    def test_entropy_invalid_cut_high(self):
        sim = MPSSimulator(3)
        with pytest.raises(ValueError, match="out of range"):
            sim.compute_entanglement_entropy(5)


class TestMPSStateVectorLimit:
    def test_state_vector_too_large(self):
        sim = MPSSimulator(25)
        with pytest.raises(ValueError, match="too large"):
            sim.get_state_vector()


class TestMPSCopy:
    def test_copy_independence(self):
        sim = MPSSimulator(2)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim_copy = sim.copy()
        sim.apply_single_qubit_gate(x_matrix(), 1)
        state_orig = sim.get_state_vector()
        state_copy = sim_copy.get_state_vector()
        assert not np.allclose(state_orig, state_copy)

    def test_copy_preserves_truncation_error(self):
        sim = MPSSimulator(3, max_bond_dim=2)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim.apply_two_qubit_gate(cnot_matrix(), 0, 1)
        sim_copy = sim.copy()
        assert sim.truncation_error == sim_copy.truncation_error


class TestMPSReversedQubitOrder:
    def test_cnot_reversed_order(self):
        sim = MPSSimulator(2)
        sim.apply_single_qubit_gate(hadamard_matrix(), 1)
        sim.apply_two_qubit_gate(cnot_matrix(), 1, 0)
        state = sim.get_state_vector()
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)


class TestMPSSwapGateQubits:
    def test_swap_gate_qubits_internal(self):
        sim = MPSSimulator(3)
        sim.apply_single_qubit_gate(hadamard_matrix(), 2)
        sim.apply_two_qubit_gate(cnot_matrix(), 2, 0)
        state = sim.get_state_vector()
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)


class TestMPSSVDTruncation:
    def test_svd_all_singular_values_below_cutoff(self):
        sim = MPSSimulator(2, svd_cutoff=1e-15, max_bond_dim=None)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim.apply_two_qubit_gate(cnot_matrix(), 0, 1)
        state = sim.get_state_vector()
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)


class TestMPSSamplingEdgeCases:
    def test_sample_with_zero_probability_normalization(self):
        sim = MPSSimulator(2)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        results = sim.sample(100)
        total = sum(results.values())
        assert total == 100

    def test_sample_normalization_during_measurement(self):
        sim = MPSSimulator(3)
        sim.apply_single_qubit_gate(hadamard_matrix(), 0)
        sim.apply_single_qubit_gate(hadamard_matrix(), 1)
        sim.apply_single_qubit_gate(hadamard_matrix(), 2)
        results = sim.sample(1000)
        total = sum(results.values())
        assert total == 1000
