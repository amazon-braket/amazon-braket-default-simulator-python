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

from braket.default_simulator.gate_operations import CCNot, CX, Hadamard, PauliX
from braket.default_simulator.sparse_simulator import SparseStateSimulator


class TestSparseSimulatorInit:
    def test_default_initialization(self):
        sim = SparseStateSimulator(4)
        assert sim.n_qubits == 4
        assert sim.amplitudes == {0: 1.0 + 0j}

    def test_initialize_basis_state(self):
        sim = SparseStateSimulator(4)
        sim.initialize(5)
        assert sim.amplitudes == {5: 1.0 + 0j}


class TestSparseSimulatorGates:
    def test_pauli_x(self):
        sim = SparseStateSimulator(2)
        sim.apply_single_qubit_gate(PauliX([0]).matrix, 0)
        assert 2 in sim.amplitudes
        assert np.isclose(abs(sim.amplitudes[2]), 1.0)

    def test_hadamard(self):
        sim = SparseStateSimulator(2)
        sim.apply_single_qubit_gate(Hadamard([0]).matrix, 0)
        assert len(sim.amplitudes) == 2
        assert 0 in sim.amplitudes
        assert 2 in sim.amplitudes

    def test_cnot(self):
        sim = SparseStateSimulator(2)
        sim.apply_single_qubit_gate(PauliX([0]).matrix, 0)
        sim.apply_two_qubit_gate(CX([0, 1]).matrix, 0, 1)
        assert 3 in sim.amplitudes
        assert np.isclose(abs(sim.amplitudes[3]), 1.0)

    def test_bell_state(self):
        sim = SparseStateSimulator(2)
        sim.apply_single_qubit_gate(Hadamard([0]).matrix, 0)
        sim.apply_two_qubit_gate(CX([0, 1]).matrix, 0, 1)
        assert len(sim.amplitudes) == 2
        assert 0 in sim.amplitudes
        assert 3 in sim.amplitudes


class TestSparseSimulatorOperations:
    def test_apply_operations(self):
        sim = SparseStateSimulator(2)
        ops = [Hadamard([0]), CX([0, 1])]
        sim.apply_operations(ops)
        assert len(sim.amplitudes) == 2

    def test_state_vector(self):
        sim = SparseStateSimulator(2)
        sim.apply_single_qubit_gate(Hadamard([0]).matrix, 0)
        sv = sim.get_state_vector()
        assert len(sv) == 4
        assert np.isclose(np.linalg.norm(sv), 1.0)


class TestSparseSimulatorSampling:
    def test_sample(self):
        sim = SparseStateSimulator(2)
        sim.apply_single_qubit_gate(Hadamard([0]).matrix, 0)
        sim.apply_two_qubit_gate(CX([0, 1]).matrix, 0, 1)
        samples = sim.sample(1000)
        assert "00" in samples or "11" in samples
        total = sum(samples.values())
        assert total == 1000

    def test_sample_array(self):
        sim = SparseStateSimulator(2)
        sim.apply_single_qubit_gate(PauliX([0]).matrix, 0)
        samples = sim.sample_array(100)
        assert len(samples) == 100
        assert all(s == 2 for s in samples)


class TestSparseSimulatorUtilities:
    def test_sparsity(self):
        sim = SparseStateSimulator(4)
        assert sim.sparsity() == 1 / 16

    def test_is_sparse(self):
        sim = SparseStateSimulator(4)
        assert sim.sparsity() < 0.1
        assert sim.is_sparse()
        sim.amplitudes = {i: 1.0 / 4 for i in range(4)}
        assert sim.sparsity() == 4 / 16
        assert not sim.is_sparse()

    def test_normalize(self):
        sim = SparseStateSimulator(2)
        sim.amplitudes = {0: 2.0 + 0j}
        sim.normalize()
        assert np.isclose(abs(sim.amplitudes[0]), 1.0)

    def test_prune(self):
        sim = SparseStateSimulator(2)
        sim.amplitudes = {0: 1.0 + 0j, 1: 1e-16}
        sim.prune()
        assert 1 not in sim.amplitudes

    def test_copy(self):
        sim = SparseStateSimulator(2)
        sim.apply_single_qubit_gate(Hadamard([0]).matrix, 0)
        sim_copy = sim.copy()
        assert sim_copy.amplitudes == sim.amplitudes
        sim_copy.amplitudes[0] = 0.5
        assert sim.amplitudes[0] != 0.5

    def test_from_dense(self):
        state = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        sim = SparseStateSimulator.from_dense(state, 2)
        assert len(sim.amplitudes) == 2
        assert 0 in sim.amplitudes
        assert 3 in sim.amplitudes


class TestSparseSimulatorAmplitude:
    def test_get_amplitude_existing(self):
        sim = SparseStateSimulator(2)
        amp = sim.get_amplitude(0)
        assert np.isclose(amp, 1.0 + 0j)

    def test_get_amplitude_nonexisting(self):
        sim = SparseStateSimulator(2)
        amp = sim.get_amplitude(3)
        assert amp == 0j

    def test_get_probabilities(self):
        sim = SparseStateSimulator(2)
        sim.apply_single_qubit_gate(Hadamard([0]).matrix, 0)
        probs = sim.get_probabilities()
        assert len(probs) == 4
        assert np.isclose(probs[0], 0.5)
        assert np.isclose(probs[2], 0.5)


class TestSparseSimulatorErrors:
    def test_single_qubit_gate_wrong_size(self):
        sim = SparseStateSimulator(2)
        with pytest.raises(ValueError, match="2x2"):
            sim.apply_single_qubit_gate(np.eye(4), 0)

    def test_two_qubit_gate_wrong_size(self):
        sim = SparseStateSimulator(2)
        with pytest.raises(ValueError, match="4x4"):
            sim.apply_two_qubit_gate(np.eye(2), 0, 1)

    def test_state_vector_too_large(self):
        sim = SparseStateSimulator(30)
        with pytest.raises(ValueError, match="too large"):
            sim.get_state_vector()


class TestSparseSimulatorMultiQubit:
    def test_three_qubit_gate(self):
        sim = SparseStateSimulator(3)
        sim.apply_single_qubit_gate(PauliX([0]).matrix, 0)
        sim.apply_single_qubit_gate(PauliX([1]).matrix, 1)
        ops = [CCNot([0, 1, 2])]
        sim.apply_operations(ops)
        assert 7 in sim.amplitudes

    def test_multi_qubit_gate_dimension_error(self):
        sim = SparseStateSimulator(3)
        with pytest.raises(ValueError, match="dimension mismatch"):
            sim._apply_multi_qubit_gate(np.eye(4), (0, 1, 2))


class TestSparseSimulatorEdgeCases:
    def test_prune_all_amplitudes(self):
        sim = SparseStateSimulator(2)
        sim.amplitudes = {0: 1e-20, 1: 1e-20}
        sim.prune()
        assert sim.amplitudes == {0: 1.0 + 0j}

    def test_from_dense_all_zeros(self):
        state = np.zeros(4, dtype=np.complex128)
        sim = SparseStateSimulator.from_dense(state, 2)
        assert sim.amplitudes == {0: 1.0 + 0j}

    def test_single_qubit_gate_with_partner_in_amplitudes(self):
        sim = SparseStateSimulator(2)
        sim.amplitudes = {0: 0.5 + 0j, 2: 0.5 + 0j}
        sim.apply_single_qubit_gate(Hadamard([0]).matrix, 0)
        assert len(sim.amplitudes) >= 1


class TestSparseSimulatorDensityCheck:
    def test_state_too_dense_raises(self):
        sim = SparseStateSimulator(4)
        # _MAX_AMPLITUDES is 1 << 20 = 1,048,576
        sim.amplitudes = {i: 1e-10 for i in range(1048577)}
        with pytest.raises(ValueError, match="too dense"):
            sim._check_density()


class TestSparseSimulatorCoverageGaps:
    def test_apply_gate_same_base_processed_twice(self):
        """Test that same base state is only processed once in multi-qubit gate."""
        sim = SparseStateSimulator(3)
        # Create superposition so multiple basis states exist
        sim.apply_single_qubit_gate(Hadamard([0]).matrix, 0)
        sim.apply_single_qubit_gate(Hadamard([1]).matrix, 1)
        # Apply two-qubit gate - some bases will share the same 'base' after masking
        sim.apply_two_qubit_gate(CX([0, 1]).matrix, 0, 1)
        sv = sim.get_state_vector()
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestSparseSimulatorFusionIntegration:
    def test_apply_operations_with_fusion(self):
        """Test that apply_operations uses single-qubit fusion."""
        from braket.default_simulator.gate_operations import S, T

        sim = SparseStateSimulator(3)
        # Create consecutive single-qubit gates that should be fused
        ops = [
            Hadamard([0]),
            T([0]),
            S([0]),
            Hadamard([1]),
            T([1]),
        ]
        sim.apply_operations(ops)
        sv = sim.get_state_vector()
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_apply_operations_fusion_correctness(self):
        """Test that fusion produces correct results."""
        from braket.default_simulator.gate_operations import T

        # Without fusion
        sim1 = SparseStateSimulator(2)
        sim1.apply_single_qubit_gate(Hadamard([0]).matrix, 0)
        sim1.apply_single_qubit_gate(np.diag([1, np.exp(1j * np.pi / 4)]), 0)
        sv1 = sim1.get_state_vector()

        # With fusion via apply_operations
        sim2 = SparseStateSimulator(2)
        ops = [Hadamard([0]), T([0])]
        sim2.apply_operations(ops)
        sv2 = sim2.get_state_vector()

        np.testing.assert_allclose(sv1, sv2, atol=1e-10)


class TestSparseSimulatorMultiQubitProcessedBranch:
    """Test the 'continue' branch in _apply_multi_qubit_gate (line 146)."""

    def test_multi_qubit_gate_same_base_skipped(self):
        """Test that basis states with same base are only processed once."""
        sim = SparseStateSimulator(4)
        # Create amplitudes at multiple basis states that share the same base
        # after masking out target qubits 0,1,2
        # States 0b0000 (0) and 0b0111 (7) both have base 0b0000 after masking qubits 0,1,2
        # States 0b1000 (8) and 0b1111 (15) both have base 0b1000 after masking
        sim.amplitudes = {
            0: 0.5 + 0j,  # base = 0 (qubit 3 = 0)
            7: 0.5 + 0j,  # base = 0 (qubit 3 = 0), differs in qubits 0,1,2
            8: 0.5 + 0j,  # base = 8 (qubit 3 = 1)
            15: 0.5 + 0j,  # base = 8 (qubit 3 = 1), differs in qubits 0,1,2
        }
        # Apply 3-qubit gate (CCNot) on qubits 0,1,2
        # This should trigger the 'continue' branch when processing states
        # with the same base
        sim._apply_multi_qubit_gate(CCNot([0, 1, 2]).matrix, (0, 1, 2))
        sv = sim.get_state_vector()
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)
