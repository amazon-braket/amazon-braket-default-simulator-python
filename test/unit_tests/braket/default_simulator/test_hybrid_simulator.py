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

from braket.default_simulator.gate_operations import (
    CPhaseShift,
    CX,
    CZ,
    Hadamard,
    PauliX,
    PauliZ,
    RotX,
    RotZ,
    S,
    T,
)
from braket.default_simulator.hybrid_simulation import HybridSimulation
from braket.default_simulator.hybrid_simulator import HybridSimulator


class TestHybridSimulatorInit:
    def test_device_id(self):
        sim = HybridSimulator()
        assert sim.DEVICE_ID == "braket_hybrid"

    def test_properties_qubit_count(self):
        sim = HybridSimulator()
        assert sim.properties.paradigm.qubitCount == 26

    def test_initialize_simulation_default(self):
        sim = HybridSimulator()
        simulation = sim.initialize_simulation(qubit_count=4, shots=100)
        assert isinstance(simulation, HybridSimulation)
        assert simulation.qubit_count == 4
        assert simulation.shots == 100


class TestHybridSimulationInit:
    def test_default_initialization(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        assert sim.qubit_count == 4
        assert sim.shots == 100

    def test_initial_state_vector(self):
        sim = HybridSimulation(qubit_count=2, shots=100)
        expected = np.array([1, 0, 0, 0], dtype=np.complex128)
        np.testing.assert_allclose(sim.state_vector, expected, atol=1e-7)


class TestBackendSelection:
    def test_product_backend_selection(self):
        sim = HybridSimulation(qubit_count=3, shots=10)
        ops = [Hadamard([0]), PauliX([1]), RotZ([2], np.pi / 4)]
        sim.evolve(ops)
        assert sim.get_last_backend() == "product"

    def test_clifford_backend_selection(self):
        sim = HybridSimulation(qubit_count=2, shots=10)
        ops = [Hadamard([0]), CX([0, 1]), S([1])]
        sim.evolve(ops)
        assert sim.get_last_backend() == "clifford"

    def test_force_backend_override(self):
        sim = HybridSimulation(qubit_count=3, shots=10, force_backend="full")
        ops = [Hadamard([0]), PauliX([1])]
        sim.evolve(ops)
        assert sim.get_last_backend() == "full"

    def test_auto_select_disabled(self):
        sim = HybridSimulation(qubit_count=3, shots=10, auto_select=False)
        ops = [Hadamard([0]), PauliX([1])]
        sim.evolve(ops)
        assert sim.get_last_backend() == "full"


class TestEvolve:
    def test_empty_operations(self):
        sim = HybridSimulation(qubit_count=2, shots=100)
        sim.evolve([])
        expected = np.array([1, 0, 0, 0], dtype=np.complex128)
        np.testing.assert_allclose(sim.state_vector, expected, atol=1e-7)

    def test_single_hadamard(self):
        sim = HybridSimulation(qubit_count=1, shots=100)
        sim.evolve([Hadamard([0])])
        expected = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
        np.testing.assert_allclose(sim.state_vector, expected, atol=1e-7)

    def test_bell_state(self):
        sim = HybridSimulation(qubit_count=2, shots=100)
        sim.evolve([Hadamard([0]), CX([0, 1])])
        expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        np.testing.assert_allclose(sim.state_vector, expected, atol=1e-7)

    def test_pauli_x(self):
        sim = HybridSimulation(qubit_count=1, shots=100)
        sim.evolve([PauliX([0])])
        expected = np.array([0, 1], dtype=np.complex128)
        np.testing.assert_allclose(sim.state_vector, expected, atol=1e-7)

    def test_state_normalization(self):
        sim = HybridSimulation(qubit_count=3, shots=100)
        sim.evolve([Hadamard([0]), Hadamard([1]), Hadamard([2])])
        norm = np.sum(np.abs(sim.state_vector) ** 2)
        np.testing.assert_allclose(norm, 1.0, atol=1e-7)


class TestRetrieveSamples:
    def test_deterministic_state(self):
        sim = HybridSimulation(qubit_count=2, shots=100)
        sim.evolve([PauliX([0]), PauliX([1])])
        samples = sim.retrieve_samples()
        assert len(samples) == 100
        assert all(s == 3 for s in samples)

    def test_superposition_samples(self):
        sim = HybridSimulation(qubit_count=1, shots=1000)
        sim.evolve([Hadamard([0])])
        samples = sim.retrieve_samples()
        assert len(samples) == 1000
        zeros = sum(1 for s in samples if s == 0)
        ones = sum(1 for s in samples if s == 1)
        assert 400 < zeros < 600
        assert 400 < ones < 600


class TestProbabilities:
    def test_ground_state_probabilities(self):
        sim = HybridSimulation(qubit_count=2, shots=100)
        probs = sim.probabilities
        expected = np.array([1, 0, 0, 0])
        np.testing.assert_allclose(probs, expected, atol=1e-7)

    def test_superposition_probabilities(self):
        sim = HybridSimulation(qubit_count=1, shots=100)
        sim.evolve([Hadamard([0])])
        probs = sim.probabilities
        expected = np.array([0.5, 0.5])
        np.testing.assert_allclose(probs, expected, atol=1e-7)


class TestDensityMatrix:
    def test_pure_state_density_matrix(self):
        sim = HybridSimulation(qubit_count=1, shots=100)
        dm = sim.density_matrix
        expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        np.testing.assert_allclose(dm, expected, atol=1e-7)


class TestCliffordBackend:
    def test_clifford_sampling(self):
        sim = HybridSimulation(qubit_count=2, shots=1000)
        sim.evolve([Hadamard([0]), CX([0, 1])])
        samples = sim.retrieve_samples()
        assert len(samples) == 1000
        zeros = sum(1 for s in samples if s == 0)
        threes = sum(1 for s in samples if s == 3)
        assert zeros + threes > 900


class TestFullBackend:
    def test_full_state_vector(self):
        sim = HybridSimulation(qubit_count=2, shots=100, force_backend="full")
        sim.evolve([Hadamard([0]), CX([0, 1])])
        expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        np.testing.assert_allclose(sim.state_vector, expected, atol=1e-7)


class TestEdgeCases:
    def test_single_qubit_system(self):
        sim = HybridSimulation(qubit_count=1, shots=100)
        sim.evolve([Hadamard([0])])
        samples = sim.retrieve_samples()
        assert len(samples) == 100

    def test_multiple_evolve_calls(self):
        sim = HybridSimulation(qubit_count=2, shots=100, force_backend="full")
        sim.evolve([Hadamard([0])])
        sim.evolve([CX([0, 1])])
        expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        np.testing.assert_allclose(sim.state_vector, expected, atol=1e-7)

    def test_rotation_gates(self):
        sim = HybridSimulation(qubit_count=2, shots=100)
        sim.evolve([RotX([0], np.pi / 4), RotZ([1], np.pi / 3)])
        norm = np.sum(np.abs(sim.state_vector) ** 2)
        np.testing.assert_allclose(norm, 1.0, atol=1e-7)

    def test_diagonal_circuit(self):
        sim = HybridSimulation(qubit_count=2, shots=100)
        sim.evolve([PauliZ([0]), S([1]), CZ([0, 1])])
        samples = sim.retrieve_samples()
        assert all(s == 0 for s in samples)


class TestProductBackend:
    def test_product_backend_with_cphaseshift(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [
            Hadamard([0]), Hadamard([1]), Hadamard([2]), Hadamard([3]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() == "product"

    def test_product_probabilities(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [
            Hadamard([0]), Hadamard([1]), Hadamard([2]), Hadamard([3]),
        ]
        sim.evolve(ops)
        probs = sim.probabilities
        assert len(probs) == 16
        assert np.isclose(np.sum(probs), 1.0, atol=1e-7)

    def test_product_sampling(self):
        sim = HybridSimulation(qubit_count=4, shots=1000)
        ops = [
            Hadamard([0]), Hadamard([1]), Hadamard([2]), Hadamard([3]),
        ]
        sim.evolve(ops)
        samples = sim.retrieve_samples()
        assert len(samples) == 1000


class TestCliffordSampling:
    def test_clifford_retrieve_samples(self):
        sim = HybridSimulation(qubit_count=2, shots=100)
        sim.evolve([Hadamard([0]), CX([0, 1])])
        samples = sim.retrieve_samples()
        assert len(samples) == 100
        for s in samples:
            assert s in [0, 3]


class TestMaterializeBackends:
    def test_materialize_clifford(self):
        sim = HybridSimulation(qubit_count=2, shots=100)
        sim.evolve([Hadamard([0]), CX([0, 1])])
        sim.evolve([T([0])])
        state = sim.state_vector
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)

    def test_materialize_product(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [
            Hadamard([0]), Hadamard([1]), Hadamard([2]), Hadamard([3]),
        ]
        sim.evolve(ops)
        sim.evolve([CX([0, 1])])
        state = sim.state_vector
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)


class TestApplyObservables:
    def test_apply_observables(self):
        from braket.default_simulator.observables import PauliZ as ObsPauliZ
        sim = HybridSimulation(qubit_count=2, shots=100)
        sim.evolve([Hadamard([0])])
        obs = ObsPauliZ([0])
        sim.apply_observables([obs])
        state = sim.state_with_observables
        assert len(state) == 4

    def test_apply_observables_twice_raises(self):
        from braket.default_simulator.observables import PauliZ as ObsPauliZ
        sim = HybridSimulation(qubit_count=2, shots=100)
        sim.evolve([Hadamard([0])])
        obs = ObsPauliZ([0])
        sim.apply_observables([obs])
        with pytest.raises(RuntimeError, match="already been applied"):
            sim.apply_observables([obs])

    def test_state_with_observables_not_applied(self):
        sim = HybridSimulation(qubit_count=2, shots=100)
        with pytest.raises(RuntimeError, match="No observables applied"):
            _ = sim.state_with_observables


class TestExpectation:
    def test_expectation_z(self):
        from braket.default_simulator.observables import PauliZ as ObsPauliZ
        sim = HybridSimulation(qubit_count=1, shots=100)
        obs = ObsPauliZ([0])
        exp = sim.expectation(obs)
        assert np.isclose(exp, 1.0, atol=1e-7)

    def test_expectation_x_superposition(self):
        from braket.default_simulator.observables import PauliX as ObsPauliX
        sim = HybridSimulation(qubit_count=1, shots=100)
        sim.evolve([Hadamard([0])])
        obs = ObsPauliX([0])
        exp = sim.expectation(obs)
        assert np.isclose(exp, 1.0, atol=1e-7)


class TestFastClassifyEdgeCases:
    def test_classify_non_clifford_two_qubit(self):
        from braket.default_simulator.gate_operations import CPhaseShift
        sim = HybridSimulation(qubit_count=2, shots=100)
        ops = [CPhaseShift([0, 1], np.pi / 3)]
        sim.evolve(ops)
        assert sim.get_last_backend() == "full"

    def test_classify_swap_uses_clifford(self):
        from braket.default_simulator.gate_operations import Swap
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [
            Hadamard([0]), Hadamard([1]), Hadamard([2]), Hadamard([3]),
            Swap([0, 3]), CX([1, 2]), CX([0, 1]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() == "clifford"

    def test_classify_single_qubit_non_clifford(self):
        sim = HybridSimulation(qubit_count=2, shots=100)
        ops = [T([0]), T([1])]
        sim.evolve(ops)
        assert sim.get_last_backend() == "product"

    def test_backend_flags_invalidated_after_non_clifford(self):
        sim = HybridSimulation(qubit_count=2, shots=100)
        ops = [Hadamard([0]), CX([0, 1])]
        sim.evolve(ops)
        assert sim._is_clifford is True
        ops2 = [RotX([0], np.pi / 3), CX([0, 1])]
        sim.evolve(ops2)
        assert sim._is_clifford is False
        assert sim._is_product is False

    def test_product_after_clifford_invalidation(self):
        sim = HybridSimulation(qubit_count=2, shots=100)
        sim.evolve([Hadamard([0]), CX([0, 1])])
        sim.evolve([Hadamard([0])])
        assert sim.get_last_backend() == "product"


class TestQFTBackend:
    def test_qft_backend_selection(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [
            PauliX([0]),
            Hadamard([0]), Hadamard([1]), Hadamard([2]), Hadamard([3]),
            CPhaseShift([0, 1], np.pi / 2),
            CPhaseShift([0, 2], np.pi / 4),
            CPhaseShift([1, 2], np.pi / 2),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() == "qft"

    def test_qft_sampling(self):
        sim = HybridSimulation(qubit_count=4, shots=1000)
        ops = [
            Hadamard([0]), Hadamard([1]), Hadamard([2]), Hadamard([3]),
            CPhaseShift([0, 1], np.pi / 2),
            CPhaseShift([0, 2], np.pi / 4),
            CPhaseShift([1, 2], np.pi / 2),
        ]
        sim.evolve(ops)
        samples = sim.retrieve_samples()
        assert len(samples) == 1000

    def test_qft_state_normalization(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [
            Hadamard([0]), Hadamard([1]), Hadamard([2]), Hadamard([3]),
            CPhaseShift([0, 1], np.pi / 2),
            CPhaseShift([0, 2], np.pi / 4),
            CPhaseShift([1, 2], np.pi / 2),
        ]
        sim.evolve(ops)
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestPhaseEstimationCircuit:
    def test_phase_estimation_uses_full_backend(self):
        sim = HybridSimulation(qubit_count=4, shots=100, force_backend="full")
        ops = [
            Hadamard([0]),
            Hadamard([1]),
            Hadamard([2]),
            CX([0, 3]),
            CPhaseShift([1, 3], np.pi / 2),
            T([0]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() == "full"

    def test_phase_estimation_state_normalization(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [
            Hadamard([0]),
            Hadamard([1]),
            Hadamard([2]),
            CX([0, 3]),
            CX([1, 3]),
            CX([2, 3]),
        ]
        sim.evolve(ops)
        state = sim.state_vector
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-7)


class TestPartitionedBackend:
    def test_partitioned_backend_selection(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = [
            RotX([0], np.pi / 3), CX([0, 1]),
            RotX([3], np.pi / 4), CX([3, 4]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"

    def test_partitioned_state_vector(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = [
            RotX([0], np.pi / 3), CX([0, 1]),
            RotX([3], np.pi / 4), CX([3, 4]),
        ]
        sim.evolve(ops)
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_partitioned_sampling(self):
        sim = HybridSimulation(qubit_count=6, shots=1000)
        ops = [
            RotX([0], np.pi / 3), CX([0, 1]),
            RotX([3], np.pi / 4), CX([3, 4]),
        ]
        sim.evolve(ops)
        samples = sim.retrieve_samples()
        assert len(samples) == 1000

    def test_partitioned_probabilities(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = [
            RotX([0], np.pi / 3), CX([0, 1]),
            RotX([3], np.pi / 4), CX([3, 4]),
        ]
        sim.evolve(ops)
        probs = sim.probabilities
        assert np.isclose(np.sum(probs), 1.0, atol=1e-7)

    def test_partitioned_mixed_backends(self):
        sim = HybridSimulation(qubit_count=8, shots=100)
        ops = [
            Hadamard([0]), CX([0, 1]),
            Hadamard([3]),
            RotX([5], np.pi / 4), RotZ([6], np.pi / 3), CX([5, 6]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_partitioned_deterministic_sampling(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = [
            PauliX([0]), PauliX([1]),
            PauliX([4]), PauliX([5]),
        ]
        sim.evolve(ops)
        samples = sim.retrieve_samples()
        assert all(s == 0b110011 for s in samples)

    def test_partitioned_not_used_for_small_circuits(self):
        sim = HybridSimulation(qubit_count=3, shots=100)
        ops = [
            RotX([0], np.pi / 3), CX([0, 1]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() != "partitioned"

    def test_partitioned_not_used_when_fully_connected(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = [
            RotX([0], np.pi / 3), CX([0, 1]), CX([1, 2]),
            CX([2, 3]), CX([3, 4]), CX([4, 5]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() != "partitioned"


class TestTemporalBlockDecomposition:
    def test_detect_terminal_qft(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [
            PauliX([0]),
            CX([0, 1]),
            Hadamard([0]), Hadamard([1]),
            CPhaseShift([0, 1], np.pi / 2),
        ]
        pre_qft, qft_qubits = sim._detect_terminal_qft(ops)
        assert qft_qubits is not None
        assert set(qft_qubits) == {0, 1}
        assert len(pre_qft) == 2

    def test_terminal_qft_sampling(self):
        sim = HybridSimulation(qubit_count=4, shots=1000)
        ops = [
            PauliX([0]),
            CX([0, 1]),
            Hadamard([0]), Hadamard([1]),
            CPhaseShift([0, 1], np.pi / 2),
        ]
        sim.evolve(ops)
        samples = sim.retrieve_samples()
        assert len(samples) == 1000

    def test_terminal_qft_state_vector(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [
            PauliX([0]),
            CX([0, 1]),
            Hadamard([0]), Hadamard([1]),
            CPhaseShift([0, 1], np.pi / 2),
        ]
        sim.evolve(ops)
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_no_terminal_qft_when_not_at_end(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [
            Hadamard([0]), Hadamard([1]),
            CPhaseShift([0, 1], np.pi / 2),
            PauliX([2]),
            CX([2, 3]),
        ]
        pre_qft, qft_qubits = sim._detect_terminal_qft(ops)
        assert qft_qubits is None

    def test_block_detection(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [
            Hadamard([0]), Hadamard([1]), Hadamard([2]),
            CX([0, 1]), CX([1, 2]),
            RotX([0], np.pi / 4), RotX([1], np.pi / 4),
        ]
        blocks = sim._detect_temporal_blocks(ops)
        assert len(blocks) >= 1


class TestBlockDetection:
    def test_detect_temporal_blocks_small_circuit(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [Hadamard([0]), PauliX([1])]
        blocks = sim._detect_temporal_blocks(ops)
        assert len(blocks) == 1
        assert blocks[0][0] == "full"

    def test_classify_single_op_product(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        op = Hadamard([0])
        result = sim._classify_single_op(op)
        assert result == "product"

    def test_classify_single_op_clifford(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        op = CX([0, 1])
        result = sim._classify_single_op(op)
        assert result == "clifford"

    def test_classify_single_op_qft_gate(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        op = CPhaseShift([0, 1], np.pi / 2)
        result = sim._classify_single_op(op)
        assert result == "qft_gate"

    def test_can_merge_blocks_product_clifford(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        assert sim._can_merge_blocks("product", "clifford") is True
        assert sim._can_merge_blocks("product", "qft_gate") is True
        assert sim._can_merge_blocks("clifford", "product") is True
        assert sim._can_merge_blocks("qft_gate", "product") is True
        assert sim._can_merge_blocks("full", "clifford") is False

    def test_merged_block_type(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        assert sim._merged_block_type("product", "clifford") == "clifford"
        assert sim._merged_block_type("product", "qft_gate") == "qft_gate"
        assert sim._merged_block_type("product", "product") == "product"

    def test_merge_adjacent_full_blocks_empty(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        result = sim._merge_adjacent_full_blocks([])
        assert result == []


class TestTerminalQFTCoverage:
    def test_terminal_qft_with_pre_ops(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [
            PauliX([0]), PauliX([1]),
            CX([0, 1]),
            Hadamard([0]), Hadamard([1]),
            CPhaseShift([0, 1], np.pi / 2),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() == "terminal_qft"
        samples = sim.retrieve_samples()
        assert len(samples) == 100

    def test_terminal_qft_probabilities(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [
            CX([0, 1]),
            Hadamard([0]), Hadamard([1]),
            CPhaseShift([0, 1], np.pi / 2),
        ]
        sim.evolve(ops)
        probs = sim.probabilities
        assert np.isclose(np.sum(probs), 1.0, atol=1e-7)

    def test_detect_terminal_qft_too_few_ops(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [Hadamard([0])]
        pre_qft, qft_qubits = sim._detect_terminal_qft(ops)
        assert qft_qubits is None

    def test_detect_terminal_qft_no_cphase(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [
            PauliX([0]), PauliX([1]), PauliX([2]), PauliX([3]),
            Hadamard([0]), Hadamard([1]),
        ]
        pre_qft, qft_qubits = sim._detect_terminal_qft(ops)
        assert qft_qubits is None


class TestStabilizerIntegration:
    def test_stabilizer_in_partitioned_simulation(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = [
            Hadamard([0]), CX([0, 1]),
            Hadamard([3]), CX([3, 4]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_stabilizer_sampling_in_partition(self):
        sim = HybridSimulation(qubit_count=6, shots=1000)
        ops = [
            Hadamard([0]), CX([0, 1]),
            Hadamard([3]), CX([3, 4]),
        ]
        sim.evolve(ops)
        samples = sim.retrieve_samples()
        assert len(samples) == 1000


class TestPartitionedSimulationCoverage:
    def test_partitioned_with_full_backend_partition(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = [
            RotX([0], np.pi / 3), CX([0, 1]),
            RotX([3], np.pi / 4), CX([3, 4]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        samples = sim.retrieve_samples()
        assert len(samples) == 100

    def test_partitioned_materialize_clifford(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = [
            Hadamard([0]), CX([0, 1]),
            Hadamard([3]),
        ]
        sim.evolve(ops)
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_partitioned_materialize_product(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = [
            Hadamard([0]),
            Hadamard([3]),
        ]
        sim.evolve(ops)
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestMarginalProbability:
    def test_marginal_prob_0(self):
        sim = HybridSimulation(qubit_count=2, shots=100)
        sim.evolve([Hadamard([0])])
        sv = sim.state_vector
        prob_0 = sim._marginal_prob_0(sv, 0)
        assert np.isclose(prob_0, 0.5, atol=0.01)

    def test_marginal_prob_0_deterministic(self):
        sim = HybridSimulation(qubit_count=2, shots=100)
        sim.evolve([PauliX([0])])
        sv = sim.state_vector
        prob_0 = sim._marginal_prob_0(sv, 0)
        assert np.isclose(prob_0, 0.0, atol=1e-7)
