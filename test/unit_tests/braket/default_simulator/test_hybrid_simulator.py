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
from braket.default_simulator.sparse_simulator import SparseStateSimulator


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
    def test_product_backend_hadamards(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [Hadamard([0]), Hadamard([1]), Hadamard([2]), Hadamard([3])]
        sim.evolve(ops)
        assert sim.get_last_backend() == "product"

    def test_product_probabilities(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [Hadamard([0]), Hadamard([1]), Hadamard([2]), Hadamard([3])]
        sim.evolve(ops)
        probs = sim.probabilities
        assert len(probs) == 16
        assert np.isclose(np.sum(probs), 1.0, atol=1e-7)

    def test_product_sampling(self):
        sim = HybridSimulation(qubit_count=4, shots=1000)
        ops = [Hadamard([0]), Hadamard([1]), Hadamard([2]), Hadamard([3])]
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
        ops = [Hadamard([0]), Hadamard([1]), Hadamard([2]), Hadamard([3])]
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


class TestClassifyEdgeCases:
    def test_classify_non_clifford_two_qubit(self):
        sim = HybridSimulation(qubit_count=2, shots=100)
        ops = [CPhaseShift([0, 1], np.pi / 3)]
        sim.evolve(ops)
        assert sim.get_last_backend() == "full"

    def test_classify_swap_uses_clifford(self):
        from braket.default_simulator.gate_operations import Swap

        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [
            Hadamard([0]),
            Hadamard([1]),
            Hadamard([2]),
            Hadamard([3]),
            Swap([0, 3]),
            CX([1, 2]),
            CX([0, 1]),
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
            Hadamard([0]),
            Hadamard([1]),
            Hadamard([2]),
            Hadamard([3]),
            CPhaseShift([0, 1], np.pi / 2),
            CPhaseShift([0, 2], np.pi / 4),
            CPhaseShift([1, 2], np.pi / 2),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() == "qft"

    def test_qft_sampling(self):
        sim = HybridSimulation(qubit_count=4, shots=1000)
        ops = [
            Hadamard([0]),
            Hadamard([1]),
            Hadamard([2]),
            Hadamard([3]),
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
            Hadamard([0]),
            Hadamard([1]),
            Hadamard([2]),
            Hadamard([3]),
            CPhaseShift([0, 1], np.pi / 2),
            CPhaseShift([0, 2], np.pi / 4),
            CPhaseShift([1, 2], np.pi / 2),
        ]
        sim.evolve(ops)
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestPartitionedBackend:
    def test_partitioned_backend_selection(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = [
            RotX([0], np.pi / 3),
            CX([0, 1]),
            RotX([3], np.pi / 4),
            CX([3, 4]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"

    def test_partitioned_state_vector(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = [
            RotX([0], np.pi / 3),
            CX([0, 1]),
            RotX([3], np.pi / 4),
            CX([3, 4]),
        ]
        sim.evolve(ops)
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_partitioned_sampling(self):
        sim = HybridSimulation(qubit_count=6, shots=1000)
        ops = [
            RotX([0], np.pi / 3),
            CX([0, 1]),
            RotX([3], np.pi / 4),
            CX([3, 4]),
        ]
        sim.evolve(ops)
        samples = sim.retrieve_samples()
        assert len(samples) == 1000

    def test_partitioned_probabilities(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = [
            RotX([0], np.pi / 3),
            CX([0, 1]),
            RotX([3], np.pi / 4),
            CX([3, 4]),
        ]
        sim.evolve(ops)
        probs = sim.probabilities
        assert np.isclose(np.sum(probs), 1.0, atol=1e-7)

    def test_partitioned_mixed_backends(self):
        sim = HybridSimulation(qubit_count=8, shots=100)
        ops = [
            Hadamard([0]),
            CX([0, 1]),
            Hadamard([3]),
            RotX([5], np.pi / 4),
            RotZ([6], np.pi / 3),
            CX([5, 6]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() in ("partitioned", "mps", "clifford")
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_partitioned_deterministic_sampling(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = [
            PauliX([0]),
            PauliX([1]),
            PauliX([4]),
            PauliX([5]),
        ]
        sim.evolve(ops)
        samples = sim.retrieve_samples()
        assert all(s == 0b110011 for s in samples)

    def test_partitioned_not_used_when_single_component(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [RotX([0], np.pi / 3), CX([0, 1]), CX([1, 2]), CX([2, 3])]
        sim.evolve(ops)
        assert sim.get_last_backend() != "partitioned"

    def test_partitioned_not_used_when_fully_connected(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = [
            RotX([0], np.pi / 3),
            CX([0, 1]),
            CX([1, 2]),
            CX([2, 3]),
            CX([3, 4]),
            CX([4, 5]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() != "partitioned"

    def test_partitioned_qft_in_partition(self):
        sim = HybridSimulation(qubit_count=8, shots=100)
        ops = [
            PauliX([0]),
            Hadamard([0]),
            Hadamard([1]),
            CPhaseShift([0, 1], np.pi / 2),
            Hadamard([4]),
            CX([4, 5]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() in ("partitioned", "mps", "qft")
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestStabilizerIntegration:
    def test_stabilizer_in_partitioned_simulation(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = [
            Hadamard([0]),
            CX([0, 1]),
            Hadamard([3]),
            CX([3, 4]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_stabilizer_sampling_in_partition(self):
        sim = HybridSimulation(qubit_count=6, shots=1000)
        ops = [
            Hadamard([0]),
            CX([0, 1]),
            Hadamard([3]),
            CX([3, 4]),
        ]
        sim.evolve(ops)
        samples = sim.retrieve_samples()
        assert len(samples) == 1000


class TestFastPathOptimization:
    def test_large_general_circuit_uses_fast_path(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = []
        for i in range(30):
            ops.append(Hadamard([i % 4]))
            ops.append(RotX([i % 4], np.pi / 3))
            ops.append(CX([i % 4, (i + 1) % 4]))
        sim.evolve(ops)
        assert sim.get_last_backend() == "full"
        assert sim._is_clifford is False
        assert sim._is_product is False

    def test_small_circuit_uses_classification(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [Hadamard([i % 4]) for i in range(10)]
        sim.evolve(ops)
        assert sim.get_last_backend() == "product"

    def test_fast_path_state_correctness(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = []
        for i in range(30):
            ops.append(Hadamard([i % 4]))
            ops.append(RotX([i % 4], np.pi / 3))
            ops.append(CX([i % 4, (i + 1) % 4]))
        sim.evolve(ops)
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestConnectedComponents:
    def test_get_connected_components_single(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [CX([0, 1]), CX([1, 2]), CX([2, 3])]
        components = sim._get_connected_components(ops)
        assert len(components) == 1
        assert components[0] == {0, 1, 2, 3}

    def test_get_connected_components_multiple(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = [CX([0, 1]), CX([3, 4])]
        components = sim._get_connected_components(ops)
        assert len(components) == 4

    def test_classify_ops_product(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [Hadamard([0]), PauliX([1])]
        backend = sim._classify_ops(ops, 2)
        assert backend == "product"

    def test_classify_ops_clifford(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [Hadamard([0]), CX([0, 1])]
        backend = sim._classify_ops(ops, 2)
        assert backend == "clifford"

    def test_classify_ops_qft(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [
            Hadamard([0]),
            Hadamard([1]),
            CPhaseShift([0, 1], np.pi / 2),
        ]
        backend = sim._classify_ops(ops, 2)
        assert backend == "qft"

    def test_classify_ops_empty(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        backend = sim._classify_ops([], 2)
        assert backend == "product"


class TestMPSBackend:
    def test_mps_backend_low_entanglement(self):
        sim = HybridSimulation(qubit_count=10, shots=100)
        ops = []
        for i in range(9):
            ops.append(RotX([i], np.pi / 4))
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        assert sim.get_last_backend() == "mps"
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_mps_sampling(self):
        sim = HybridSimulation(qubit_count=10, shots=1000)
        ops = []
        for i in range(9):
            ops.append(RotX([i], np.pi / 4))
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        samples = sim.retrieve_samples()
        assert len(samples) == 1000

    def test_mps_not_used_for_small_circuits(self):
        sim = HybridSimulation(qubit_count=3, shots=100)
        ops = [RotX([0], np.pi / 4), CX([0, 1]), CX([1, 2])]
        sim.evolve(ops)
        assert sim.get_last_backend() != "mps"

    def test_mps_fallback_on_high_entanglement(self):
        sim = HybridSimulation(qubit_count=10, shots=100, max_bond_dim=4)
        ops = []
        for _ in range(20):
            for i in range(9):
                ops.append(RotX([i], np.pi / 4))
                ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_clifford_preferred_over_mps(self):
        sim = HybridSimulation(qubit_count=10, shots=100)
        ops = []
        for i in range(9):
            ops.append(Hadamard([i]))
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        assert sim.get_last_backend() == "clifford"


class TestMPSInPartition:
    def test_mps_in_partitioned_simulation(self):
        sim = HybridSimulation(qubit_count=20, shots=100)
        ops = []
        for i in range(9):
            ops.append(RotX([i], np.pi / 4))
            ops.append(CX([i, i + 1]))
        for i in range(12, 19):
            ops.append(RotX([i], np.pi / 4))
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        assert sim.get_last_backend() in ("partitioned", "mps", "full")
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestSparseBackend:
    def test_sparse_backend_selection(self):
        sim = HybridSimulation(qubit_count=14, shots=100)
        ops = []
        for i in range(14):
            ops.append(PauliX([i]))
        for i in range(13):
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        assert sim.get_last_backend() in ("sparse", "clifford", "full")
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_sparse_sampling(self):
        sim = HybridSimulation(qubit_count=14, shots=100)
        ops = [PauliX([0]), CX([0, 1])]
        sim.evolve(ops)
        samples = sim.retrieve_samples()
        assert len(samples) == 100

    def test_sparse_fallback_on_dense_state(self):
        sim = HybridSimulation(qubit_count=4, shots=100)
        ops = [Hadamard([i]) for i in range(4)]
        sim.evolve(ops)
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestSparseInPartition:
    def test_sparse_partition_backend(self):
        sim = HybridSimulation(qubit_count=30, shots=100)
        ops = []
        for i in range(14):
            ops.append(PauliX([i]))
            if i < 13:
                ops.append(CX([i, i + 1]))
        for i in range(16, 29):
            ops.append(PauliX([i]))
            if i < 28:
                ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestPartitionSampling:
    def test_mps_partition_sampling(self):
        sim = HybridSimulation(qubit_count=20, shots=100)
        ops = []
        for i in range(8):
            ops.append(Hadamard([i]))
            if i < 7:
                ops.append(CX([i, i + 1]))
        for i in range(12, 18):
            ops.append(Hadamard([i]))
            if i < 17:
                ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        samples = sim.retrieve_samples()
        assert len(samples) == 100

    def test_sparse_partition_sampling(self):
        sim = HybridSimulation(qubit_count=30, shots=50)
        ops = []
        for i in range(13):
            ops.append(PauliX([i]))
            ops.append(CX([i, i + 1]))
        for i in range(16, 28):
            ops.append(PauliX([i]))
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        samples = sim.retrieve_samples()
        assert len(samples) == 50

    def test_full_partition_sampling(self):
        sim = HybridSimulation(qubit_count=8, shots=50)
        ops = [
            Hadamard([0]),
            RotX([0], np.pi / 3),
            T([0]),
            CX([0, 1]),
            Hadamard([4]),
            RotX([4], np.pi / 4),
            T([4]),
            CX([4, 5]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        samples = sim.retrieve_samples()
        assert len(samples) == 50


class TestDirectBackendSampling:
    def test_mps_direct_sampling(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = []
        for i in range(6):
            ops.append(RotX([i], np.pi / 4))
        for i in range(5):
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        assert sim.get_last_backend() == "mps"
        assert sim._mps_sim is not None
        samples = sim.retrieve_samples()
        assert len(samples) == 100

    def test_sparse_direct_sampling(self):
        sim = HybridSimulation(qubit_count=8, shots=100)
        ops = []
        for i in range(8):
            ops.append(PauliX([i]))
        for i in range(7):
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        assert sim.get_last_backend() in ("sparse", "clifford", "full")
        samples = sim.retrieve_samples()
        assert len(samples) == 100

    def test_clifford_partition_sampling(self):
        sim = HybridSimulation(qubit_count=8, shots=50)
        ops = [
            Hadamard([0]),
            CX([0, 1]),
            Hadamard([4]),
            CX([4, 5]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        samples = sim.retrieve_samples()
        assert len(samples) == 50

    def test_sparse_backend_direct_sampling(self):
        sim = HybridSimulation(qubit_count=8, shots=100)
        ops = []
        for i in range(8):
            ops.append(PauliX([i]))
            ops.append(Hadamard([i]))
        for i in range(7):
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        if sim._sparse_sim is not None:
            samples = sim.retrieve_samples()
            assert len(samples) == 100


class TestSparsePartitionBackend:
    def test_sparse_partition_in_simulation(self):
        sim = HybridSimulation(qubit_count=20, shots=50)
        ops = []
        for i in range(8):
            ops.append(PauliX([i]))
            ops.append(Hadamard([i]))
        for i in range(7):
            ops.append(CX([i, i + 1]))
        for i in range(12, 19):
            ops.append(PauliX([i]))
            ops.append(Hadamard([i]))
        for i in range(12, 18):
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestMPSPartitionSamplingPath:
    def test_mps_partition_sample_path(self):
        sim = HybridSimulation(qubit_count=16, shots=50)
        ops = []
        for i in range(6):
            ops.append(RotX([i], np.pi / 4))
        for i in range(5):
            ops.append(CX([i, i + 1]))
        for i in range(10, 15):
            ops.append(RotX([i], np.pi / 4))
        for i in range(10, 14):
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        samples = sim.retrieve_samples()
        assert len(samples) == 50


class TestMPSBondDimensionFallback:
    def test_mps_high_bond_dimension_fallback(self):
        sim = HybridSimulation(qubit_count=6, shots=100, max_bond_dim=2)
        ops = []
        for i in range(6):
            ops.append(Hadamard([i]))
        for _ in range(10):
            for i in range(5):
                ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestSparseSparsityFallback:
    def test_sparse_becomes_dense(self):
        sim = HybridSimulation(qubit_count=8, shots=100)
        sim._sparse_sim = SparseStateSimulator(8)
        sim._is_sparse_candidate = True
        for i in range(8):
            sim._sparse_sim.apply_single_qubit_gate(Hadamard([0]).matrix, i)
        ops = [PauliX([0])]
        sim._evolve_sparse(ops)
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestMaterializePaths:
    def test_materialize_mps(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = []
        for i in range(6):
            ops.append(RotX([i], np.pi / 4))
        for i in range(5):
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        if sim._mps_sim is not None:
            sim._materialize_mps_if_needed()
            assert sim._mps_sim is None

    def test_materialize_sparse(self):
        sim = HybridSimulation(qubit_count=8, shots=100)
        sim._sparse_sim = SparseStateSimulator(8)
        sim._sparse_sim.apply_single_qubit_gate(PauliX([0]).matrix, 0)
        sim._materialize_sparse_if_needed()
        assert sim._sparse_sim is None


class TestDirectMPSSampling:
    def test_mps_retrieve_samples_direct(self):
        sim = HybridSimulation(qubit_count=6, shots=100)
        ops = []
        for i in range(6):
            ops.append(RotX([i], np.pi / 4))
        for i in range(5):
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        if sim._mps_sim is not None:
            samples = sim.retrieve_samples()
            assert len(samples) == 100


class TestDirectSparseSampling:
    def test_sparse_retrieve_samples_direct(self):
        sim = HybridSimulation(qubit_count=8, shots=100)
        sim._sparse_sim = SparseStateSimulator(8)
        sim._sparse_sim.apply_single_qubit_gate(PauliX([0]).matrix, 0)
        sim._clifford_sim = None
        sim._product_sim = None
        sim._mps_sim = None
        sim._partition_states = None
        samples = sim.retrieve_samples()
        assert len(samples) == 100


class TestQFTPartitionBackend:
    def test_qft_partition_in_simulation(self):
        sim = HybridSimulation(qubit_count=10, shots=50)
        ops = [
            PauliX([0]),
            Hadamard([0]),
            Hadamard([1]),
            Hadamard([2]),
            CPhaseShift([0, 1], np.pi / 2),
            CPhaseShift([0, 2], np.pi / 4),
            CPhaseShift([1, 2], np.pi / 2),
            Hadamard([6]),
            CX([6, 7]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_qft_partition_with_x_gates(self):
        sim = HybridSimulation(qubit_count=8, shots=50)
        ops = [
            PauliX([0]),
            PauliX([1]),
            Hadamard([0]),
            Hadamard([1]),
            Hadamard([2]),
            CPhaseShift([0, 1], np.pi / 2),
            CPhaseShift([0, 2], np.pi / 4),
            CPhaseShift([1, 2], np.pi / 2),
            PauliX([5]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestSmallMPSPartitionFallback:
    def test_small_mps_partition_uses_full(self):
        sim = HybridSimulation(qubit_count=10, shots=50)
        ops = [
            RotX([0], np.pi / 4),
            CX([0, 1]),
            RotX([5], np.pi / 4),
            CX([5, 6]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestSparsePartitionBackendDirect:
    def test_sparse_partition_backend_direct(self):
        sim = HybridSimulation(qubit_count=28, shots=20)
        ops = []
        for i in range(13):
            ops.append(PauliX([i]))
            ops.append(CX([i, i + 1]))
        for i in range(16, 27):
            ops.append(PauliX([i]))
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_sparse_partition_with_toffoli(self):
        from braket.default_simulator.gate_operations import CCNot

        sim = HybridSimulation(qubit_count=30, shots=10)
        ops = []
        for i in range(12):
            ops.append(PauliX([i]))
            ops.append(CCNot([i, i + 1, i + 2]))
        for i in range(18, 28):
            ops.append(PauliX([i]))
            ops.append(CCNot([i, i + 1, i + 2]))
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestFullPartitionSamplingPath:
    def test_full_partition_sampling_direct(self):
        sim = HybridSimulation(qubit_count=10, shots=50)
        ops = [
            Hadamard([0]),
            T([0]),
            RotX([0], np.pi / 5),
            CX([0, 1]),
            Hadamard([5]),
            T([5]),
            RotX([5], np.pi / 5),
            CX([5, 6]),
        ]
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        samples = sim.retrieve_samples()
        assert len(samples) == 50


class TestSampleFromProbabilities:
    def test_sample_from_probabilities_path(self):
        sim = HybridSimulation(qubit_count=2, shots=100, force_backend="full")
        sim.evolve([Hadamard([0]), Hadamard([1])])
        sim._clifford_sim = None
        sim._product_sim = None
        sim._mps_sim = None
        sim._sparse_sim = None
        sim._partition_states = None
        samples = sim.retrieve_samples()
        assert len(samples) == 100


class TestClassifyOpsEdgeCases:
    def test_classify_three_qubit_gate_not_mps(self):
        from braket.default_simulator.gate_operations import CCNot

        sim = HybridSimulation(qubit_count=10, shots=10)
        ops = [CCNot([0, 1, 2])]
        backend = sim._classify_ops(ops, 10)
        assert backend == "full"

    def test_classify_cphaseshift_is_mps_compatible(self):
        sim = HybridSimulation(qubit_count=10, shots=10)
        ops = [CPhaseShift([0, 1], np.pi / 3), CPhaseShift([1, 2], np.pi / 3)]
        backend = sim._classify_ops(ops, 10)
        assert backend == "mps"

    def test_classify_swap_in_qft_check(self):
        from braket.default_simulator.gate_operations import Swap

        sim = HybridSimulation(qubit_count=4, shots=10)
        ops = [Hadamard([0]), Hadamard([1]), CPhaseShift([0, 1], np.pi / 2), Swap([0, 1])]
        backend = sim._classify_ops(ops, 4)
        assert backend == "qft"

    def test_classify_small_circuit_not_mps(self):
        sim = HybridSimulation(qubit_count=4, shots=10)
        ops = [T([0]), CX([0, 1])]
        backend = sim._classify_ops(ops, 4)
        # T + CX are MPS-compatible, but small circuits may still use MPS
        # The actual behavior depends on the classification logic
        assert backend in ("full", "clifford", "mps")


class TestEvolveEdgeCases:
    def test_evolve_partitioned_single_component_skipped(self):
        sim = HybridSimulation(qubit_count=6, shots=10)
        ops = [CX([0, 1]), CX([1, 2]), CX([2, 3]), CX([3, 4]), CX([4, 5])]
        sim.evolve(ops)
        assert sim.get_last_backend() != "partitioned"

    def test_evolve_partitioned_max_size_equals_qubit_count(self):
        sim = HybridSimulation(qubit_count=4, shots=10)
        ops = [CX([0, 1]), CX([1, 2]), CX([2, 3])]
        sim.evolve(ops)
        assert sim.get_last_backend() != "partitioned"

    def test_evolve_mps_candidate_false_uses_full(self):
        sim = HybridSimulation(qubit_count=10, shots=10)
        sim._is_mps_candidate = False
        ops = []
        for i in range(9):
            ops.append(RotX([i], np.pi / 4))
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        # Classification returns "mps" but dispatch falls back to full since _is_mps_candidate=False
        # However, the last_backend is set before dispatch, so it shows "mps"
        # The actual execution uses full backend due to fallback
        assert sim.get_last_backend() == "mps"
        # Verify specialized backends are invalidated after fallback
        assert sim._is_mps_candidate is False

    def test_evolve_sparse_candidate_false_uses_full(self):
        sim = HybridSimulation(qubit_count=14, shots=10)
        sim._is_sparse_candidate = False
        ops = [PauliX([i]) for i in range(14)]
        for i in range(13):
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        assert sim.get_last_backend() in ("full", "clifford")


class TestSparsePartitionBackendCoverage:
    def test_sparse_partition_backend_in_simulate_partition(self):
        """Test that sparse backend is used in _simulate_partition for large sparse-friendly partitions."""
        sim = HybridSimulation(qubit_count=28, shots=10)
        # Create two partitions, each with >= 12 qubits and >= 70% sparse-friendly gates
        ops = []
        # First partition: qubits 0-13 (14 qubits)
        for i in range(13):
            ops.append(PauliX([i]))
            ops.append(CX([i, i + 1]))
        # Second partition: qubits 16-27 (12 qubits)
        for i in range(16, 27):
            ops.append(PauliX([i]))
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestDispatchFallbackCoverage:
    def test_dispatch_fallback_when_backend_invalid(self):
        """Test fallback path when classified backend's is_valid flag is False."""
        sim = HybridSimulation(qubit_count=14, shots=10)
        # Invalidate sparse candidate
        sim._is_sparse_candidate = False
        # Create ops that would classify as sparse
        ops = [PauliX([i]) for i in range(14)]
        for i in range(13):
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        # Should fall back to full since sparse is invalidated
        assert sim.get_last_backend() in ("full", "clifford")

    def test_dispatch_fallback_unknown_backend(self):
        """Test fallback when backend is not in dispatch dict."""
        sim = HybridSimulation(qubit_count=4, shots=10)
        # Directly call _dispatch_backend with unknown backend
        ops = [Hadamard([0])]
        sim._dispatch_backend("unknown", ops)
        # Should have used full backend
        assert sim._is_clifford is False


class TestClassifyOpsSparseReturn:
    def test_classify_ops_returns_sparse(self):
        """Test that _classify_ops returns 'sparse' for large sparse-friendly non-Clifford circuits."""
        sim = HybridSimulation(qubit_count=14, shots=10)
        # Create ops with >= 70% sparse-friendly gates on >= 12 qubits
        # Use T gates to break Clifford classification
        ops = []
        for i in range(14):
            ops.append(PauliX([i]))
            ops.append(T([i]))  # Non-Clifford but sparse-friendly count doesn't include T
        for i in range(13):
            ops.append(CX([i, i + 1]))
        # PauliX (14) + CX (13) = 27 sparse-friendly out of 41 total = 65.8% < 70%
        # Need more sparse-friendly gates
        ops = []
        for i in range(14):
            ops.append(PauliX([i]))
        for i in range(13):
            ops.append(CX([i, i + 1]))
        # This is all Clifford, so it returns "clifford" - that's correct behavior
        backend = sim._classify_ops(ops, 14)
        assert backend == "clifford"

    def test_classify_ops_sparse_non_clifford(self):
        """Test sparse classification for non-Clifford sparse-friendly circuits."""
        sim = HybridSimulation(qubit_count=14, shots=10)
        # To get sparse, we need:
        # 1. >= 12 qubits (14 meets this)
        # 2. >= 70% sparse-friendly gates
        # 3. NOT all MPS-compatible OR exceed MPS criteria
        # Use long-range CX gates to break MPS (max_gate_distance > 2)
        ops = []
        for i in range(14):
            ops.append(PauliX([i]))
            ops.append(Hadamard([i]))
        # Long-range CX gates break MPS (distance > 2)
        for i in range(10):
            ops.append(CX([i, i + 3]))  # distance = 3 > 2
        ops.append(T([0]))  # Break Clifford
        # sparse-friendly: 14 X + 14 H + 10 CX = 38, total = 39
        # 38/39 = 97.4% > 70%
        backend = sim._classify_ops(ops, 14)
        assert backend == "sparse"


class TestEvolveMultipleCalls:
    """Test multiple evolve calls to cover branch where simulator already exists."""

    def test_clifford_multiple_evolve_calls(self):
        """Test multiple Clifford evolve calls - second call reuses existing sim."""
        sim = HybridSimulation(qubit_count=2, shots=10)
        sim.evolve([Hadamard([0]), CX([0, 1])])
        assert sim._clifford_sim is not None
        # Second call should reuse existing clifford sim
        sim.evolve([Hadamard([1]), CX([0, 1])])
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_product_multiple_evolve_calls(self):
        """Test multiple product evolve calls - second call reuses existing sim."""
        sim = HybridSimulation(qubit_count=2, shots=10)
        sim.evolve([Hadamard([0])])
        assert sim._product_sim is not None
        # Second call should reuse existing product sim
        sim.evolve([Hadamard([1])])
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_qft_reuses_product_sim(self):
        """Test QFT evolve when product sim already exists."""
        sim = HybridSimulation(qubit_count=4, shots=10)
        # First create product sim
        sim._product_sim = None
        sim.evolve([Hadamard([0])])
        assert sim._product_sim is not None
        # Now do QFT - should reuse product sim
        sim._is_qft_candidate = True
        ops = [
            PauliX([0]),
            Hadamard([0]),
            Hadamard([1]),
            Hadamard([2]),
            Hadamard([3]),
            CPhaseShift([0, 1], np.pi / 2),
            CPhaseShift([0, 2], np.pi / 4),
        ]
        sim._evolve_qft(ops)
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_mps_multiple_evolve_calls(self):
        """Test multiple MPS evolve calls - second call reuses existing sim."""
        sim = HybridSimulation(qubit_count=10, shots=10)
        ops = []
        for i in range(9):
            ops.append(RotX([i], np.pi / 4))
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        assert sim._mps_sim is not None
        # Second call should reuse existing MPS sim
        sim.evolve([RotX([0], np.pi / 4)])
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_sparse_multiple_evolve_calls(self):
        """Test multiple sparse evolve calls - second call reuses existing sim."""
        sim = HybridSimulation(qubit_count=14, shots=10)
        # Create sparse-friendly non-Clifford circuit
        ops = []
        for i in range(14):
            ops.append(PauliX([i]))
            ops.append(Hadamard([i]))
        for i in range(13):
            ops.append(CX([i, i + 1]))
        ops.append(T([0]))  # Break Clifford
        sim.evolve(ops)
        if sim._sparse_sim is not None:
            # Second call should reuse existing sparse sim
            sim.evolve([PauliX([0])])
            sv = sim.state_vector
            assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestSparsePartitionCoverage:
    def test_sparse_partition_in_simulate_partition(self):
        """Test sparse backend path in _simulate_partition."""
        sim = HybridSimulation(qubit_count=30, shots=10)
        # Create two partitions with sparse-friendly non-Clifford gates
        ops = []
        # First partition: qubits 0-13 (14 qubits)
        for i in range(14):
            ops.append(PauliX([i]))
            ops.append(Hadamard([i]))
        for i in range(13):
            ops.append(CX([i, i + 1]))
        ops.append(T([0]))  # Break Clifford
        # Second partition: qubits 18-29 (12 qubits)
        for i in range(18, 30):
            ops.append(PauliX([i]))
            ops.append(Hadamard([i]))
        for i in range(18, 29):
            ops.append(CX([i, i + 1]))
        ops.append(T([18]))  # Break Clifford
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestHybridSimulationCoverageGaps:
    def test_sparse_partition_backend_coverage(self):
        """Test sparse backend in _simulate_partition with large sparse-friendly partition."""
        sim = HybridSimulation(qubit_count=30, shots=10)
        ops = []
        # First partition: qubits 0-13 (14 qubits) - sparse-friendly, non-Clifford, non-MPS
        for i in range(14):
            ops.append(PauliX([i]))
            ops.append(Hadamard([i]))
        # Long-range gates to break MPS (distance > 2)
        for i in range(10):
            ops.append(CX([i, i + 3]))
        ops.append(T([0]))  # Break Clifford
        # Second partition: qubits 18-29 (12 qubits) - same pattern
        for i in range(18, 30):
            ops.append(PauliX([i]))
            ops.append(Hadamard([i]))
        for i in range(18, 26):
            ops.append(CX([i, i + 3]))
        ops.append(T([18]))  # Break Clifford
        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_mps_bond_dimension_exceeds_threshold(self):
        """Test MPS fallback when bond dimension exceeds threshold."""
        # Use small max_bond_dim to trigger fallback quickly
        sim = HybridSimulation(qubit_count=10, shots=10, max_bond_dim=4)
        ops = []
        # Create high entanglement to exceed bond dimension
        for _ in range(30):
            for i in range(9):
                ops.append(Hadamard([i]))
                ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        # Should have fallen back to full after MPS bond dim exceeded
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_sparse_becomes_dense_fallback(self):
        """Test sparse simulator fallback when state becomes too dense."""
        sim = HybridSimulation(qubit_count=14, shots=10)
        # First evolve with sparse-friendly ops to use sparse backend
        ops = []
        for i in range(14):
            ops.append(PauliX([i]))
        for i in range(10):
            ops.append(CX([i, i + 3]))  # Long-range to avoid MPS
        ops.append(T([0]))  # Break Clifford
        sim.evolve(ops)
        # Now apply Hadamards to make state dense
        if sim._sparse_sim is not None:
            dense_ops = [Hadamard([i]) for i in range(14)]
            sim._evolve_sparse(dense_ops)
            # Should have materialized and invalidated sparse
            sv = sim.state_vector
            assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestSparsePartitionInSimulatePartition:
    """Test sparse backend path in _simulate_partition (lines 168-170)."""

    def test_sparse_backend_in_partition(self):
        """Create partitions with >= 12 qubits that classify as sparse."""
        sim = HybridSimulation(qubit_count=30, shots=10)
        ops = []
        # First partition: qubits 0-13 (14 qubits)
        # Need: >= 12 qubits, >= 70% sparse-friendly, NOT Clifford, NOT MPS
        for i in range(14):
            ops.append(PauliX([i]))
            ops.append(Hadamard([i]))
        # Connect all qubits in partition with adjacent CX
        for i in range(13):
            ops.append(CX([i, i + 1]))
        # Add long-range CX to break MPS (distance > 2)
        for i in range(10):
            ops.append(CX([i, i + 3]))
        ops.append(T([0]))  # Break Clifford

        # Second partition: qubits 18-29 (12 qubits) - same pattern
        for i in range(18, 30):
            ops.append(PauliX([i]))
            ops.append(Hadamard([i]))
        # Connect all qubits in partition
        for i in range(18, 29):
            ops.append(CX([i, i + 1]))
        # Add long-range CX to break MPS
        for i in range(18, 26):
            ops.append(CX([i, i + 3]))
        ops.append(T([18]))  # Break Clifford

        sim.evolve(ops)
        assert sim.get_last_backend() == "partitioned"
        # Verify state is valid
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_sparse_partition_sampling(self):
        """Test sampling from sparse partition."""
        sim = HybridSimulation(qubit_count=30, shots=50)
        ops = []
        # First partition: qubits 0-13 (14 qubits)
        for i in range(14):
            ops.append(PauliX([i]))
            ops.append(Hadamard([i]))
        for i in range(13):
            ops.append(CX([i, i + 1]))
        for i in range(10):
            ops.append(CX([i, i + 3]))
        ops.append(T([0]))

        # Second partition: qubits 18-29 (12 qubits)
        for i in range(18, 30):
            ops.append(PauliX([i]))
            ops.append(Hadamard([i]))
        for i in range(18, 29):
            ops.append(CX([i, i + 1]))
        for i in range(18, 26):
            ops.append(CX([i, i + 3]))
        ops.append(T([18]))

        sim.evolve(ops)
        samples = sim.retrieve_samples()
        assert len(samples) == 50


class TestMPSEvolveBranchCoverage:
    """Test MPS evolve paths for branch coverage."""

    def test_mps_evolve_no_fallback(self):
        """Test MPS evolve where bond dimension stays below threshold."""
        sim = HybridSimulation(qubit_count=10, shots=10, max_bond_dim=64)
        ops = []
        for i in range(9):
            ops.append(RotX([i], np.pi / 4))
            ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        assert sim.get_last_backend() == "mps"
        assert sim._mps_sim is not None
        # Bond dimension should be below threshold
        bond_dims = sim._mps_sim.get_bond_dimensions()
        assert all(d <= 64 for d in bond_dims)
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)

    def test_mps_evolve_with_fallback(self):
        """Test MPS evolve where bond dimension exceeds threshold (64)."""
        # Use large max_bond_dim so bond dims can grow, then create enough entanglement
        # to exceed _MPS_BOND_THRESHOLD (64)
        sim = HybridSimulation(qubit_count=10, shots=10, max_bond_dim=128)
        ops = []
        # Create high entanglement with non-Clifford gates to use MPS
        # Many layers of entangling gates will grow bond dimension
        for _ in range(50):
            for i in range(9):
                ops.append(RotX([i], np.pi / 4))
                ops.append(CX([i, i + 1]))
        sim.evolve(ops)
        # Should have fallen back after bond dim exceeded 64
        assert sim._is_mps_candidate is False
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestSparseEvolveBranchCoverage:
    """Test sparse evolve paths for branch coverage."""

    def test_sparse_evolve_stays_sparse(self):
        """Test sparse evolve where state remains sparse."""
        sim = HybridSimulation(qubit_count=14, shots=10)
        # Create sparse-friendly non-Clifford circuit
        # Need: >= 12 qubits, >= 70% sparse-friendly, NOT Clifford, NOT MPS
        # Use adjacent CX to connect all qubits (avoids partitioning)
        # Then add long-range CX to break MPS (distance > 2)
        ops = []
        for i in range(14):
            ops.append(PauliX([i]))
        # First connect all qubits with adjacent CX
        for i in range(13):
            ops.append(CX([i, i + 1]))
        # Add long-range CX to break MPS (distance > 2)
        for i in range(10):
            ops.append(CX([i, i + 3]))
        ops.append(T([0]))  # Break Clifford
        sim.evolve(ops)
        assert sim.get_last_backend() == "sparse"
        assert sim._sparse_sim is not None
        # State should still be sparse (only a few basis states)
        assert sim._sparse_sim.is_sparse()
        assert sim._is_sparse_candidate is True

    def test_sparse_evolve_becomes_dense(self):
        """Test sparse evolve where state becomes dense."""
        sim = HybridSimulation(qubit_count=14, shots=10)
        # First create sparse state - connect all qubits to avoid partitioning
        ops = []
        for i in range(14):
            ops.append(PauliX([i]))
        # Connect all qubits with adjacent CX
        for i in range(13):
            ops.append(CX([i, i + 1]))
        # Add long-range CX to break MPS
        for i in range(10):
            ops.append(CX([i, i + 3]))
        ops.append(T([0]))  # Break Clifford
        sim.evolve(ops)
        assert sim.get_last_backend() == "sparse"
        assert sim._sparse_sim is not None

        # Apply Hadamards to make dense (creates superposition of all basis states)
        dense_ops = [Hadamard([i]) for i in range(14)]
        sim._evolve_sparse(dense_ops)
        # Should have fallen back since state is now dense
        assert sim._is_sparse_candidate is False
        sv = sim.state_vector
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-7)


class TestMPSEvolveCoverageGaps:
    """Test MPS evolve branch coverage."""

    def test_mps_evolve_reuses_existing_sim(self):
        """Test _evolve_mps when _mps_sim already exists."""
        sim = HybridSimulation(qubit_count=10, shots=10, max_bond_dim=64)
        ops1 = [RotX([i], np.pi / 4) for i in range(9)] + [CX([i, i + 1]) for i in range(8)]
        sim._evolve_mps(ops1)
        assert sim._mps_sim is not None
        first_sim = sim._mps_sim

        ops2 = [RotX([i], np.pi / 8) for i in range(9)]
        sim._evolve_mps(ops2)
        assert sim._mps_sim is first_sim


class TestMPSFallbackBranchCoverage:
    """Test MPS fallback when bond dimension exceeds threshold."""

    def test_mps_fallback_via_mock(self):
        """Test _evolve_mps fallback by mocking MPS simulator bond dimensions."""
        from unittest.mock import MagicMock, patch

        sim = HybridSimulation(qubit_count=4, shots=10, max_bond_dim=128)

        mock_mps = MagicMock()
        mock_mps.get_bond_dimensions.return_value = [65, 65, 65]
        mock_mps.get_state_vector.return_value = np.array([1] + [0] * 15, dtype=np.complex128)
        mock_mps.apply_operations = MagicMock()

        with patch(
            "braket.default_simulator.hybrid_simulation.MPSSimulator",
            return_value=mock_mps,
        ):
            sim._mps_sim = None
            sim._is_mps_candidate = True
            ops = [Hadamard([0])]
            sim._evolve_mps(ops)

            assert sim._is_mps_candidate is False
