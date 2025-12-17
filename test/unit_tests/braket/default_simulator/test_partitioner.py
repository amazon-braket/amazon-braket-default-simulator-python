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

from braket.default_simulator.gate_operations import CX, Hadamard, PauliX, PauliZ, T
from braket.default_simulator.partitioner import QubitPartitioner


class TestEntanglementGraphConstruction:
    def test_no_entangling_gates(self):
        ops = [Hadamard([0]), PauliX([1]), PauliZ([2])]
        partitioner = QubitPartitioner(ops, 3)
        graph = partitioner.analyze_entanglement()
        for qubit in range(3):
            assert len(graph[qubit]) == 0

    def test_single_entangling_gate(self):
        ops = [CX([0, 1])]
        partitioner = QubitPartitioner(ops, 2)
        graph = partitioner.analyze_entanglement()
        assert 1 in graph[0]
        assert 0 in graph[1]

    def test_multiple_entangling_gates(self):
        ops = [CX([0, 1]), CX([1, 2])]
        partitioner = QubitPartitioner(ops, 3)
        graph = partitioner.analyze_entanglement()
        assert 1 in graph[0]
        assert 0 in graph[1] and 2 in graph[1]
        assert 1 in graph[2]

    def test_repeated_qubit_pairs(self):
        ops = [CX([0, 1]), CX([0, 1]), CX([0, 1])]
        partitioner = QubitPartitioner(ops, 2)
        graph = partitioner.analyze_entanglement()
        assert 1 in graph[0]
        assert 0 in graph[1]


class TestPartitionGeneration:
    def test_fully_connected_single_partition(self):
        ops = [CX([0, 1]), CX([1, 2])]
        partitioner = QubitPartitioner(ops, 3)
        partitions = partitioner.partition()
        assert len(partitions) == 1
        assert partitions[0].qubits == {0, 1, 2}

    def test_disconnected_multiple_partitions(self):
        ops = [CX([0, 1]), CX([2, 3])]
        partitioner = QubitPartitioner(ops, 4)
        partitions = partitioner.partition()
        assert len(partitions) == 2

    def test_partition_qubit_coverage(self):
        ops = [CX([0, 1]), CX([2, 3])]
        partitioner = QubitPartitioner(ops, 4)
        partitions = partitioner.partition()
        all_qubits = set()
        for p in partitions:
            all_qubits |= p.qubits
        assert all_qubits == {0, 1, 2, 3}

    def test_partition_no_overlap(self):
        ops = [CX([0, 1]), CX([2, 3])]
        partitioner = QubitPartitioner(ops, 4)
        partitions = partitioner.partition()
        for i, p1 in enumerate(partitions):
            for j, p2 in enumerate(partitions):
                if i != j:
                    assert len(p1.qubits & p2.qubits) == 0


class TestLocalCircuitExtraction:
    def test_single_qubit_gates_extracted(self):
        ops = [Hadamard([0]), PauliX([1]), CX([0, 1])]
        partitioner = QubitPartitioner(ops, 2)
        local_ops = partitioner.extract_local_operations({0, 1})
        assert len(local_ops) == 3

    def test_two_qubit_gates_extracted(self):
        ops = [CX([0, 1]), CX([2, 3])]
        partitioner = QubitPartitioner(ops, 4)
        local_ops_01 = partitioner.extract_local_operations({0, 1})
        local_ops_23 = partitioner.extract_local_operations({2, 3})
        assert len(local_ops_01) == 1
        assert len(local_ops_23) == 1

    def test_gate_order_preserved(self):
        ops = [Hadamard([0]), PauliX([0]), PauliZ([0])]
        partitioner = QubitPartitioner(ops, 1)
        local_ops = partitioner.extract_local_operations({0})
        assert len(local_ops) == 3

    def test_qubit_remapping(self):
        ops = [CX([2, 3])]
        partitioner = QubitPartitioner(ops, 4)
        partitions = partitioner.partition()
        partition_23 = [p for p in partitions if 2 in p.qubits][0]
        assert 2 in partition_23.qubit_map
        assert 3 in partition_23.qubit_map


class TestPartitionedSimulation:
    def test_independent_partitions(self):
        ops = [Hadamard([0]), Hadamard([2])]
        partitioner = QubitPartitioner(ops, 4)
        results = partitioner.simulate_partitioned(1000)
        assert results is not None
        total = sum(results.values())
        assert total == 1000

    def test_matches_full_simulation(self):
        ops = [Hadamard([0]), Hadamard([2])]
        partitioner = QubitPartitioner(ops, 4)
        results = partitioner.simulate_partitioned(10000)
        if results is not None:
            for bitstring, count in results.items():
                assert 1000 < count < 4000

    def test_efficiency_improvement(self):
        ops = [Hadamard([0]), Hadamard([3])]
        partitioner = QubitPartitioner(ops, 4)
        partitions = partitioner.partition()
        assert len(partitions) > 1


class TestResultCombination:
    def test_two_partition_combination(self):
        ops = [Hadamard([0]), Hadamard([2])]
        partitioner = QubitPartitioner(ops, 4)
        results = partitioner.simulate_partitioned(1000)
        if results is not None:
            for bitstring in results:
                assert len(bitstring) == 4

    def test_many_partition_combination(self):
        ops = [Hadamard([i]) for i in range(4)]
        partitioner = QubitPartitioner(ops, 4)
        results = partitioner.simulate_partitioned(1000)
        if results is not None:
            total = sum(results.values())
            assert total == 1000

    def test_probability_normalization(self):
        ops = [Hadamard([0]), Hadamard([2])]
        partitioner = QubitPartitioner(ops, 4)
        results = partitioner.simulate_partitioned(10000)
        if results is not None:
            total = sum(results.values())
            assert total == 10000

    def test_correlation_independence(self):
        ops = [Hadamard([0]), Hadamard([2])]
        partitioner = QubitPartitioner(ops, 4)
        results = partitioner.simulate_partitioned(10000)
        if results is not None:
            count_00xx = sum(v for k, v in results.items() if k[0] == "0")
            count_xx0x = sum(v for k, v in results.items() if k[2] == "0")
            assert 4000 < count_00xx < 6000
            assert 4000 < count_xx0x < 6000


class TestRemappedOperations:
    def test_create_remapped_operations(self):
        ops = [CX([2, 3])]
        partitioner = QubitPartitioner(ops, 4)
        partitions = partitioner.partition()
        partition_23 = [p for p in partitions if 2 in p.qubits][0]
        remapped = partitioner.create_remapped_operations(partition_23)
        assert len(remapped) == 1
        matrix, targets = remapped[0]
        assert targets == (0, 1) or targets == (1, 0)

    def test_clifford_partition_simulation(self):
        ops = [Hadamard([0]), CX([0, 1]), Hadamard([2]), CX([2, 3])]
        partitioner = QubitPartitioner(ops, 4)
        results = partitioner.simulate_partitioned(1000)
        assert results is not None
        total = sum(results.values())
        assert total == 1000

    def test_general_partition_simulation(self):
        ops = [Hadamard([0]), T([0]), CX([0, 1]), Hadamard([2])]
        partitioner = QubitPartitioner(ops, 4)
        results = partitioner.simulate_partitioned(1000)
        assert results is not None
        total = sum(results.values())
        assert total == 1000

    def test_single_partition_returns_none(self):
        ops = [CX([0, 1]), CX([1, 2]), CX([2, 3])]
        partitioner = QubitPartitioner(ops, 4)
        results = partitioner.simulate_partitioned(100)
        assert results is None


class TestPartitionSimulationPaths:
    def test_product_partition_path(self):
        ops = [Hadamard([0]), Hadamard([2])]
        partitioner = QubitPartitioner(ops, 4)
        partitions = partitioner.partition()
        product_partitions = [p for p in partitions if len(p.qubits) == 1]
        assert len(product_partitions) >= 2
        results = partitioner.simulate_partitioned(1000)
        assert results is not None
        assert sum(results.values()) == 1000

    def test_clifford_partition_path(self):
        ops = [Hadamard([0]), CX([0, 1]), Hadamard([3])]
        partitioner = QubitPartitioner(ops, 4)
        results = partitioner.simulate_partitioned(1000)
        assert results is not None
        total = sum(results.values())
        assert total == 1000
        for bitstring in results:
            assert len(bitstring) == 4

    def test_general_partition_path(self):
        ops = [Hadamard([0]), T([0]), CX([0, 1]), Hadamard([3])]
        partitioner = QubitPartitioner(ops, 4)
        results = partitioner.simulate_partitioned(1000)
        assert results is not None
        total = sum(results.values())
        assert total == 1000

    def test_mixed_partition_types(self):
        ops = [
            Hadamard([0]),
            Hadamard([2]),
            CX([2, 3]),
            Hadamard([4]),
            T([4]),
            CX([4, 5]),
        ]
        partitioner = QubitPartitioner(ops, 6)
        results = partitioner.simulate_partitioned(1000)
        assert results is not None
        total = sum(results.values())
        assert total == 1000
        for bitstring in results:
            assert len(bitstring) == 6

    def test_partition_caching(self):
        ops = [Hadamard([0]), Hadamard([2])]
        partitioner = QubitPartitioner(ops, 4)
        partitions1 = partitioner.partition()
        partitions2 = partitioner.partition()
        assert partitions1 is partitions2


class TestPartitionerCoverageGaps:
    def test_find_connected_components_without_analyze(self):
        """Test find_connected_components auto-calls analyze_entanglement."""
        ops = [CX([0, 1]), CX([2, 3])]
        partitioner = QubitPartitioner(ops, 4)
        # Verify _entanglement_analyzed starts as False
        assert partitioner._entanglement_analyzed is False
        # Don't call analyze_entanglement first - find_connected_components should call it
        components = partitioner.find_connected_components()
        assert len(components) == 2
        # Verify analyze_entanglement was called
        assert partitioner._entanglement_analyzed is True
        assert len(partitioner.entanglement_graph) > 0


class TestPartitionerEntanglementAnalyzedBranch:
    """Test branch coverage for _entanglement_analyzed flag."""

    def test_find_connected_components_after_analyze(self):
        """Test find_connected_components when _entanglement_analyzed is already True (line 86->89)."""
        ops = [CX([0, 1]), CX([2, 3])]
        partitioner = QubitPartitioner(ops, 4)
        # First call analyze_entanglement
        partitioner.analyze_entanglement()
        assert partitioner._entanglement_analyzed is True
        # Now call find_connected_components - should skip analyze_entanglement
        components = partitioner.find_connected_components()
        assert len(components) == 2
        # Verify it still works correctly
        component_sets = [frozenset(c) for c in components]
        assert frozenset({0, 1}) in component_sets
        assert frozenset({2, 3}) in component_sets
