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

from braket.default_simulator.circuit_analyzer import (
    CircuitAnalyzer,
    CircuitClass,
)
from braket.default_simulator.gate_operations import (
    CPhaseShift,
    CX,
    Hadamard,
    PauliX,
    PauliZ,
    RotX,
    RotZ,
    S,
    T,
    XX,
)


class TestCircuitClassification:
    def test_empty_circuit(self):
        analyzer = CircuitAnalyzer([], 3)
        assert analyzer.classify() == CircuitClass.PRODUCT

    def test_single_gate_classification(self):
        ops = [Hadamard([0])]
        analyzer = CircuitAnalyzer(ops, 1)
        assert analyzer.classify() == CircuitClass.PRODUCT

    def test_product_circuit_detection(self):
        ops = [Hadamard([0]), PauliX([1]), RotZ([2], np.pi / 4)]
        analyzer = CircuitAnalyzer(ops, 3)
        assert analyzer.classify() == CircuitClass.PRODUCT

    def test_clifford_circuit_detection(self):
        ops = [Hadamard([0]), CX([0, 1]), S([1])]
        analyzer = CircuitAnalyzer(ops, 2)
        result = analyzer.classify()
        assert result == CircuitClass.CLIFFORD

    def test_non_clifford_detection(self):
        ops = [Hadamard([0]), T([0]), CX([0, 1])]
        analyzer = CircuitAnalyzer(ops, 2)
        result = analyzer.classify()
        assert result != CircuitClass.CLIFFORD

    def test_mixed_circuit_classification(self):
        ops = [Hadamard([0]), RotX([1], np.pi / 3), CX([0, 1])]
        analyzer = CircuitAnalyzer(ops, 2)
        result = analyzer.classify()
        assert result in [CircuitClass.GENERAL, CircuitClass.LOW_ENTANGLEMENT]


class TestEntanglementAnalysis:
    def test_no_entanglement(self):
        ops = [Hadamard([0]), PauliX([1]), PauliZ([2])]
        analyzer = CircuitAnalyzer(ops, 3)
        graph = analyzer.get_entanglement_graph()
        for qubit in range(3):
            assert len(graph[qubit]) == 0

    def test_single_cnot(self):
        ops = [CX([0, 1])]
        analyzer = CircuitAnalyzer(ops, 2)
        graph = analyzer.get_entanglement_graph()
        assert 1 in graph[0]
        assert 0 in graph[1]

    def test_linear_chain(self):
        ops = [CX([0, 1]), CX([1, 2]), CX([2, 3])]
        analyzer = CircuitAnalyzer(ops, 4)
        graph = analyzer.get_entanglement_graph()
        assert 1 in graph[0]
        assert 0 in graph[1] and 2 in graph[1]
        assert 1 in graph[2] and 3 in graph[2]
        assert 2 in graph[3]


class TestConnectedComponents:
    def test_single_component(self):
        ops = [CX([0, 1]), CX([1, 2])]
        analyzer = CircuitAnalyzer(ops, 3)
        components = analyzer.get_connected_components()
        assert len(components) == 1
        assert components[0] == {0, 1, 2}

    def test_two_components(self):
        ops = [CX([0, 1]), CX([2, 3])]
        analyzer = CircuitAnalyzer(ops, 4)
        components = analyzer.get_connected_components()
        assert len(components) == 2
        component_sets = [frozenset(c) for c in components]
        assert frozenset({0, 1}) in component_sets
        assert frozenset({2, 3}) in component_sets

    def test_many_components(self):
        ops = [Hadamard([0]), PauliX([1]), PauliZ([2])]
        analyzer = CircuitAnalyzer(ops, 3)
        components = analyzer.get_connected_components()
        assert len(components) == 3


class TestBondDimensionEstimation:
    def test_product_state_bond_dim(self):
        ops = [Hadamard([0]), PauliX([1])]
        analyzer = CircuitAnalyzer(ops, 2)
        bond_dim = analyzer.estimate_bond_dimension()
        assert bond_dim == 1

    def test_entangled_state_bond_dim(self):
        ops = [Hadamard([0]), CX([0, 1]), CX([1, 2])]
        analyzer = CircuitAnalyzer(ops, 3)
        bond_dim = analyzer.estimate_bond_dimension()
        assert bond_dim >= 2


class TestAnalysisReport:
    def test_report_completeness(self):
        ops = [Hadamard([0]), CX([0, 1])]
        analyzer = CircuitAnalyzer(ops, 2)
        report = analyzer.analyze()

        assert report.n_qubits == 2
        assert report.gate_count == 2
        assert isinstance(report.gate_distribution, dict)
        assert isinstance(report.circuit_class, CircuitClass)
        assert isinstance(report.connected_components, list)
        assert isinstance(report.recommended_backend, str)

    def test_gate_count_accuracy(self):
        ops = [Hadamard([0]), PauliX([1]), CX([0, 1]), S([0])]
        analyzer = CircuitAnalyzer(ops, 2)
        report = analyzer.analyze()
        assert report.gate_count == 4

    def test_gate_distribution_accuracy(self):
        ops = [Hadamard([0]), Hadamard([1]), CX([0, 1])]
        analyzer = CircuitAnalyzer(ops, 2)
        report = analyzer.analyze()
        assert report.gate_distribution.get("hadamard", 0) == 2
        assert report.gate_distribution.get("cx", 0) == 1

    def test_recommendation_validity(self):
        ops = [Hadamard([0]), PauliX([1])]
        analyzer = CircuitAnalyzer(ops, 2)
        report = analyzer.analyze()
        valid_backends = ["product", "clifford", "mps", "partitioned", "full", "diagonal"]
        assert report.recommended_backend in valid_backends


class TestSubcircuitIdentification:
    def test_uniform_circuit(self):
        ops = [Hadamard([0]), Hadamard([1]), Hadamard([2])]
        analyzer = CircuitAnalyzer(ops, 3)
        regions = analyzer.identify_subcircuit_classes()
        assert len(regions) == 1
        assert regions[0][1] == CircuitClass.PRODUCT

    def test_alternating_regions(self):
        ops = [Hadamard([0]), CX([0, 1]), T([0]), Hadamard([1])]
        analyzer = CircuitAnalyzer(ops, 2)
        regions = analyzer.identify_subcircuit_classes()
        assert len(regions) >= 1
        total_ops = sum(len(r) for r, _ in regions)
        assert total_ops == len(ops)

    def test_empty_circuit_subcircuits(self):
        analyzer = CircuitAnalyzer([], 2)
        regions = analyzer.identify_subcircuit_classes()
        assert len(regions) == 0


class TestDiagonalCircuitDetection:
    def test_diagonal_circuit(self):
        ops = [PauliZ([0]), S([1]), T([0]), RotZ([1], np.pi / 4)]
        analyzer = CircuitAnalyzer(ops, 2)
        result = analyzer.classify()
        assert result == CircuitClass.PRODUCT

    def test_diagonal_with_two_qubit_gates(self):
        from braket.default_simulator.gate_operations import CPhaseShift
        ops = [PauliZ([0]), CPhaseShift([0, 1], np.pi / 4), T([1])]
        analyzer = CircuitAnalyzer(ops, 2)
        result = analyzer.classify()
        assert result == CircuitClass.DIAGONAL


class TestQFTLikeDetection:
    def test_qft_like_circuit(self):
        from braket.default_simulator.gate_operations import CPhaseShift
        ops = [
            Hadamard([0]), Hadamard([1]), Hadamard([2]), Hadamard([3]),
            CPhaseShift([0, 1], np.pi / 2),
            CPhaseShift([0, 2], np.pi / 4),
            CPhaseShift([0, 3], np.pi / 8),
            CPhaseShift([1, 2], np.pi / 2),
            CPhaseShift([1, 3], np.pi / 4),
            CPhaseShift([2, 3], np.pi / 2),
        ]
        analyzer = CircuitAnalyzer(ops, 4)
        result = analyzer.classify()
        assert result == CircuitClass.QFT_LIKE


class TestLowEntanglementDetection:
    def test_nearest_neighbor_circuit(self):
        ops = [
            Hadamard([0]), CX([0, 1]),
            Hadamard([1]), CX([1, 2]),
            Hadamard([2]), CX([2, 3]),
        ]
        analyzer = CircuitAnalyzer(ops, 4)
        report = analyzer.analyze()
        assert report.is_nearest_neighbor is True

    def test_non_nearest_neighbor_circuit(self):
        ops = [Hadamard([0]), CX([0, 3])]
        analyzer = CircuitAnalyzer(ops, 4)
        report = analyzer.analyze()
        assert report.is_nearest_neighbor is False
        assert report.max_gate_distance == 3


class TestPartitionedBackendRecommendation:
    def test_disconnected_components_recommend_partitioned(self):
        ops = [
            Hadamard([0]), T([0]), CX([0, 1]),
            Hadamard([2]), T([2]), CX([2, 3]),
            Hadamard([4]), T([4]), CX([4, 5]),
            Hadamard([6]), T([6]), CX([6, 7]),
        ]
        analyzer = CircuitAnalyzer(ops, 8)
        report = analyzer.analyze()
        assert len(report.connected_components) > 1


class TestFusableBlocks:
    def test_get_fusable_blocks_empty(self):
        analyzer = CircuitAnalyzer([], 2)
        blocks = analyzer.get_fusable_blocks()
        assert blocks == []

    def test_get_fusable_blocks_single_op(self):
        ops = [Hadamard([0])]
        analyzer = CircuitAnalyzer(ops, 2)
        blocks = analyzer.get_fusable_blocks()
        assert blocks == []

    def test_get_fusable_blocks_adjacent(self):
        ops = [Hadamard([0]), CX([0, 1]), S([1])]
        analyzer = CircuitAnalyzer(ops, 2)
        blocks = analyzer.get_fusable_blocks(max_block_qubits=4)
        assert len(blocks) >= 1

    def test_get_fusable_blocks_non_overlapping(self):
        ops = [Hadamard([0]), Hadamard([2]), Hadamard([4])]
        analyzer = CircuitAnalyzer(ops, 5)
        blocks = analyzer.get_fusable_blocks(max_block_qubits=2)
        assert blocks == []

    def test_get_fusable_blocks_max_qubits_limit(self):
        ops = [Hadamard([0]), CX([0, 1]), CX([1, 2]), CX([2, 3]), CX([3, 4])]
        analyzer = CircuitAnalyzer(ops, 5)
        blocks = analyzer.get_fusable_blocks(max_block_qubits=3)
        for block in blocks:
            block_qubits = set()
            for idx in block:
                block_qubits.update(ops[idx].targets)
            assert len(block_qubits) <= 3


class TestHighQubitIndexHandling:
    def test_operations_with_high_qubit_indices(self):
        ops = [Hadamard([5]), CX([5, 7])]
        analyzer = CircuitAnalyzer(ops, 3)
        report = analyzer.analyze()
        assert report.gate_count == 2
        components = report.connected_components
        all_qubits = set()
        for c in components:
            all_qubits.update(c)
        assert 5 in all_qubits
        assert 7 in all_qubits


class TestGeneralCircuitClassification:
    def test_high_entanglement_non_nearest_neighbor(self):
        ops = []
        n = 14
        for i in range(n):
            ops.append(Hadamard([i]))
        for i in range(n):
            for j in range(i + 2, n):
                ops.append(CX([i, j]))
                ops.append(T([i]))
        analyzer = CircuitAnalyzer(ops, n)
        report = analyzer.analyze()
        assert report.circuit_class == CircuitClass.GENERAL
        assert report.recommended_backend == "full"

    def test_high_bond_dimension_triggers_general(self):
        ops = []
        n = 16
        for i in range(n):
            ops.append(Hadamard([i]))
        for i in range(n - 2):
            ops.append(CX([i, i + 2]))
            ops.append(T([i]))
        analyzer = CircuitAnalyzer(ops, n)
        report = analyzer.analyze()
        assert report.estimated_bond_dimension > 64
        assert report.circuit_class == CircuitClass.GENERAL

    def test_low_entanglement_non_nearest_neighbor_low_bond_dim(self):
        ops = [Hadamard([0]), CX([0, 2]), T([0]), T([1])]
        analyzer = CircuitAnalyzer(ops, 4)
        report = analyzer.analyze()
        assert report.is_nearest_neighbor is False
        assert report.estimated_bond_dimension <= 64
        assert report.circuit_class == CircuitClass.LOW_ENTANGLEMENT
        assert report.recommended_backend == "mps"


class TestFusableBlocksFinalAppend:
    def test_fusable_blocks_ends_with_multi_op_block(self):
        ops = [Hadamard([0]), CX([0, 1]), S([1]), CX([0, 1])]
        analyzer = CircuitAnalyzer(ops, 2)
        blocks = analyzer.get_fusable_blocks(max_block_qubits=4)
        assert len(blocks) >= 1
        total_ops_in_blocks = sum(len(b) for b in blocks)
        assert total_ops_in_blocks >= 2

    def test_fusable_blocks_all_connected(self):
        ops = [Hadamard([0]), CX([0, 1]), CX([0, 1]), S([0])]
        analyzer = CircuitAnalyzer(ops, 2)
        blocks = analyzer.get_fusable_blocks(max_block_qubits=4)
        assert len(blocks) == 1
        assert len(blocks[0]) == 4


class TestSubcircuitClassificationBranches:
    def test_classify_diagonal_op(self):
        ops = [RotZ([0], np.pi / 4), CX([0, 1])]
        analyzer = CircuitAnalyzer(ops, 2)
        regions = analyzer.identify_subcircuit_classes()
        assert len(regions) >= 1

    def test_classify_general_op(self):
        ops = [RotX([0], np.pi / 3), CX([0, 1])]
        analyzer = CircuitAnalyzer(ops, 2)
        regions = analyzer.identify_subcircuit_classes()
        assert len(regions) >= 1

    def test_subcircuit_with_diagonal_two_qubit(self):
        ops = [CPhaseShift([0, 1], np.pi / 4), CX([0, 1])]
        analyzer = CircuitAnalyzer(ops, 2)
        regions = analyzer.identify_subcircuit_classes()
        assert len(regions) >= 1
        classes = [c for _, c in regions]
        assert CircuitClass.DIAGONAL in classes or CircuitClass.CLIFFORD in classes


class TestFusableBlocksBranchCoverage:
    def test_fusable_blocks_single_op_final_block(self):
        ops = [Hadamard([0]), CX([0, 1]), Hadamard([2])]
        analyzer = CircuitAnalyzer(ops, 3)
        blocks = analyzer.get_fusable_blocks(max_block_qubits=4)
        assert isinstance(blocks, list)

    def test_fusable_blocks_exceeds_max_qubits(self):
        ops = [Hadamard([0]), CX([0, 1]), CX([1, 2]), CX([2, 3])]
        analyzer = CircuitAnalyzer(ops, 4)
        blocks = analyzer.get_fusable_blocks(max_block_qubits=2)
        for block in blocks:
            assert len(block) <= 2

    def test_fusable_blocks_final_single_op_not_appended(self):
        ops = [Hadamard([0]), CX([0, 1]), Hadamard([3])]
        analyzer = CircuitAnalyzer(ops, 4)
        blocks = analyzer.get_fusable_blocks(max_block_qubits=4)
        for block in blocks:
            assert len(block) > 1


class TestClassifyOpBranches:
    def test_classify_two_qubit_general_gate(self):
        from braket.default_simulator.gate_operations import XX
        ops = [XX([0, 1], np.pi / 4)]
        analyzer = CircuitAnalyzer(ops, 2)
        regions = analyzer.identify_subcircuit_classes()
        assert len(regions) == 1
        assert regions[0][1] == CircuitClass.GENERAL

    def test_classify_two_qubit_clifford_gate(self):
        ops = [CX([0, 1])]
        analyzer = CircuitAnalyzer(ops, 2)
        regions = analyzer.identify_subcircuit_classes()
        assert len(regions) == 1
        assert regions[0][1] == CircuitClass.CLIFFORD

    def test_classify_two_qubit_diagonal_gate(self):
        ops = [CPhaseShift([0, 1], np.pi / 4)]
        analyzer = CircuitAnalyzer(ops, 2)
        regions = analyzer.identify_subcircuit_classes()
        assert len(regions) == 1
        assert regions[0][1] == CircuitClass.DIAGONAL
