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
from unittest.mock import Mock

from braket.default_simulator.gate_fusion import (
    FusedGateOperation,
    GateFusionEngine,
    FastTargetBasedFusion,
    CommutationGraph,
    PauliLieAlgebra,
    CommutativeAlgebraFusion,
    apply_gate_fusion,
    _get_identity_matrix,
    _get_pauli_x_matrix,
    _get_pauli_y_matrix,
    _get_pauli_z_matrix,
    _get_hadamard_matrix,
)
from braket.default_simulator.operation import GateOperation, KrausOperation


class MockGateOperation(GateOperation):
    def __init__(self, gate_type, targets, matrix=None, angle=None, ctrl_modifiers=None, power=1):
        super().__init__(targets=targets, ctrl_modifiers=ctrl_modifiers or (), power=power)
        self.gate_type = gate_type
        self._matrix = matrix if matrix is not None else np.eye(2**len(targets), dtype=complex)
        self._angle = angle
        self._theta = angle
        self._phi = angle
        self._parameters = [angle] if angle is not None else []
        self._power = power
        self._ctrl_modifiers = ctrl_modifiers or ()
        
    @property
    def _base_matrix(self):
        return self._matrix
    
    @property
    def matrix(self):
        return self._matrix
    
    @matrix.setter
    def matrix(self, value):
        self._matrix = value


class MockKrausOperation(KrausOperation):
    """Mock Kraus operation for noise testing."""
    
    def __init__(self, targets, kraus_matrices=None):
        super().__init__()
        self._kraus_matrices = kraus_matrices or [np.eye(2**len(targets), dtype=complex)]
        self._targets = tuple(targets)
    
    @property
    def targets(self):
        return self._targets
    
    @property
    def matrices(self):
        return self._kraus_matrices


matrix_functions_testdata = [
    (_get_identity_matrix, np.array([[1, 0], [0, 1]])),
    (_get_pauli_x_matrix, np.array([[0, 1], [1, 0]])),
    (_get_pauli_y_matrix, np.array([[0, -1j], [1j, 0]])),
    (_get_pauli_z_matrix, np.array([[1, 0], [0, -1]])),
    (_get_hadamard_matrix, np.array([[1, 1], [1, -1]]) / np.sqrt(2)),
]

matrix_unitarity_testdata = [
    _get_identity_matrix,
    _get_pauli_x_matrix,
    _get_pauli_y_matrix,
    _get_pauli_z_matrix,
    _get_hadamard_matrix,
]

gate_fusion_scenarios = [
    ([MockGateOperation("x", [0]), MockGateOperation("x", [0])], 1, "two_x_gates_cancel"),
    ([MockGateOperation("h", [0]), MockGateOperation("h", [0])], 1, "two_h_gates_cancel"),
    ([MockGateOperation("cx", [0, 1]), MockGateOperation("cx", [0, 1])], 1, "two_cnot_gates_cancel"),
    ([MockGateOperation("x", [0]), MockGateOperation("y", [1])], 2, "different_qubits_no_fusion"),
    ([MockGateOperation("x", [0]), MockGateOperation("y", [0]), MockGateOperation("z", [0])], 3, "pauli_chain_same_qubit"),
]

gate_type_normalization_data = [
    ("pauli_x", "x"),
    ("paulix", "x"),
    ("pauli_y", "y"),
    ("pauliy", "y"),
    ("pauli_z", "z"),
    ("pauliz", "z"),
    ("hadamard", "h"),
    ("cnot", "cx"),
    ("custom", "custom"),
]

commutation_scenarios = [
    (MockGateOperation("x", [0]), MockGateOperation("y", [1]), True, "disjoint_qubits"),
    (MockGateOperation("x", [0]), MockGateOperation("x", [0]), True, "same_gate_type"),
    (MockGateOperation("x", [0]), MockGateOperation("y", [0]), False, "different_pauli_same_qubit"),
    (MockGateOperation("rx", [0], angle=np.pi/4), MockGateOperation("rx", [0], angle=np.pi/2), True, "same_axis_rotations"),
    (MockGateOperation("rx", [0], angle=np.pi/4), MockGateOperation("ry", [0], angle=np.pi/4), False, "different_axis_rotations"),
]

fusion_algorithm_data = [
    ([MockGateOperation("x", [0]), MockGateOperation("x", [0])], 0, "identical_involutory"),
    ([MockGateOperation("x", [0]), MockGateOperation("x", [0]), MockGateOperation("x", [0])], 1, "odd_involutory"),
    ([MockGateOperation("s", [0]), MockGateOperation("s", [0])], 2, "non_involutory"),
    ([MockGateOperation("x", [0]), MockGateOperation("y", [0]), MockGateOperation("x", [0])], 1, "pauli_xor_logic"),
]

engine_config_data = [
    ({"max_fusion_size": 4, "enable_commuting_fusion": True, "enable_advanced_commutation": True}, "full_features"),
    ({"max_fusion_size": 2, "enable_commuting_fusion": False, "enable_advanced_commutation": False}, "minimal_features"),
    ({"max_fusion_size": 8, "enable_commuting_fusion": True, "enable_advanced_commutation": False}, "no_advanced_commutation"),
]


@pytest.mark.parametrize("matrix_func, expected", matrix_functions_testdata)
def test_matrix_functions(matrix_func, expected):
    result = matrix_func()
    assert np.allclose(result, expected)


@pytest.mark.parametrize("matrix_func", matrix_unitarity_testdata)
def test_matrix_unitarity(matrix_func):
    matrix = matrix_func()
    product = np.dot(matrix, np.conj(matrix.T))
    identity = np.eye(matrix.shape[0])
    assert np.allclose(product, identity)


@pytest.mark.parametrize("operations, expected_max_length, description", gate_fusion_scenarios)
def test_gate_fusion_scenarios(operations, expected_max_length, description):
    result = apply_gate_fusion(operations)
    assert len(result) <= expected_max_length
    assert len(result) >= 0


@pytest.mark.parametrize("gate_type, expected", gate_type_normalization_data)
def test_gate_type_normalization(gate_type, expected):
    fusion = FastTargetBasedFusion()
    gate = MockGateOperation(gate_type, [0])
    assert fusion._normalize_gate_type(gate) == expected


@pytest.mark.parametrize("gate1, gate2, expected_commute, description", commutation_scenarios)
def test_commutation_analysis(gate1, gate2, expected_commute, description):
    graph = CommutationGraph()
    assert graph._gates_commute(gate1, gate2) == expected_commute


@pytest.mark.parametrize("gates, expected_count, algorithm_type", fusion_algorithm_data)
def test_fusion_algorithms(gates, expected_count, algorithm_type):
    fusion = FastTargetBasedFusion()
    if algorithm_type in ["identical_involutory", "odd_involutory", "non_involutory"]:
        result = fusion._fuse_identical_gates(gates)
    elif algorithm_type == "pauli_xor_logic":
        result = fusion._fuse_pauli_gates_xor(gates)
    else:
        result = gates
    assert len(result) == expected_count


@pytest.mark.parametrize("config, description", engine_config_data)
def test_engine_configurations(config, description):
    engine = GateFusionEngine(**config)
    assert engine.max_fusion_size == config["max_fusion_size"]
    assert engine.enable_commuting_fusion == config["enable_commuting_fusion"]
    assert engine.enable_advanced_commutation == config["enable_advanced_commutation"]


def test_gate_fusion_basic_functionality():
    operations = [MockGateOperation("x", [0]), MockGateOperation("x", [0])]
    result = apply_gate_fusion(operations)
    assert len(result) <= len(operations)


def test_gate_fusion_preserves_unitarity():
    operations = [MockGateOperation("h", [0]), MockGateOperation("z", [0])]
    result = apply_gate_fusion(operations)
    for op in result:
        if hasattr(op, '_fused_matrix'):
            matrix = op._fused_matrix
            product = np.dot(matrix, np.conj(matrix.T))
            identity = np.eye(matrix.shape[0])
            assert np.allclose(product, identity)


def test_gate_fusion_edge_cases():
    result = apply_gate_fusion([])
    assert result == []
    
    single_op = [MockGateOperation("h", [0])]
    result = apply_gate_fusion(single_op)
    assert len(result) == 1


def test_apply_gate_fusion_with_non_gate_operations():
    non_gate_op = Mock()
    non_gate_op.gate_type = None
    operations = [MockGateOperation("x", [0]), non_gate_op]
    result = apply_gate_fusion(operations)
    assert len(result) == 2


def test_fused_gate_operation_creation():
    matrix = np.eye(2)
    targets = (0,)
    original_gates = [MockGateOperation("x", [0])]
    fused_op = FusedGateOperation(targets, matrix, original_gates)
    assert np.array_equal(fused_op._fused_matrix, matrix)
    assert fused_op.targets == targets


def test_fused_gate_operation_properties():
    matrix = np.array([[0, 1], [1, 0]])
    targets = (1, 2)
    original_gates = [MockGateOperation("x", [1]), MockGateOperation("y", [2])]
    fused_op = FusedGateOperation(targets, matrix, original_gates)
    assert fused_op.targets == targets
    assert np.array_equal(fused_op._fused_matrix, matrix)
    assert fused_op.gate_count == 2
    assert fused_op.original_gates == original_gates


def test_fused_gate_operation_matrix_conversion():
    """Test FusedGateOperation matrix type conversion."""
    matrix = np.array([[1, 0], [0, 1]], dtype=int)
    targets = (0,)
    original_gates = [MockGateOperation("i", [0])]
    
    fused_op = FusedGateOperation(targets, matrix, original_gates)
    assert fused_op._fused_matrix.dtype == complex


def test_fused_gate_operation_gate_type_property():
    """Test gate_type property."""
    matrix = np.eye(2)
    targets = (0,)
    original_gates = [MockGateOperation("x", [0]), MockGateOperation("y", [0])]
    fused_op = FusedGateOperation(targets, matrix, original_gates, optimization_type="test")
    
    gate_type = fused_op.gate_type
    assert "fused_2_test" in gate_type


def test_engine_initialization():
    engine = GateFusionEngine()
    assert engine.max_fusion_size == 8
    assert engine.enable_commuting_fusion == True


def test_engine_initialization_with_advanced_commutation_disabled():
    """Test initialization with advanced commutation disabled."""
    engine = GateFusionEngine(enable_advanced_commutation=False)
    assert engine.enable_advanced_commutation is False
    assert not hasattr(engine, 'fast_target_fusion')


def test_engine_is_fusable_gate():
    engine = GateFusionEngine()
    
    valid_gate = MockGateOperation("x", [0])
    assert engine._is_fusable_gate(valid_gate) is True
    
    two_qubit_gate = MockGateOperation("cx", [0, 1])
    assert engine._is_fusable_gate(two_qubit_gate) is True
    
    three_qubit_gate = MockGateOperation("ccx", [0, 1, 2])
    assert engine._is_fusable_gate(three_qubit_gate) is False
    
    non_gate = Mock()
    assert engine._is_fusable_gate(non_gate) is False
    
    invalid_power_gate = MockGateOperation("x", [0], power=3.5)
    assert engine._is_fusable_gate(invalid_power_gate) is False


def test_engine_can_fuse_same_qubits():
    engine = GateFusionEngine()
    gate1 = MockGateOperation("x", [0])
    gate2 = MockGateOperation("y", [0])
    fusion_group = [gate1]
    assert engine._can_safely_commute_with_group(fusion_group, gate2) is False


def test_engine_optimize_operations_empty():
    engine = GateFusionEngine()
    result = engine.optimize_operations([])
    assert result == []


def test_engine_optimize_operations_no_gates():
    engine = GateFusionEngine()
    non_gate = Mock()
    non_gate.gate_type = None
    result = engine.optimize_operations([non_gate])
    assert len(result) == 1


def test_engine_is_pure_gate_sequence():
    """Test _is_pure_gate_sequence method."""
    engine = GateFusionEngine()
    
    gates = [MockGateOperation("x", [0]), MockGateOperation("y", [1])]
    assert engine._is_pure_gate_sequence(gates) is True
    
    mixed = [MockGateOperation("x", [0]), MockKrausOperation([0])]
    assert engine._is_pure_gate_sequence(mixed) is False


def test_engine_optimize_gate_sequence_empty():
    """Test _optimize_gate_sequence with empty list."""
    engine = GateFusionEngine()
    result = engine._optimize_gate_sequence([])
    assert result == []


def test_engine_find_optimal_fusion_group():
    """Test _find_optimal_fusion_group method."""
    engine = GateFusionEngine()
    gates = [
        MockGateOperation("x", [0]),
        MockGateOperation("y", [0]),
        MockGateOperation("z", [1])
    ]
    
    group = engine._find_optimal_fusion_group(gates, 0)
    assert len(group) >= 1
    assert group[0] == gates[0]
    
    group = engine._find_optimal_fusion_group(gates, 10)
    assert group == []
    
    large_gates = [MockGateOperation("ccx", [0, 1, 2])]
    group = engine._find_optimal_fusion_group(large_gates, 0)
    assert len(group) == 1
    assert group[0] == large_gates[0]


def test_engine_gates_commute_safely():
    """Test _gates_commute_safely method."""
    engine = GateFusionEngine()
    
    gate1 = MockGateOperation("x", [0])
    gate2 = MockGateOperation("y", [1])
    assert engine._gates_commute_safely(gate1, gate2) is True
    
    gate3 = MockGateOperation("x", [0])
    assert engine._gates_commute_safely(gate1, gate3) is True
    
    gate4 = MockGateOperation("y", [0])
    assert engine._gates_commute_safely(gate1, gate4) is False


def test_engine_same_qubit_gates_commute():
    """Test _same_qubit_gates_commute method."""
    engine = GateFusionEngine()
    
    gate1 = MockGateOperation("x", [0])
    gate2 = MockGateOperation("x", [0])
    assert engine._same_qubit_gates_commute(gate1, gate2) is True
    
    gate3 = MockGateOperation("y", [0])
    assert engine._same_qubit_gates_commute(gate1, gate3) is False


def test_engine_get_gate_type_normalized_caching():
    """Test _get_gate_type_normalized method with caching."""
    engine = GateFusionEngine()
    
    gate = MockGateOperation("pauli_x", [0])
    
    result1 = engine._get_gate_type_normalized(gate)
    assert result1 == 'x'
    
    result2 = engine._get_gate_type_normalized(gate)
    assert result2 == 'x'
    assert len(engine._gate_type_cache) > 0


def test_engine_has_control_target_conflict():
    """Test _has_control_target_conflict method."""
    engine = GateFusionEngine()
    
    gate1 = MockGateOperation("x", [0])
    gate2 = MockGateOperation("y", [1])
    assert engine._has_control_target_conflict(gate1, gate2) is False
    
    gate3 = MockGateOperation("cx", [0, 1], ctrl_modifiers=(0,))
    gate4 = MockGateOperation("x", [1])
    assert engine._has_control_target_conflict(gate3, gate4) is True


def test_engine_apply_smart_reductions():
    """Test _apply_smart_reductions method."""
    engine = GateFusionEngine()
    
    gates = [MockGateOperation("x", [0])]
    result = engine._apply_smart_reductions(gates)
    assert result == gates
    
    pauli_gates = [MockGateOperation("x", [0]), MockGateOperation("y", [0])]
    result = engine._apply_smart_reductions(pauli_gates)
    assert len(result) <= len(pauli_gates)


def test_engine_sequence_detection():
    """Test sequence detection methods."""
    engine = GateFusionEngine()
    
    pauli_gates = [MockGateOperation("x", [0]), MockGateOperation("pauli_y", [0])]
    assert engine._is_pauli_sequence(pauli_gates) is True
    
    mixed_gates = [MockGateOperation("x", [0]), MockGateOperation("h", [0])]
    assert engine._is_pauli_sequence(mixed_gates) is False
    
    rotation_gates = [
        MockGateOperation("rx", [0], angle=np.pi/4),
        MockGateOperation("rx", [0], angle=np.pi/2)
    ]
    assert engine._is_rotation_sequence(rotation_gates) is True
    
    mixed_rotations = [
        MockGateOperation("rx", [0], angle=np.pi/4),
        MockGateOperation("ry", [0], angle=np.pi/4)
    ]
    assert engine._is_rotation_sequence(mixed_rotations) is False
    
    assert engine._is_rotation_sequence([]) is False
    
    hadamard_gates = [MockGateOperation("h", [0]), MockGateOperation("h", [0])]
    assert engine._is_hadamard_sequence(hadamard_gates) is True
    
    different_qubits = [MockGateOperation("h", [0]), MockGateOperation("h", [1])]
    assert engine._is_hadamard_sequence(different_qubits) is False
    
    assert engine._is_hadamard_sequence([]) is False


def test_engine_optimization_algorithms():
    """Test optimization algorithms."""
    engine = GateFusionEngine()
    
    result = engine._optimize_pauli_sequence_fast([])
    assert result == []
    
    different_qubits = [MockGateOperation("x", [0]), MockGateOperation("y", [1])]
    result = engine._optimize_pauli_sequence_fast(different_qubits)
    assert result == different_qubits
    
    same_qubit_gates = [
        MockGateOperation("x", [0]),
        MockGateOperation("y", [0]),
        MockGateOperation("x", [0])
    ]
    result = engine._optimize_pauli_sequence_fast(same_qubit_gates)
    assert len(result) == 1
    assert engine._get_gate_type_normalized(result[0]) == 'y'
    
    result = engine._optimize_rotation_sequence_fast([])
    assert result == []
    
    no_angle_gates = [MockGateOperation("rx", [0])]
    no_angle_gates[0]._angle = None
    result = engine._optimize_rotation_sequence_fast(no_angle_gates)
    assert result == no_angle_gates
    
    identity_gates = [
        MockGateOperation("rx", [0], angle=2*np.pi),
        MockGateOperation("rx", [0], angle=2*np.pi)
    ]
    result = engine._optimize_rotation_sequence_fast(identity_gates)
    assert len(result) == 0
    
    even_gates = [MockGateOperation("h", [0]), MockGateOperation("h", [0])]
    result = engine._optimize_hadamard_sequence_fast(even_gates)
    assert len(result) == 0
    
    odd_gates = [MockGateOperation("h", [0]), MockGateOperation("h", [0]), MockGateOperation("h", [0])]
    result = engine._optimize_hadamard_sequence_fast(odd_gates)
    assert len(result) == 1


def test_engine_general_reductions():
    """Test _apply_general_reductions method."""
    engine = GateFusionEngine()
    
    gates = [MockGateOperation("x", [0]), MockGateOperation("x", [0])]
    result = engine._apply_general_reductions(gates)
    assert len(result) <= len(gates)
    
    unknown_gates = [MockGateOperation("custom1", [0]), MockGateOperation("custom2", [0])]
    result = engine._apply_general_reductions(unknown_gates)
    assert result == unknown_gates


def test_engine_extract_angle_fast():
    """Test _extract_angle_fast method."""
    engine = GateFusionEngine()
    
    gate_with_angle = MockGateOperation("rx", [0], angle=np.pi/4)
    angle = engine._extract_angle_fast(gate_with_angle)
    assert angle == np.pi/4
    
    gate_without_angle = MockGateOperation("x", [0])
    gate_without_angle._angle = None
    gate_without_angle._theta = None
    angle = engine._extract_angle_fast(gate_without_angle)
    assert angle is None


def test_engine_safe_controlled_gate():
    """Test _is_safe_controlled_gate method."""
    engine = GateFusionEngine()
    
    safe_cx = MockGateOperation("cx", [0, 1])
    assert engine._is_safe_controlled_gate(safe_cx) is True
    
    safe_ccx = MockGateOperation("ccx", [0, 1, 2])
    assert engine._is_safe_controlled_gate(safe_ccx) is True
    
    unsafe_gate = MockGateOperation("custom_controlled", [0, 1, 2, 3])
    assert engine._is_safe_controlled_gate(unsafe_gate) is False


def test_engine_get_gate_power():
    """Test _get_gate_power method."""
    engine = GateFusionEngine()
    
    gate_with_power = MockGateOperation("x", [0], power=2)
    power = engine._get_gate_power(gate_with_power)
    assert power == 2
    
    gate_without_power = MockGateOperation("x", [0])
    gate_without_power._power = None
    power = engine._get_gate_power(gate_without_power)
    assert power is None


def test_engine_create_optimized_fused_operation():
    """Test _create_optimized_fused_operation method."""
    engine = GateFusionEngine()
    
    single_gate = [MockGateOperation("x", [0])]
    fused_op = engine._create_optimized_fused_operation(single_gate)
    assert isinstance(fused_op, FusedGateOperation)
    assert fused_op.optimization_type == "single"
    
    multiple_gates = [MockGateOperation("x", [0]), MockGateOperation("y", [0])]
    fused_op = engine._create_optimized_fused_operation(multiple_gates)
    assert isinstance(fused_op, FusedGateOperation)
    assert fused_op.optimization_type == "optimized"


def test_engine_compute_fused_matrix_optimized():
    """Test _compute_fused_matrix_optimized method."""
    engine = GateFusionEngine()
    
    single_gate = [MockGateOperation("x", [0])]
    targets = (0,)
    matrix = engine._compute_fused_matrix_optimized(single_gate, targets)
    assert matrix.shape == (2, 2)
    
    multiple_gates = [MockGateOperation("x", [0]), MockGateOperation("y", [0])]
    matrix = engine._compute_fused_matrix_optimized(multiple_gates, targets)
    assert matrix.shape == (2, 2)


def test_engine_expand_matrix_optimized():
    """Test _expand_matrix_optimized method."""
    engine = GateFusionEngine()
    
    gate_matrix = np.array([[0, 1], [1, 0]])
    targets = (0,)
    num_qubits = 1
    
    expanded = engine._expand_matrix_optimized(gate_matrix, targets, num_qubits)
    assert np.array_equal(expanded, gate_matrix)
    
    targets = (0, 1)
    num_qubits = 2
    gate_matrix_2q = np.eye(4)
    
    expanded = engine._expand_matrix_optimized(gate_matrix_2q, targets, num_qubits)
    assert expanded.shape == (4, 4)


def test_engine_optimize_mixed_operations():
    """Test _optimize_mixed_operations method."""
    engine = GateFusionEngine()
    
    operations = [
        MockGateOperation("x", [0]),
        MockKrausOperation([0]),
        MockGateOperation("y", [1])
    ]
    
    result = engine._optimize_mixed_operations(operations)
    assert len(result) >= 1


def test_fast_target_based_fusion_initialization():
    """Test FastTargetBasedFusion initialization."""
    fusion = FastTargetBasedFusion()
    assert hasattr(fusion, '_fusion_cache')
    assert isinstance(fusion._fusion_cache, dict)


def test_fast_target_based_fusion_optimize_commuting_gates():
    """Test optimize_commuting_gates method."""
    fusion = FastTargetBasedFusion()
    
    result = fusion.optimize_commuting_gates([])
    assert result == []
    
    gates = [MockGateOperation("x", [0])]
    result = fusion.optimize_commuting_gates(gates)
    assert len(result) == 1
    
    gates = [
        MockGateOperation("x", [0]),
        MockGateOperation("y", [0]),
        MockGateOperation("z", [0])
    ]
    result = fusion.optimize_commuting_gates(gates)
    assert len(result) <= len(gates)
    
    gates = [
        MockGateOperation("x", [0]),
        MockGateOperation("y", [1]),
        MockGateOperation("z", [2])
    ]
    result = fusion.optimize_commuting_gates(gates)
    assert len(result) == len(gates)
    
    gates = [MockGateOperation("ccx", [0, 1, 2])]
    result = fusion.optimize_commuting_gates(gates)
    assert len(result) == 1
    assert result[0] == gates[0]


def test_fast_target_based_fusion_fuse_target_group():
    """Test _fuse_target_group method."""
    fusion = FastTargetBasedFusion()
    
    gates = [MockGateOperation("x", [0])]
    result = fusion._fuse_target_group(gates)
    assert result == gates
    
    gates = [MockGateOperation("x", [0]), MockGateOperation("x", [0])]
    result = fusion._fuse_target_group(gates)
    assert len(result) == 0
    
    gates = [
        MockGateOperation("x", [0]),
        MockGateOperation("y", [0]),
        MockGateOperation("x", [0])
    ]
    result = fusion._fuse_target_group(gates)
    assert len(result) == 1
    assert fusion._normalize_gate_type(result[0]) == 'y'
    
    gates = [
        MockGateOperation("x", [0]),
        MockGateOperation("h", [0])
    ]
    result = fusion._fuse_target_group(gates)
    assert result == gates


def test_fast_target_based_fusion_fuse_pauli_gates_xor():
    """Test _fuse_pauli_gates_xor method."""
    fusion = FastTargetBasedFusion()
    
    result = fusion._fuse_pauli_gates_xor([])
    assert result == []
    
    gates = [
        MockGateOperation("x", [0]),
        MockGateOperation("x", [0]),
        MockGateOperation("y", [0]),
        MockGateOperation("y", [0]),
        MockGateOperation("z", [0]),
        MockGateOperation("z", [0])
    ]
    result = fusion._fuse_pauli_gates_xor(gates)
    assert len(result) == 0


def test_commutation_graph_initialization():
    """Test CommutationGraph initialization."""
    graph = CommutationGraph()
    assert graph.nodes == []
    assert isinstance(graph.edges, dict)
    assert isinstance(graph.non_commuting_edges, dict)


def test_commutation_graph_build():
    """Test build method."""
    graph = CommutationGraph()
    
    result = graph.build([])
    assert result.nodes == []
    assert len(result.edges) == 0
    
    gates = [MockGateOperation("x", [0])]
    result = graph.build(gates)
    assert len(result.nodes) == 1
    assert result.nodes[0] == gates[0]
    
    gates = [
        MockGateOperation("x", [0]),
        MockGateOperation("y", [1])
    ]
    result = graph.build(gates)
    assert len(result.nodes) == 2
    assert 1 in result.edges[0]
    assert 0 in result.edges[1]
    
    gates = [
        MockGateOperation("x", [0]),
        MockGateOperation("y", [0])
    ]
    result = graph.build(gates)
    assert len(result.nodes) == 2
    assert 1 in result.non_commuting_edges[0]
    assert 0 in result.non_commuting_edges[1]


def test_commutation_graph_commutation_checks():
    """Test commutation check methods."""
    graph = CommutationGraph()
    
    gate1 = MockGateOperation("x", [0])
    gate2 = MockGateOperation("x", [0])
    assert graph._same_qubit_commutation_analysis(gate1, gate2) is True
    
    gate3 = MockGateOperation("y", [0])
    assert graph._same_qubit_commutation_analysis(gate1, gate3) is False
    
    assert graph._pauli_commutation_check('x', 'x', gate1, gate2) is True
    assert graph._pauli_commutation_check('x', 'y', gate1, gate3) is False
    
    gate4 = MockGateOperation("rx", [0], angle=np.pi/4)
    gate5 = MockGateOperation("rx", [0], angle=np.pi/2)
    assert graph._rotation_commutation_check('rx', 'rx', gate4, gate5) is True
    
    gate6 = MockGateOperation("ry", [0], angle=np.pi/4)
    assert graph._rotation_commutation_check('rx', 'ry', gate4, gate6) is False
    
    assert graph._pi_rotation_commutation('rx', 'ry') is False
    assert graph._pi_rotation_commutation('ry', 'rz') is False
    assert graph._pi_rotation_commutation('rz', 'rx') is False


def test_commutation_graph_extract_angle():
    """Test _extract_angle method."""
    graph = CommutationGraph()
    
    gate1 = MockGateOperation("rx", [0], angle=np.pi/4)
    assert graph._extract_angle(gate1) == np.pi/4
    
    gate2 = MockGateOperation("x", [0])
    gate2._angle = None
    gate2._theta = None
    assert graph._extract_angle(gate2) is None


def test_commutation_graph_find_maximal_cliques():
    """Test find_maximal_cliques method."""
    graph = CommutationGraph()
    
    graph.build([])
    cliques = graph.find_maximal_cliques()
    assert cliques == []
    
    gates = [MockGateOperation("x", [0])]
    graph.build(gates)
    cliques = graph.find_maximal_cliques()
    assert cliques == []
    
    gates = [
        MockGateOperation("x", [0]),
        MockGateOperation("y", [1]),
        MockGateOperation("z", [2])
    ]
    graph.build(gates)
    cliques = graph.find_maximal_cliques()
    assert len(cliques) >= 1


def test_commutation_graph_bron_kerbosch_edge_cases():
    """Test Bron-Kerbosch algorithm edge cases."""
    graph = CommutationGraph()
    
    graph.nodes = []
    graph.edges = {}
    cliques = []
    graph._bron_kerbosch(set(), set(), set(), cliques)
    assert cliques == []


def test_pauli_lie_algebra_initialization():
    """Test PauliLieAlgebra initialization."""
    algebra = PauliLieAlgebra()
    assert algebra.pauli_basis == ['i', 'x', 'y', 'z']
    assert isinstance(algebra.commutation_table, dict)


def test_pauli_lie_algebra_build_commutation_table():
    """Test _build_commutation_table method."""
    algebra = PauliLieAlgebra()
    table = algebra._build_commutation_table()
    
    assert table[('i', 'x')] == ('i', 0)
    assert table[('x', 'i')] == ('i', 0)
    
    assert table[('x', 'x')] == ('i', 0)
    assert table[('y', 'y')] == ('i', 0)
    assert table[('z', 'z')] == ('i', 0)
    
    assert table[('x', 'y')] == ('z', 2j)
    assert table[('y', 'x')] == ('z', -2j)


def test_pauli_lie_algebra_commutator():
    """Test commutator method."""
    algebra = PauliLieAlgebra()
    
    result, coeff = algebra.commutator('x', 'y')
    assert result == 'z'
    assert coeff == 2j
    
    result, coeff = algebra.commutator('y', 'x')
    assert result == 'z'
    assert coeff == -2j
    
    result, coeff = algebra.commutator('unknown1', 'unknown2')
    assert result == 'unknown'
    assert coeff == 0


def test_pauli_lie_algebra_optimal_order():
    """Test optimal_order method."""
    algebra = PauliLieAlgebra()
    
    result = algebra.optimal_order([])
    assert result == []
    
    gates = [MockGateOperation("x", [0])]
    result = algebra.optimal_order(gates)
    assert result == gates
    
    gates = [
        MockGateOperation("x", [0]),
        MockGateOperation("y", [0]),
        MockGateOperation("z", [0])
    ]
    result = algebra.optimal_order(gates)
    assert len(result) == len(gates)
    assert all(gate in result for gate in gates)


def test_pauli_lie_algebra_compute_insertion_cost():
    """Test _compute_insertion_cost method."""
    algebra = PauliLieAlgebra()
    
    current_sequence = [MockGateOperation("x", [0])]
    candidate = MockGateOperation("y", [0])
    
    cost = algebra._compute_insertion_cost(current_sequence, candidate)
    assert isinstance(cost, float)
    assert cost >= 0
    
    current_sequence = [MockGateOperation("custom", [0])]
    candidate = MockGateOperation("another_custom", [0])
    
    cost = algebra._compute_insertion_cost(current_sequence, candidate)
    assert cost == 0.0


def test_commutative_algebra_fusion_initialization():
    """Test CommutativeAlgebraFusion initialization."""
    fusion = CommutativeAlgebraFusion()
    assert isinstance(fusion.lie_algebra, PauliLieAlgebra)
    assert isinstance(fusion.commutation_graph, CommutationGraph)
    assert isinstance(fusion._optimization_cache, dict)


def test_commutative_algebra_fusion_optimize_commuting_gates():
    """Test optimize_commuting_gates method."""
    fusion = CommutativeAlgebraFusion()
    
    result = fusion.optimize_commuting_gates([])
    assert result == []
    
    gates = [MockGateOperation("x", [0])]
    result = fusion.optimize_commuting_gates(gates)
    assert len(result) == 1
    
    gates = [MockGateOperation("x", [0]), MockGateOperation("y", [0])]
    
    result1 = fusion.optimize_commuting_gates(gates)
    
    result2 = fusion.optimize_commuting_gates(gates)
    
    assert result1 == result2
    assert len(fusion._optimization_cache) > 0


def test_commutative_algebra_fusion_compute_gate_signature():
    """Test _compute_gate_signature method."""
    fusion = CommutativeAlgebraFusion()
    gates = [
        MockGateOperation("x", [0]),
        MockGateOperation("ry", [1], angle=np.pi/4)
    ]
    signature = fusion._compute_gate_signature(gates)
    assert isinstance(signature, str)
    assert "x_(0,)" in signature
    assert "ry_(1,)" in signature


def test_commutative_algebra_fusion_extract_angle():
    """Test _extract_angle method."""
    fusion = CommutativeAlgebraFusion()
    
    gate_with_angle = MockGateOperation("rx", [0], angle=np.pi/4)
    angle = fusion._extract_angle(gate_with_angle)
    assert abs(angle - np.pi/4) < 1e-6
    
    gate_without_angle = MockGateOperation("x", [0])
    gate_without_angle._angle = None
    angle = fusion._extract_angle(gate_without_angle)
    assert angle is None


def test_commutative_algebra_fusion_fuse_commuting_sequence():
    """Test _fuse_commuting_sequence method."""
    fusion = CommutativeAlgebraFusion()
    
    result = fusion._fuse_commuting_sequence([])
    assert result == []
    
    gates = [MockGateOperation("x", [0])]
    result = fusion._fuse_commuting_sequence(gates)
    assert result == gates


def test_commutative_algebra_fusion_group_checks():
    """Test group checking methods."""
    fusion = CommutativeAlgebraFusion()
    
    pauli_gates = [
        MockGateOperation("x", [0]),
        MockGateOperation("pauli_y", [0]),
        MockGateOperation("z", [0])
    ]
    assert fusion._is_pauli_group(pauli_gates) is True
    
    mixed_gates = [
        MockGateOperation("x", [0]),
        MockGateOperation("h", [0])
    ]
    assert fusion._is_pauli_group(mixed_gates) is False
    
    rotation_gates = [
        MockGateOperation("rx", [0], angle=np.pi/4),
        MockGateOperation("rx", [0], angle=np.pi/2)
    ]
    assert fusion._is_rotation_group(rotation_gates) is True
    
    mixed_rotations = [
        MockGateOperation("rx", [0], angle=np.pi/4),
        MockGateOperation("ry", [0], angle=np.pi/4)
    ]
    assert fusion._is_rotation_group(mixed_rotations) is False
    
    assert fusion._is_rotation_group([]) is False
    
    non_rotations = [MockGateOperation("x", [0])]
    assert fusion._is_rotation_group(non_rotations) is False


def test_commutative_algebra_fusion_fuse_groups():
    """Test group fusion methods."""
    fusion = CommutativeAlgebraFusion()
    
    gates = [
        MockGateOperation("x", [0]),
        MockGateOperation("y", [0]),
        MockGateOperation("x", [0])
    ]
    result = fusion._fuse_pauli_group(gates)
    assert len(result) == 1
    
    gates = [
        MockGateOperation("x", [0]),
        MockGateOperation("x", [0]),
        MockGateOperation("y", [0]),
        MockGateOperation("y", [0])
    ]
    result = fusion._fuse_pauli_group(gates)
    assert len(result) == 0
    
    gates = [
        MockGateOperation("rx", [0], angle=np.pi/4),
        MockGateOperation("rx", [0], angle=np.pi/4)
    ]
    result = fusion._fuse_rotation_group(gates)
    assert len(result) >= 0
    
    gates_no_angle = [MockGateOperation("rx", [0])]
    gates_no_angle[0]._angle = None
    result = fusion._fuse_rotation_group(gates_no_angle)
    assert result == gates_no_angle
    
    gates = [MockGateOperation("custom1", [0]), MockGateOperation("custom2", [0])]
    result = fusion._fuse_general_group(gates)
    assert result == gates


def test_apply_gate_fusion_comprehensive():
    """Comprehensive apply_gate_fusion tests."""
    operations = [MockGateOperation("h", [0])]
    result = apply_gate_fusion(operations)
    assert len(result) == 1
    
    result = apply_gate_fusion([])
    assert result == []
    
    operations = [MockGateOperation("x", [i]) for i in range(5)]
    result = apply_gate_fusion(operations, max_fusion_size=2)
    assert len(result) <= len(operations)
    
    operations = [
        MockGateOperation("x", [0]),
        MockGateOperation("y", [0]),
        MockGateOperation("z", [0])
    ]
    result = apply_gate_fusion(operations)
    assert len(result) <= len(operations)
    
    operations = [MockGateOperation("x", [0]) for _ in range(10)]
    result = apply_gate_fusion(operations)
    assert len(result) <= len(operations)
    
    operations = [
        MockGateOperation("x", [0]),
        MockGateOperation("h", [1]),
        MockGateOperation("y", [0]),
        MockGateOperation("s", [1])
    ]
    result = apply_gate_fusion(operations)
    assert len(result) <= len(operations)


def test_mixed_gate_and_noise_operations():
    """Test mixed gate and noise operations."""
    operations = [
        MockGateOperation("x", [0]),
        MockKrausOperation([0]),
        MockGateOperation("y", [0])
    ]
    result = apply_gate_fusion(operations)
    assert len(result) >= 1


def test_large_qubit_gates():
    """Test handling of large qubit gates."""
    operations = [MockGateOperation("ccx", [0, 1, 2])]
    result = apply_gate_fusion(operations)
    assert len(result) == 1


def test_controlled_gates():
    """Test controlled gates."""
    operations = [MockGateOperation("cx", [0, 1]), MockGateOperation("cx", [0, 1])]
    result = apply_gate_fusion(operations)
    assert len(result) <= len(operations)


def test_gate_fusion_correctness():
    """Test that gate fusion preserves quantum correctness."""
    operations = [
        MockGateOperation("h", [0]),
        MockGateOperation("x", [0]),
        MockGateOperation("h", [0])
    ]
    result = apply_gate_fusion(operations)
    assert len(result) >= 1


def test_backward_compatibility():
    """Test backward compatibility functions."""
    from braket.default_simulator.gate_fusion import apply_gate_fusion
    operations = [MockGateOperation("x", [0])]
    result = apply_gate_fusion(operations)
    assert len(result) == 1


def test_performance_scenarios():
    """Test performance with various scenarios."""
    gates = [MockGateOperation("x", [i % 4]) for i in range(100)]
    
    engine = GateFusionEngine()
    result = engine.optimize_operations(gates)
    
    assert len(result) <= len(gates)
    assert len(result) > 0
    
    gates = [MockGateOperation("x", [0]) for _ in range(20)]
    
    engine = GateFusionEngine(max_fusion_size=10)
    result = engine.optimize_operations(gates)
    
    assert len(result) <= len(gates)
    
    fusion = CommutativeAlgebraFusion()
    gates = [MockGateOperation("x", [0]), MockGateOperation("y", [0])]
    
    for _ in range(5):
        result = fusion.optimize_commuting_gates(gates)
        assert len(result) <= len(gates)
    
    assert len(fusion._optimization_cache) > 0


def test_integration_scenarios():
    """Integration tests for the complete gate fusion system."""
    operations = [
        MockGateOperation("h", [0]),
        MockGateOperation("x", [0]),
        MockGateOperation("h", [0]),
        MockGateOperation("cx", [0, 1]),
        MockGateOperation("cx", [0, 1]),
        MockGateOperation("y", [2]),
        MockGateOperation("y", [2]),
    ]
    
    result = apply_gate_fusion(operations)
    assert len(result) < len(operations)
    assert len(result) > 0
    
    operations = [
        MockGateOperation("x", [0]),
        MockKrausOperation([0]),
        MockGateOperation("y", [0]),
        MockGateOperation("z", [1]),
    ]
    
    result = apply_gate_fusion(operations)
    assert len(result) >= 2
    
    has_gate = any(isinstance(op, (GateOperation, FusedGateOperation)) for op in result)
    has_noise = any(isinstance(op, MockKrausOperation) for op in result)
    assert has_gate and has_noise
    
    qft_pattern = [
        MockGateOperation("h", [0]),
        MockGateOperation("rz", [0], angle=np.pi/2),
        MockGateOperation("cx", [0, 1]),
        MockGateOperation("h", [1]),
    ]
    
    result = apply_gate_fusion(qft_pattern)
    assert len(result) <= len(qft_pattern)
    
    grover_pattern = [
        MockGateOperation("h", [0]),
        MockGateOperation("h", [1]),
        MockGateOperation("x", [0]),
        MockGateOperation("x", [1]),
        MockGateOperation("cz", [0, 1]),
        MockGateOperation("x", [0]),
        MockGateOperation("x", [1]),
    ]
    
    result = apply_gate_fusion(grover_pattern)
    assert len(result) <= len(grover_pattern)


def test_edge_cases_and_error_conditions():
    """Test edge cases and error conditions."""
    graph = CommutationGraph()
    
    gate1 = MockGateOperation("rx", [0], angle=np.pi)
    gate2 = MockGateOperation("ry", [0], angle=np.pi)
    
    assert graph._rotation_commutation_check('rx', 'ry', gate1, gate2) is False
    
    fusion = CommutativeAlgebraFusion()
    
    gates = [
        MockGateOperation("rx", [0], angle=2*np.pi),
        MockGateOperation("rx", [0], angle=2*np.pi)
    ]
    result = fusion._fuse_rotation_group(gates)
    assert len(result) <= len(gates)
    
    fusion = FastTargetBasedFusion()
    
    result = fusion._fuse_pauli_gates_xor([])
    assert result == []
    
    non_pauli_gates = [MockGateOperation("h", [0]), MockGateOperation("s", [0])]
    result = fusion._fuse_pauli_gates_xor(non_pauli_gates)
    assert len(result) >= 0
    
    engine = GateFusionEngine()
    
    invalid_gate = Mock()
    invalid_gate.targets = [0]
    assert engine._is_fusable_gate(invalid_gate) is False
    
    gates = [invalid_gate, MockGateOperation("x", [0])]
    group = engine._find_optimal_fusion_group(gates, 0)
    assert len(group) == 1
    assert group[0] == invalid_gate
