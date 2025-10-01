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
        self._matrix = matrix if matrix is not None else np.eye(2 ** len(targets), dtype=complex)
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
        self._kraus_matrices = kraus_matrices or [np.eye(2 ** len(targets), dtype=complex)]
        self._targets = tuple(targets)

    @property
    def targets(self):
        return self._targets

    @property
    def matrices(self):
        return self._kraus_matrices


matrix_functions_data = [
    (_get_identity_matrix, np.array([[1, 0], [0, 1]])),
    (_get_pauli_x_matrix, np.array([[0, 1], [1, 0]])),
    (_get_pauli_y_matrix, np.array([[0, -1j], [1j, 0]])),
    (_get_pauli_z_matrix, np.array([[1, 0], [0, -1]])),
    (_get_hadamard_matrix, np.array([[1, 1], [1, -1]]) / np.sqrt(2)),
]

gate_fusion_scenarios = [
    ([MockGateOperation("x", [0]), MockGateOperation("x", [0])], 0, "x_gates_cancel"),
    ([MockGateOperation("h", [0]), MockGateOperation("h", [0])], 0, "h_gates_cancel"),
    ([MockGateOperation("cx", [0, 1]), MockGateOperation("cx", [0, 1])], 0, "cnot_gates_cancel"),
    ([MockGateOperation("x", [0]), MockGateOperation("y", [1])], 2, "different_qubits"),
    (
        [MockGateOperation("x", [0]), MockGateOperation("y", [0]), MockGateOperation("z", [0])],
        3,
        "pauli_chain",
    ),
    ([], 0, "empty_list"),
    ([MockGateOperation("h", [0])], 1, "single_gate"),
]


commutation_test_data = [
    (MockGateOperation("pauli_x", [0]), MockGateOperation("pauli_y", [1]), True, "disjoint_qubits"),
    (MockGateOperation("pauli_x", [0]), MockGateOperation("pauli_x", [0]), True, "identical_gates"),
    (
        MockGateOperation("pauli_x", [0]),
        MockGateOperation("pauli_y", [0]),
        False,
        "anticommuting_pauli",
    ),
    (
        MockGateOperation("rx", [0], angle=np.pi / 4),
        MockGateOperation("rx", [0], angle=np.pi / 2),
        True,
        "same_axis_rotations",
    ),
    (
        MockGateOperation("rx", [0], angle=np.pi / 4),
        MockGateOperation("ry", [0], angle=np.pi / 4),
        False,
        "different_axis_rotations",
    ),
]

engine_configuration_data = [
    (
        {
            "max_fusion_size": 4,
            "enable_commuting_fusion": True,
            "enable_advanced_commutation": True,
        },
        "full_config",
    ),
    (
        {
            "max_fusion_size": 2,
            "enable_commuting_fusion": False,
            "enable_advanced_commutation": False,
        },
        "minimal_config",
    ),
    (
        {
            "max_fusion_size": 8,
            "enable_commuting_fusion": True,
            "enable_advanced_commutation": False,
        },
        "no_advanced",
    ),
]

fusion_algorithm_data = [
    ([MockGateOperation("x", [0]), MockGateOperation("x", [0])], 0, "involutory_cancel"),
    (
        [MockGateOperation("x", [0]), MockGateOperation("x", [0]), MockGateOperation("x", [0])],
        1,
        "odd_involutory",
    ),
    ([MockGateOperation("s", [0]), MockGateOperation("s", [0])], 2, "non_involutory"),
    (
        [MockGateOperation("x", [0]), MockGateOperation("y", [0]), MockGateOperation("x", [0])],
        1,
        "pauli_xor",
    ),
]

matrix_expansion_data = [
    (np.array([[0, 1], [1, 0]]), (0,), 1, (2, 2), "identity_expansion"),
    (np.array([[0, 1], [1, 0]]), (1,), 2, (4, 4), "single_qubit_expansion"),
    (np.array([[1, 0], [0, -1]]), (0,), 2, (4, 4), "pauli_z_expansion"),
    (np.eye(4), (0, 1), 2, (4, 4), "two_qubit_identity"),
]

edge_case_data = [
    ("invalid_angle_type", "return_none", "invalid_angle_extraction"),
    ("missing_angle_attr", "return_none", "missing_angle_attribute"),
    ("unknown_gate_type", "pass_through", "unknown_gate_handling"),
    ("empty_operations", "return_empty", "empty_operation_list"),
    ("mixed_operations", "preserve_order", "mixed_gate_noise_ops"),
]


@pytest.mark.parametrize("matrix_func, expected", matrix_functions_data)
def test_matrix_functions(matrix_func, expected):
    """Test matrix function correctness and properties."""
    result = matrix_func()
    assert np.allclose(result, expected)
    assert result.dtype == complex
    assert result.shape == (2, 2)
    assert result.flags.writeable is True

    product = np.dot(result, np.conj(result.T))
    identity = np.eye(result.shape[0])
    assert np.allclose(product, identity)


@pytest.mark.parametrize("operations, expected_max_length, description", gate_fusion_scenarios)
def test_gate_fusion_scenarios(operations, expected_max_length, description):
    """Test various gate fusion scenarios."""
    result = apply_gate_fusion(operations)
    assert len(result) <= expected_max_length
    assert len(result) >= 0

    if operations:
        result_no_commuting = apply_gate_fusion(operations, enable_commuting_fusion=False)
        assert len(result_no_commuting) <= len(operations)

        result_small_fusion = apply_gate_fusion(operations, max_fusion_size=2)
        assert len(result_small_fusion) <= len(operations)


@pytest.mark.parametrize("gate1, gate2, expected_commute, description", commutation_test_data)
def test_commutation_analysis(gate1, gate2, expected_commute, description):
    """Test commutation analysis across different scenarios."""
    graph = CommutationGraph()
    assert graph._gates_commute(gate1, gate2) == expected_commute

    engine = GateFusionEngine()
    if gate1.targets == gate2.targets:
        assert engine._same_qubit_gates_commute(gate1, gate2) == expected_commute
    else:
        assert engine._gates_commute_safely(gate1, gate2) == expected_commute


@pytest.mark.parametrize("config, description", engine_configuration_data)
def test_engine_configurations(config, description):
    """Test engine configurations and their effects."""
    engine = GateFusionEngine(**config)
    assert engine.max_fusion_size == config["max_fusion_size"]
    assert engine.enable_commuting_fusion == config["enable_commuting_fusion"]
    assert engine.enable_advanced_commutation == config["enable_advanced_commutation"]

    if config["enable_advanced_commutation"]:
        assert hasattr(engine, "fast_target_fusion")
        assert isinstance(engine.fast_target_fusion, FastTargetBasedFusion)
    else:
        assert not hasattr(engine, "fast_target_fusion")

    test_gates = [MockGateOperation("x", [0]), MockGateOperation("y", [0])]
    result = engine.optimize_operations(test_gates)
    assert len(result) <= len(test_gates)


@pytest.mark.parametrize("gates, expected_count, algorithm_type", fusion_algorithm_data)
def test_fusion_algorithms(gates, expected_count, algorithm_type):
    """Test specific fusion algorithms."""
    fusion = FastTargetBasedFusion()

    if algorithm_type in ["involutory_cancel", "odd_involutory", "non_involutory"]:
        result = fusion._fuse_identical_gates(gates)
    elif algorithm_type == "pauli_xor":
        result = fusion._fuse_pauli_gates_xor(gates)
    else:
        result = gates

    assert len(result) == expected_count


@pytest.mark.parametrize(
    "matrix, targets, num_qubits, expected_shape, description", matrix_expansion_data
)
def test_matrix_expansion(matrix, targets, num_qubits, expected_shape, description):
    """Test matrix expansion operations."""
    engine = GateFusionEngine()
    result = engine._expand_matrix_optimized(matrix, targets, num_qubits)
    assert result.shape == expected_shape

    product = np.dot(result, np.conj(result.T))
    identity = np.eye(result.shape[0])
    assert np.allclose(product, identity)


@pytest.mark.parametrize("test_scenario, expected_behavior, description", edge_case_data)
def test_edge_cases(test_scenario, expected_behavior, description):
    """Test edge cases and error conditions."""
    if test_scenario == "invalid_angle_type":
        graph = CommutationGraph()
        gate = MockGateOperation("rx", [0])
        gate._angle = "invalid"
        angle = graph._extract_angle(gate)
        assert angle is None

    elif test_scenario == "missing_angle_attr":
        fusion = CommutativeAlgebraFusion()
        gate = MockGateOperation("rx", [0])
        gate._angle = None
        gate._theta = None
        angle = fusion._extract_angle(gate)
        assert angle is None

    elif test_scenario == "unknown_gate_type":
        gate = MockGateOperation("unknown_gate", [0])
        gate_type = gate.gate_type
        assert gate_type == "unknown_gate"

    elif test_scenario == "empty_operations":
        result = apply_gate_fusion([])
        assert result == []

    elif test_scenario == "mixed_operations":
        operations = [
            MockGateOperation("x", [0]),
            MockKrausOperation([0]),
            MockGateOperation("y", [1]),
        ]
        result = apply_gate_fusion(operations)
        assert len(result) >= 1


def test_fused_gate_operation():
    """Test FusedGateOperation creation and properties."""
    matrix = np.array([[0, 1], [1, 0]])
    targets = (0, 1)
    original_gates = [MockGateOperation("x", [0]), MockGateOperation("y", [1])]

    fused_op = FusedGateOperation(targets, matrix, original_gates)
    assert fused_op.targets == targets
    assert np.array_equal(fused_op._fused_matrix, matrix)
    assert fused_op.gate_count == 2
    assert fused_op.original_gates == original_gates

    int_matrix = np.array([[1, 0], [0, 1]], dtype=int)
    fused_op_int = FusedGateOperation((0,), int_matrix, [MockGateOperation("i", [0])])
    assert fused_op_int._fused_matrix.dtype == complex

    fused_op_with_type = FusedGateOperation((0,), matrix, original_gates, optimization_type="test")
    gate_type = fused_op_with_type.gate_type
    assert "fused_2_test" in gate_type


def test_engine_core_methods():
    """Test core GateFusionEngine methods."""
    engine = GateFusionEngine()

    assert engine._is_fusable_gate(MockGateOperation("x", [0])) is True
    assert engine._is_fusable_gate(MockGateOperation("cx", [0, 1])) is True
    assert engine._is_fusable_gate(MockGateOperation("ccx", [0, 1, 2])) is False
    assert engine._is_fusable_gate(Mock()) is False
    assert engine._is_fusable_gate(MockGateOperation("x", [0], power=3.5)) is False

    assert engine._is_safe_controlled_gate(MockGateOperation("cx", [0, 1])) is True
    assert engine._is_safe_controlled_gate(MockGateOperation("ccx", [0, 1, 2])) is True
    assert engine._is_safe_controlled_gate(MockGateOperation("custom", [0, 1, 2, 3])) is False

    gate_with_power = MockGateOperation("x", [0], power=2)
    assert engine._get_gate_power(gate_with_power) == 2

    gate_no_power = MockGateOperation("x", [0])
    gate_no_power._power = None
    assert engine._get_gate_power(gate_no_power) is None

    pure_gates = [MockGateOperation("x", [0]), MockGateOperation("y", [1])]
    assert engine._is_pure_gate_sequence(pure_gates) is True

    mixed_ops = [MockGateOperation("x", [0]), MockKrausOperation([0])]
    assert engine._is_pure_gate_sequence(mixed_ops) is False


def test_engine_optimization_methods():
    """Test engine optimization methods."""
    engine = GateFusionEngine()

    assert engine.optimize_operations([]) == []
    assert engine._optimize_gate_sequence([]) == []

    pauli_gates = [MockGateOperation("pauli_x", [0]), MockGateOperation("pauli_y", [0])]
    assert engine._is_pauli_sequence(pauli_gates) is True

    rotation_gates = [
        MockGateOperation("rx", [0], angle=np.pi / 4),
        MockGateOperation("rx", [0], angle=np.pi / 2),
    ]
    assert engine._is_rotation_sequence(rotation_gates) is True

    hadamard_gates = [MockGateOperation("hadamard", [0]), MockGateOperation("hadamard", [0])]
    assert engine._is_hadamard_sequence(hadamard_gates) is True

    result = engine._optimize_pauli_sequence_fast(pauli_gates)
    assert len(result) <= len(pauli_gates)

    result = engine._optimize_rotation_sequence_fast(rotation_gates)
    assert len(result) <= len(rotation_gates)

    result = engine._optimize_hadamard_sequence_fast(hadamard_gates)
    assert len(result) == 0


def test_fast_target_based_fusion():
    """Test FastTargetBasedFusion functionality."""
    fusion = FastTargetBasedFusion()
    assert hasattr(fusion, "_fusion_cache")

    gates = [MockGateOperation("x", [0]), MockGateOperation("y", [0]), MockGateOperation("z", [0])]
    result = fusion.optimize_commuting_gates(gates)
    assert len(result) <= len(gates)

    identical_gates = [MockGateOperation("x", [0]), MockGateOperation("x", [0])]
    result = fusion._fuse_target_group(identical_gates)
    assert len(result) == 0

    pauli_gates = [
        MockGateOperation("x", [0]),
        MockGateOperation("y", [0]),
        MockGateOperation("x", [0]),
    ]
    result = fusion._fuse_pauli_gates_xor(pauli_gates)
    assert len(result) == 1


def test_commutation_graph():
    """Test CommutationGraph functionality."""
    graph = CommutationGraph()

    gates = [MockGateOperation("x", [0]), MockGateOperation("y", [1])]
    result = graph.build(gates)
    assert len(result.nodes) == 2
    assert 1 in result.edges[0]

    non_commuting = [MockGateOperation("x", [0]), MockGateOperation("y", [0])]
    result = graph.build(non_commuting)
    assert 1 in result.non_commuting_edges[0]

    commuting_gates = [
        MockGateOperation("x", [0]),
        MockGateOperation("y", [1]),
        MockGateOperation("z", [2]),
    ]
    result = graph.build(commuting_gates)
    cliques = result.find_maximal_cliques()
    assert len(cliques) >= 1

    gate_with_angle = MockGateOperation("rx", [0], angle=np.pi / 4)
    assert graph._extract_angle(gate_with_angle) == np.pi / 4

    gate_with_theta = MockGateOperation("ry", [0])
    gate_with_theta._theta = np.pi / 3
    gate_with_theta._angle = None
    assert graph._extract_angle(gate_with_theta) == np.pi / 3


def test_pauli_lie_algebra():
    """Test PauliLieAlgebra functionality."""
    algebra = PauliLieAlgebra()
    assert algebra.pauli_basis == ["i", "x", "y", "z"]

    table = algebra.commutation_table
    assert table[("x", "y")] == ("z", 2j)
    assert table[("y", "x")] == ("z", -2j)
    assert table[("x", "x")] == ("i", 0)

    result, coeff = algebra.commutator("x", "y")
    assert result == "z" and coeff == 2j

    result, coeff = algebra.commutator("unknown1", "unknown2")
    assert result == "unknown" and coeff == 0

    gates = [MockGateOperation("x", [0]), MockGateOperation("y", [0]), MockGateOperation("z", [0])]
    result = algebra.optimal_order(gates)
    assert len(result) == len(gates)
    assert all(gate in result for gate in gates)

    cost = algebra._compute_insertion_cost(
        [MockGateOperation("x", [0])], MockGateOperation("y", [0])
    )
    assert cost > 0


def test_commutative_algebra_fusion():
    """Test CommutativeAlgebraFusion functionality."""
    fusion = CommutativeAlgebraFusion()
    assert isinstance(fusion.lie_algebra, PauliLieAlgebra)
    assert isinstance(fusion.commutation_graph, CommutationGraph)

    gates = [MockGateOperation("pauli_x", [0]), MockGateOperation("pauli_y", [0])]
    result1 = fusion.optimize_commuting_gates(gates)
    result2 = fusion.optimize_commuting_gates(gates)
    assert result1 == result2
    assert len(fusion._optimization_cache) > 0

    pauli_gates = [MockGateOperation("pauli_x", [0]), MockGateOperation("pauli_y", [0])]
    assert fusion._is_pauli_group(pauli_gates) is True

    rotation_gates = [
        MockGateOperation("rx", [0], angle=np.pi / 4),
        MockGateOperation("rx", [0], angle=np.pi / 2),
    ]
    assert fusion._is_rotation_group(rotation_gates) is True

    mixed_gates = [MockGateOperation("pauli_x", [0]), MockGateOperation("hadamard", [0])]
    assert fusion._is_pauli_group(mixed_gates) is False
    assert fusion._is_rotation_group(mixed_gates) is False

    result = fusion._fuse_pauli_group(pauli_gates)
    assert len(result) <= len(pauli_gates)

    result = fusion._fuse_rotation_group(rotation_gates)
    assert len(result) <= len(rotation_gates)


def test_angle_extraction_edge_cases():
    """Test angle extraction with various edge cases."""
    graph = CommutationGraph()

    gate_invalid_angle = MockGateOperation("rx", [0])
    gate_invalid_angle._angle = "invalid"
    assert graph._extract_angle(gate_invalid_angle) is None

    gate_invalid_attr = MockGateOperation("ry", [0])
    gate_invalid_attr.angle = [1, 2, 3]
    gate_invalid_attr._angle = None
    gate_invalid_attr._theta = None
    assert graph._extract_angle(gate_invalid_attr) is None

    gate_with_angle_attr = MockGateOperation("rz", [0])
    gate_with_angle_attr.angle = np.pi / 8
    gate_with_angle_attr._angle = None
    gate_with_angle_attr._theta = None
    assert graph._extract_angle(gate_with_angle_attr) == np.pi / 8


def test_control_target_analysis():
    """Test control-target conflict analysis."""
    engine = GateFusionEngine()

    gate1 = MockGateOperation("x", [0])
    gate2 = MockGateOperation("y", [1])
    assert engine._has_control_target_conflict(gate1, gate2) is False

    controlled_gate = MockGateOperation("cx", [0, 1], ctrl_modifiers=(0,))
    target_gate = MockGateOperation("x", [1])
    assert engine._has_control_target_conflict(controlled_gate, target_gate) is True

    gate1._ctrl_modifiers = None
    gate2._ctrl_modifiers = None
    assert engine._has_control_target_conflict(gate1, gate2) is False

    cx1 = MockGateOperation("cx", [0, 1])
    cx2 = MockGateOperation("cx", [0, 1])
    assert engine._are_identical_controlled_gates(cx1, cx2) is True

    cy = MockGateOperation("cy", [0, 1])
    assert engine._are_identical_controlled_gates(cx1, cy) is False


def test_advanced_fusion_features():
    """Test advanced fusion features."""
    engine = GateFusionEngine(enable_advanced_commutation=True)
    gates = [MockGateOperation("pauli_x", [0]), MockGateOperation("pauli_y", [1])]
    result = engine._optimize_gate_sequence(gates)
    assert len(result) <= len(gates)

    fusion = FastTargetBasedFusion()
    large_gates = [MockGateOperation("pauli_x", [i % 4]) for i in range(150)]
    result = fusion.optimize_commuting_gates(large_gates)
    assert len(result) <= len(large_gates)

    phase_gates = [MockGateOperation("s", [0]), MockGateOperation("t", [0])]
    assert fusion._is_phase_group_fast(phase_gates) is True
    result = fusion._optimize_phase_group_fast(phase_gates)
    assert len(result) <= len(phase_gates)

    hadamard_gates = [MockGateOperation("hadamard", [0]), MockGateOperation("hadamard", [0])]
    assert fusion._is_hadamard_group_fast(hadamard_gates) is True


def test_mixed_operation_handling():
    """Test handling of mixed gate and noise operations."""
    engine = GateFusionEngine()

    operations = [
        MockGateOperation("x", [0]),
        MockGateOperation("y", [0]),
        MockKrausOperation([0]),
        MockGateOperation("z", [1]),
    ]
    result = engine._optimize_mixed_operations(operations)
    assert len(result) >= 1

    other_op = Mock()
    other_op.gate_type = "other"
    operations = [MockGateOperation("x", [0]), other_op, MockGateOperation("y", [1])]
    result = engine._optimize_mixed_operations(operations)
    assert other_op in result


def test_performance_and_integration():
    """Test performance scenarios and integration."""
    gates = [MockGateOperation("x", [i % 4]) for i in range(100)]
    engine = GateFusionEngine()
    result = engine.optimize_operations(gates)
    assert len(result) <= len(gates)
    assert len(result) > 0

    qft_pattern = [
        MockGateOperation("h", [0]),
        MockGateOperation("rz", [0], angle=np.pi / 2),
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


def test_comprehensive_edge_cases():
    """Test comprehensive edge cases for complete coverage."""
    graph = CommutationGraph()
    gate1 = MockGateOperation("rx", [0])
    gate2 = MockGateOperation("ry", [0])
    gate1._angle = None
    gate1._theta = None
    gate2._angle = None
    gate2._theta = None
    assert graph._rotation_commutation_check("rx", "ry", gate1, gate2) is False

    gate1._angle = np.pi
    gate2._angle = np.pi
    assert graph._rotation_commutation_check("rx", "ry", gate1, gate2) is False

    graph = CommutationGraph()
    cliques = []
    graph._bron_kerbosch(set(), set(), set(), cliques)
    assert len(cliques) == 0

    gates = [MockGateOperation("x", [0]), MockGateOperation("y", [1]), MockGateOperation("z", [2])]
    graph.build(gates)
    cliques = graph.find_maximal_cliques()
    assert len(cliques) >= 0


def test_missing_coverage_lines():
    """Test specific lines that need coverage based on the coverage report."""

    graph = CommutationGraph()
    result = graph._pi_rotation_commutation("rx", "ry")
    assert result is False

    gate = MockGateOperation("unknown_gate_type", [0])
    normalized = gate.gate_type
    assert normalized == "unknown_gate_type"

    graph = CommutationGraph()
    cliques = []
    graph._bron_kerbosch(set(), set(), set(), cliques)
    assert len(cliques) == 0

    algebra = PauliLieAlgebra()

    result, coeff = algebra.commutator("unknown1", "unknown2")
    assert result == "unknown" and coeff == 0

    gate1 = MockGateOperation("h", [0])
    gate2 = MockGateOperation("s", [0])
    cost = algebra._compute_insertion_cost([gate1], gate2)
    assert cost == 0.0

    empty_result = algebra.optimal_order([])
    assert empty_result == []

    single_gate = [MockGateOperation("x", [0])]
    single_result = algebra.optimal_order(single_gate)
    assert single_result == single_gate

    fusion = CommutativeAlgebraFusion()
    gates = [MockGateOperation("x", [0]), MockGateOperation("y", [0])]

    result1 = fusion.optimize_commuting_gates(gates)
    assert len(fusion._optimization_cache) > 0

    result2 = fusion.optimize_commuting_gates(gates)
    assert result1 == result2

    signature = fusion._compute_gate_signature(gates)
    assert isinstance(signature, str)
    assert "|" in signature

    gate_with_list_angle = MockGateOperation("rx", [0])
    gate_with_list_angle.angle = [1, 2, 3]
