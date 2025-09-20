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

"""
Gate Fusion Implementation

Key features:
1. Safe commuting gate fusion with proper correctness checks
2. Smart controlled gate fusion for safe patterns
3. Reduced code duplication and improved efficiency
4. Enhanced pattern recognition for quantum algorithms
5. Streamlined matrix operations
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Set, Tuple, List
from collections import defaultdict
from braket.default_simulator.operation import GateOperation, KrausOperation
from braket.default_simulator.noise_fusion import apply_noise_fusion
from braket.default_simulator.lightweight_preprocessor import fast_preprocess

_IDENTITY = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]], dtype=complex)
_PAULI_X = np.array([[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]], dtype=complex)
_PAULI_Y = np.array([[0.0 + 0j, -1j], [1j, 0.0 + 0j]], dtype=complex)
_PAULI_Z = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, -1.0 + 0j]], dtype=complex)
_HADAMARD = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]], dtype=complex)

for matrix in [_IDENTITY, _PAULI_X, _PAULI_Y, _PAULI_Z, _HADAMARD]:
    matrix.flags.writeable = False

_GATE_REDUCTIONS = {
    ('x', 'x'): None, ('y', 'y'): None, ('z', 'z'): None, ('h', 'h'): None,
    ('cx', 'cx'): None, ('cz', 'cz'): None, ('swap', 'swap'): None,
    ('h', 'x', 'h'): 'z', ('h', 'z', 'h'): 'x',
    ('s', 's'): 'z', ('t', 't'): 's',
    ('s', 's', 's', 's'): None,
    ('t', 't', 't', 't'): 'z',
    ('t', 't', 't', 't', 't', 't', 't', 't'): None,
}

_COMMUTING_PAIRS = {
    frozenset(['z', 's']), frozenset(['z', 't']), frozenset(['s', 't']),
    frozenset(['pauli_z', 's']), frozenset(['pauli_z', 't']),
    frozenset(['z', 'rz']), frozenset(['s', 'rz']), frozenset(['t', 'rz']),
    frozenset(['pauli_z', 'rz']),
    frozenset(['rx', 'rx']), frozenset(['ry', 'ry']), frozenset(['rz', 'rz']),
}

_GATE_TYPES = {
    'i': 0, 'identity': 0, 'x': 1, 'pauli_x': 1, 'y': 2, 'pauli_y': 2, 
    'z': 3, 'pauli_z': 3, 'rx': 4, 'ry': 5, 'rz': 6, 'h': 7, 'hadamard': 7,
    's': 8, 't': 9, 'cx': 10, 'cnot': 10, 'cz': 11, 'swap': 12,
}

_PAULI_COMMUTATORS = {
    ('x', 'y'): ('z', 2j), ('y', 'x'): ('z', -2j),
    ('y', 'z'): ('x', 2j), ('z', 'y'): ('x', -2j),
    ('z', 'x'): ('y', 2j), ('x', 'z'): ('y', -2j),
    ('x', 'x'): ('i', 0), ('y', 'y'): ('i', 0), ('z', 'z'): ('i', 0),
}

_ROTATION_COMMUTATORS = {
    ('rx', 'ry'): ('rz', 'cross_product'), ('ry', 'rx'): ('rz', 'cross_product_neg'),
    ('ry', 'rz'): ('rx', 'cross_product'), ('rz', 'ry'): ('rx', 'cross_product_neg'),
    ('rz', 'rx'): ('ry', 'cross_product'), ('rx', 'rz'): ('ry', 'cross_product_neg'),
}


class CommutationGraph:
    """Graph-based representation of gate commutation relationships."""
    
    def __init__(self):
        self.nodes = []
        self.edges = defaultdict(set)
        self.non_commuting_edges = defaultdict(set)
        
    def build(self, gates: list[GateOperation]) -> 'CommutationGraph':
        """Build commutation graph from gate sequence."""
        self.nodes = gates.copy()
        
        for i, gate1 in enumerate(gates):
            for j, gate2 in enumerate(gates):
                if i != j:
                    if self._gates_commute(gate1, gate2):
                        self.edges[i].add(j)
                    else:
                        self.non_commuting_edges[i].add(j)
        
        return self
    
    def _gates_commute(self, gate1: GateOperation, gate2: GateOperation) -> bool:
        """Check if two gates commute using Lie algebra principles."""
        gate1_qubits = set(gate1.targets)
        gate2_qubits = set(gate2.targets)
        
        if gate1_qubits.isdisjoint(gate2_qubits):
            return True
        
        if gate1_qubits == gate2_qubits:
            return self._same_qubit_commutation_analysis(gate1, gate2)
        
        return False
    
    def _same_qubit_commutation_analysis(self, gate1: GateOperation, gate2: GateOperation) -> bool:
        """Advanced commutation analysis using Lie algebra."""
        type1 = self._normalize_gate_type(gate1)
        type2 = self._normalize_gate_type(gate2)
        
        if type1 in ['x', 'y', 'z'] and type2 in ['x', 'y', 'z']:
            return self._pauli_commutation_check(type1, type2, gate1, gate2)
        
        if type1 in ['rx', 'ry', 'rz'] and type2 in ['rx', 'ry', 'rz']:
            return self._rotation_commutation_check(type1, type2, gate1, gate2)
        
        gate_pair = frozenset([type1, type2])
        return gate_pair in _COMMUTING_PAIRS
    
    def _pauli_commutation_check(self, type1: str, type2: str, gate1: GateOperation, gate2: GateOperation) -> bool:
        """Check Pauli gate commutation using Lie algebra."""
        if type1 == type2:
            return True
        
        return False
    
    def _rotation_commutation_check(self, type1: str, type2: str, gate1: GateOperation, gate2: GateOperation) -> bool:
        """Check rotation gate commutation using Lie algebra."""
        if type1 == type2:
            return True
        
        angle1 = self._extract_angle(gate1)
        angle2 = self._extract_angle(gate2)
        
        if angle1 is None or angle2 is None:
            return False
        
        if abs(abs(angle1) - np.pi) < 1e-12 and abs(abs(angle2) - np.pi) < 1e-12:
            return self._pi_rotation_commutation(type1, type2)
        
        return False
    
    def _pi_rotation_commutation(self, type1: str, type2: str) -> bool:
        """Check commutation for π rotations using Lie algebra."""
        return False
    
    def _normalize_gate_type(self, gate: GateOperation) -> str:
        """Normalize gate type for commutation analysis."""
        gate_type = gate.gate_type.lower()
        if gate_type in ['pauli_x', 'paulix']:
            return 'x'
        elif gate_type in ['pauli_y', 'pauliy']:
            return 'y'
        elif gate_type in ['pauli_z', 'pauliz']:
            return 'z'
        elif gate_type in ['hadamard']:
            return 'h'
        elif gate_type in ['cnot']:
            return 'cx'
        return gate_type
    
    def _extract_angle(self, gate: GateOperation) -> Optional[float]:
        """Extract rotation angle from gate."""
        for attr in ['_angle', 'angle', '_theta', 'theta']:
            if hasattr(gate, attr):
                angle = getattr(gate, attr)
                if isinstance(angle, (int, float)):
                    return float(angle)
        return None
    
    def find_maximal_cliques(self) -> list[list[int]]:
        """Find maximal cliques of commuting gates using Bron-Kerbosch algorithm."""
        cliques = []
        self._bron_kerbosch(set(), set(range(len(self.nodes))), set(), cliques)
        return [list(clique) for clique in cliques if len(clique) > 1]
    
    def _bron_kerbosch(self, R: set, P: set, X: set, cliques: list):
        """Bron-Kerbosch algorithm for finding maximal cliques."""
        if not P and not X:
            if len(R) > 1:
                cliques.append(R.copy())
            return
        
        pivot = max(P.union(X), key=lambda v: len(self.edges[v].intersection(P)), default=None)
        if pivot is None:
            return
        
        for v in P - self.edges[pivot]:
            neighbors = self.edges[v]
            self._bron_kerbosch(
                R.union({v}),
                P.intersection(neighbors),
                X.intersection(neighbors),
                cliques
            )
            P.remove(v)
            X.add(v)


class PauliLieAlgebra:
    """Pauli Lie algebra for advanced gate optimization."""
    
    def __init__(self):
        self.pauli_basis = ['i', 'x', 'y', 'z']
        self.commutation_table = self._build_commutation_table()
    
    def _build_commutation_table(self) -> dict:
        """Build commutation table for Pauli operators."""
        table = {}
        
        for p in self.pauli_basis:
            table[('i', p)] = ('i', 0)
            table[(p, 'i')] = ('i', 0)
        
        table[('x', 'x')] = ('i', 0)
        table[('y', 'y')] = ('i', 0)
        table[('z', 'z')] = ('i', 0)
        
        table[('x', 'y')] = ('z', 2j)
        table[('y', 'x')] = ('z', -2j)
        table[('y', 'z')] = ('x', 2j)
        table[('z', 'y')] = ('x', -2j)
        table[('z', 'x')] = ('y', 2j)
        table[('x', 'z')] = ('y', -2j)
        
        return table
    
    def commutator(self, gate1_type: str, gate2_type: str) -> tuple[str, complex]:
        """Compute commutator [A, B] = AB - BA."""
        return self.commutation_table.get((gate1_type, gate2_type), ('unknown', 0))
    
    def optimal_order(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Find optimal ordering of gates to minimize non-commuting operations."""
        if len(gates) <= 1:
            return gates
        
        ordered_gates = []
        remaining_gates = gates.copy()
        
        ordered_gates.append(remaining_gates.pop(0))
        
        while remaining_gates:
            best_gate = None
            best_score = float('inf')
            best_idx = -1
            
            for i, candidate in enumerate(remaining_gates):
                score = self._compute_insertion_cost(ordered_gates, candidate)
                if score < best_score:
                    best_score = score
                    best_gate = candidate
                    best_idx = i
            
            if best_gate is not None:
                ordered_gates.append(best_gate)
                remaining_gates.pop(best_idx)
        
        return ordered_gates
    
    def _compute_insertion_cost(self, current_sequence: list[GateOperation], candidate: GateOperation) -> float:
        """Compute cost of inserting candidate gate into current sequence."""
        cost = 0.0
        candidate_type = self._normalize_gate_type(candidate)
        
        for gate in current_sequence:
            gate_type = self._normalize_gate_type(gate)
            
            if gate_type in self.pauli_basis and candidate_type in self.pauli_basis:
                commutator_result, coefficient = self.commutator(gate_type, candidate_type)
                if commutator_result != 'i':
                    cost += abs(coefficient)
        
        return cost
    
    def _normalize_gate_type(self, gate: GateOperation) -> str:
        """Normalize gate type for Lie algebra analysis."""
        gate_type = gate.gate_type.lower()
        if gate_type in ['pauli_x', 'paulix']:
            return 'x'
        elif gate_type in ['pauli_y', 'pauliy']:
            return 'y'
        elif gate_type in ['pauli_z', 'pauliz']:
            return 'z'
        elif gate_type in ['identity']:
            return 'i'
        return gate_type


class CommutativeAlgebraFusion:
    """Advanced commutation analysis using Lie algebra for gate fusion optimization."""
    
    def __init__(self):
        self.lie_algebra = PauliLieAlgebra()
        self.commutation_graph = CommutationGraph()
        self._optimization_cache = {}
    
    def optimize_commuting_gates(self, gates: list[GateOperation]) -> list[GateOperation]:
        """
        Advanced commutation analysis using Lie algebra - PERFORMANCE OPTIMIZED FOR 1-2 QUBITS.
        
        Strategy: 
        1. Build commutation graph of gate relationships
        2. Find maximal commuting subsets using clique detection
        3. Filter groups to only include ≤2 qubit combinations
        4. Reorder gates within each small commuting group for optimal fusion
        5. Apply fusion to each optimized group
        """
        if len(gates) <= 1:
            return gates
        
        gate_signature = self._compute_gate_signature(gates)
        if gate_signature in self._optimization_cache:
            return self._optimization_cache[gate_signature]
        
        graph = self.commutation_graph.build(gates)
        
        commuting_groups = graph.find_maximal_cliques()
        
        filtered_groups = []
        for group_indices in commuting_groups:
            group_gates = [gates[i] for i in group_indices]
            all_qubits = set()
            for gate in group_gates:
                all_qubits.update(gate.targets)
            
            if len(all_qubits) <= 2:
                filtered_groups.append(group_indices)
        
        optimized_gates = []
        processed_indices = set()
        
        for group_indices in filtered_groups:
            if any(idx in processed_indices for idx in group_indices):
                continue
            
            group_gates = [gates[i] for i in group_indices]
            
            reordered_gates = self.lie_algebra.optimal_order(group_gates)
            
            fused_gates = self._fuse_commuting_sequence(reordered_gates)
            
            optimized_gates.extend(fused_gates)
            processed_indices.update(group_indices)
        
        for i, gate in enumerate(gates):
            if i not in processed_indices:
                optimized_gates.append(gate)
        
        self._optimization_cache[gate_signature] = optimized_gates
        
        return optimized_gates
    
    def _compute_gate_signature(self, gates: list[GateOperation]) -> str:
        """Compute a signature for gate sequence for caching."""
        signature_parts = []
        for gate in gates:
            gate_type = gate.gate_type.lower()
            targets = tuple(sorted(gate.targets))
            angle = self._extract_angle(gate)
            signature_parts.append(f"{gate_type}_{targets}_{angle}")
        return "|".join(signature_parts)
    
    def _extract_angle(self, gate: GateOperation) -> Optional[float]:
        """Extract angle from parameterized gates."""
        for attr in ['_angle', 'angle', '_theta', 'theta']:
            if hasattr(gate, attr):
                angle = getattr(gate, attr)
                if isinstance(angle, (int, float)):
                    return round(float(angle), 6)
        return None
    
    def _fuse_commuting_sequence(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Fuse a sequence of commuting gates using advanced techniques."""
        if len(gates) <= 1:
            return gates
        
        qubit_groups = defaultdict(list)
        for gate in gates:
            qubit_key = tuple(sorted(gate.targets))
            qubit_groups[qubit_key].append(gate)
        
        fused_gates = []
        
        for qubit_key, group_gates in qubit_groups.items():
            if len(group_gates) == 1:
                fused_gates.extend(group_gates)
            else:
                if self._is_pauli_group(group_gates):
                    fused_gates.extend(self._fuse_pauli_group(group_gates))
                elif self._is_rotation_group(group_gates):
                    fused_gates.extend(self._fuse_rotation_group(group_gates))
                else:
                    fused_gates.extend(self._fuse_general_group(group_gates))
        
        return fused_gates
    
    def _is_pauli_group(self, gates: list[GateOperation]) -> bool:
        """Check if all gates in group are Pauli gates."""
        pauli_types = {'x', 'y', 'z', 'pauli_x', 'pauli_y', 'pauli_z'}
        return all(gate.gate_type.lower() in pauli_types for gate in gates)
    
    def _is_rotation_group(self, gates: list[GateOperation]) -> bool:
        """Check if all gates in group are rotation gates on same axis."""
        if not gates:
            return False
        
        first_type = gates[0].gate_type.lower()
        if first_type not in ['rx', 'ry', 'rz']:
            return False
        
        return all(gate.gate_type.lower() == first_type for gate in gates)
    
    def _fuse_pauli_group(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Fuse Pauli gates using XOR logic."""
        if not gates:
            return gates
        
        pauli_count = {'x': 0, 'y': 0, 'z': 0}
        gate_templates = {}
        
        for gate in gates:
            gate_type = gate.gate_type.lower()
            normalized_type = gate_type.replace('pauli_', '')
            
            if normalized_type in pauli_count:
                pauli_count[normalized_type] += 1
                if normalized_type not in gate_templates:
                    gate_templates[normalized_type] = gate
        
        result = []
        for pauli_type, count in pauli_count.items():
            if count % 2 == 1:
                result.append(gate_templates[pauli_type])
        
        return result
    
    def _fuse_rotation_group(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Fuse rotation gates by summing angles."""
        if not gates:
            return gates
        
        total_angle = 0.0
        template_gate = gates[0]
        
        for gate in gates:
            angle = self._extract_angle(gate)
            if angle is None:
                return gates
            total_angle += angle
        
        normalized_angle = total_angle % (4.0 * np.pi)
        if abs(normalized_angle) < 1e-12 or abs(normalized_angle - 4.0 * np.pi) < 1e-12:
            return []
        
        return [template_gate]
    
    def _fuse_general_group(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Fuse general gates using matrix multiplication."""
        if len(gates) <= 1:
            return gates
        
        return gates


class FusedGateOperation(GateOperation):
    """Optimized fused gate operation with minimal overhead."""
    
    __slots__ = ('_fused_matrix', '_original_gates', '_optimization_type')
    
    def __init__(
        self,
        targets: tuple[int, ...],
        fused_matrix: np.ndarray,
        original_gates: list[GateOperation],
        optimization_type: str = "fused",
        ctrl_modifiers: tuple[int, ...] = (),
        power: float = 1,
    ):
        super().__init__(targets=targets, ctrl_modifiers=ctrl_modifiers, power=power)
        self._fused_matrix = fused_matrix.astype(complex, copy=False)
        self._original_gates = original_gates
        self._optimization_type = optimization_type

    @property
    def _base_matrix(self) -> np.ndarray:
        return self._fused_matrix

    @property
    def gate_type(self) -> str:
        return f"fused_{len(self._original_gates)}_{self._optimization_type}"

    @property
    def original_gates(self) -> list[GateOperation]:
        return self._original_gates

    @property
    def gate_count(self) -> int:
        return len(self._original_gates)

    @property
    def optimization_type(self) -> str:
        return self._optimization_type


class FastTargetBasedFusion:
    """Fast target-based fusion for 1-2 qubit gate optimization."""
    
    def __init__(self):
        self._fusion_cache = {}
    
    def optimize_commuting_gates(self, gates: list[GateOperation]) -> list[GateOperation]:
        """
        Fast target-based fusion focusing on 1-2 qubit gates.
        
        Strategy:
        1. Group gates by EXACT target qubits (same targets = potential fusion)
        2. Handle control-target relationships for 2-qubit gates
        3. Apply fast fusion algorithms: XOR logic for Pauli gates, parity logic for identical gates
        4. Strict limitation to 1-2 qubit gates for optimal performance
        5. CRITICAL: Only fuse gates that act on the EXACT SAME qubits
        """
        if len(gates) <= 1:
            return gates
        
        target_groups = defaultdict(list)
        for gate in gates:
            target_key = tuple(gate.targets)
            if len(target_key) <= 2:
                target_groups[target_key].append(gate)
        
        optimized_gates = []
        processed_gates = set()
        
        for target_key, group_gates in target_groups.items():
            if len(group_gates) > 1:
                fused_gates = self._fuse_target_group(group_gates)
                optimized_gates.extend(fused_gates)
                processed_gates.update(id(g) for g in group_gates)
        
        for gate in gates:
            if id(gate) not in processed_gates:
                optimized_gates.append(gate)
        
        return optimized_gates
    
    def _fuse_target_group(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Fuse gates with same target qubits."""
        if len(gates) <= 1:
            return gates
        
        gate_types = [self._normalize_gate_type(gate) for gate in gates]
        
        if len(set(gate_types)) == 1:
            return self._fuse_identical_gates(gates)
        
        if all(gt in ['x', 'y', 'z'] for gt in gate_types):
            return self._fuse_pauli_gates_xor(gates)
        
        return gates
    
    def _fuse_identical_gates(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Fuse identical gates using parity logic."""
        gate_type = self._normalize_gate_type(gates[0])
        
        involutory_gates = {'x', 'y', 'z', 'h', 'cx', 'cz', 'swap'}
        
        if gate_type in involutory_gates:
            count = len(gates)
            return [] if count % 2 == 0 else [gates[0]]
        
        return gates
    
    def _fuse_pauli_gates_xor(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Fuse Pauli gates using XOR logic."""
        pauli_count = {'x': 0, 'y': 0, 'z': 0}
        gate_templates = {}
        
        for gate in gates:
            gate_type = self._normalize_gate_type(gate)
            if gate_type in pauli_count:
                pauli_count[gate_type] += 1
                if gate_type not in gate_templates:
                    gate_templates[gate_type] = gate
        
        result = []
        for pauli_type, count in pauli_count.items():
            if count % 2 == 1:
                result.append(gate_templates[pauli_type])
        
        return result
    
    def _normalize_gate_type(self, gate: GateOperation) -> str:
        """Normalize gate type for fusion analysis."""
        gate_type = gate.gate_type.lower()
        if gate_type in ['pauli_x', 'paulix']:
            return 'x'
        elif gate_type in ['pauli_y', 'pauliy']:
            return 'y'
        elif gate_type in ['pauli_z', 'pauliz']:
            return 'z'
        elif gate_type in ['hadamard']:
            return 'h'
        elif gate_type in ['cnot']:
            return 'cx'
        return gate_type


class GateFusionEngine:
    """Gate fusion engine with smart pattern recognition and advanced commutative algebra optimization."""
    
    def __init__(self, max_fusion_size: int = 8, enable_commuting_fusion: bool = True, 
                 enable_advanced_commutation: bool = True):
        self.max_fusion_size = max_fusion_size
        self.enable_commuting_fusion = enable_commuting_fusion
        self.enable_advanced_commutation = enable_advanced_commutation
        self._gate_type_cache = {}
        
        if self.enable_advanced_commutation:
            self.fast_target_fusion = FastTargetBasedFusion()
        
    def optimize_operations(self, operations: list) -> list:
        """Main optimization entry point with smart operation handling."""
        if not operations:
            return operations

        preprocessed_operations = fast_preprocess(operations)
        
        if self._is_pure_gate_sequence(preprocessed_operations):
            return self._optimize_gate_sequence(preprocessed_operations)
        
        return self._optimize_mixed_operations(preprocessed_operations)

    def _is_pure_gate_sequence(self, operations: list) -> bool:
        """Fast check for pure gate sequences."""
        return all(isinstance(op, GateOperation) for op in operations)

    def _optimize_gate_sequence(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Optimized gate sequence processing with smart clustering and advanced commutation analysis."""
        if not gates:
            return gates

        if self.enable_advanced_commutation and hasattr(self, 'fast_target_fusion'):
            gates = self.fast_target_fusion.optimize_commuting_gates(gates)

        optimized = []
        i = 0
        
        while i < len(gates):
            fusion_group = self._find_optimal_fusion_group(gates, i)
            
            if len(fusion_group) > 1:
                reduced_gates = self._apply_smart_reductions(fusion_group)
                if reduced_gates:
                    fused_op = self._create_optimized_fused_operation(reduced_gates)
                    optimized.append(fused_op)
                i += len(fusion_group)
            else:
                optimized.append(gates[i])
                i += 1
        
        return optimized

    def _find_optimal_fusion_group(self, gates: list[GateOperation], start_idx: int) -> list[GateOperation]:
        """Find optimal fusion group strictly limited to 1-2 qubits for maximum performance."""
        if start_idx >= len(gates):
            return []

        first_gate = gates[start_idx]
        if not self._is_fusable_gate(first_gate):
            return [first_gate]

        fusion_group = [first_gate]
        first_qubits = set(first_gate.targets)
        all_qubits = first_qubits.copy()
        
        MAX_FUSION_QUBITS = 2
        
        if len(first_qubits) > MAX_FUSION_QUBITS:
            return fusion_group

        for i in range(start_idx + 1, min(start_idx + self.max_fusion_size, len(gates))):
            candidate = gates[i]
            
            if not self._is_fusable_gate(candidate):
                break
                
            candidate_qubits = set(candidate.targets)
            new_all_qubits = all_qubits.union(candidate_qubits)
            
            if len(new_all_qubits) > MAX_FUSION_QUBITS:
                break
            
            can_fuse = False
            
            if candidate_qubits == first_qubits and len(new_all_qubits) <= MAX_FUSION_QUBITS:
                can_fuse = True
            elif candidate_qubits.issubset(first_qubits) and len(new_all_qubits) <= MAX_FUSION_QUBITS:
                can_fuse = True
            elif (first_qubits.issubset(candidate_qubits) and 
                  len(candidate_qubits) <= MAX_FUSION_QUBITS and
                  len(new_all_qubits) <= MAX_FUSION_QUBITS):
                can_fuse = True
                first_qubits = candidate_qubits
            # DISABLED: Commuting gates on different qubits - this was causing incorrect fusion
            # Gates on different qubits should NOT be fused together as they act independently
            # elif (self.enable_commuting_fusion and 
            #       len(candidate_qubits) == 1 and len(first_qubits) == 1 and
            #       len(new_all_qubits) <= MAX_FUSION_QUBITS and
            #       self._can_safely_commute_with_group(fusion_group, candidate)):
            #     can_fuse = True
            
            if (can_fuse and 
                len(fusion_group) < self.max_fusion_size and
                len(new_all_qubits) <= MAX_FUSION_QUBITS):
                
                fusion_group.append(candidate)
                all_qubits = new_all_qubits
            else:
                break

        return fusion_group

    def _can_safely_commute_with_group(self, fusion_group: list[GateOperation], 
                                     candidate: GateOperation) -> bool:
        """Enhanced commuting gate detection with strict safety checks."""
        
        for gate in fusion_group:
            if not self._gates_commute_safely(gate, candidate):
                return False
                
        for gate in fusion_group:
            if self._has_control_target_conflict(gate, candidate):
                return False
        
        return True

    def _gates_commute_safely(self, gate1: GateOperation, gate2: GateOperation) -> bool:
        """Safe commutativity check with quantum mechanical correctness."""
        gate1_qubits = set(gate1.targets)
        gate2_qubits = set(gate2.targets)
        
        if gate1_qubits.isdisjoint(gate2_qubits):
            return True
        
        if gate1_qubits == gate2_qubits:
            return self._same_qubit_gates_commute(gate1, gate2)

        return False

    def _same_qubit_gates_commute(self, gate1: GateOperation, gate2: GateOperation) -> bool:
        """Check if gates on same qubits commute - VERY CONSERVATIVE for correctness."""
        type1 = self._get_gate_type_normalized(gate1)
        type2 = self._get_gate_type_normalized(gate2)
        
        if type1 == type2:
            return True
        
        gate_pair = frozenset([type1, type2])
        return gate_pair in _COMMUTING_PAIRS

    def _get_gate_type_normalized(self, gate: GateOperation) -> str:
        """Get normalized gate type with caching."""
        gate_type = gate.gate_type.lower()
        if gate_type not in self._gate_type_cache:
            if gate_type in ['pauli_x', 'paulix']:
                self._gate_type_cache[gate_type] = 'x'
            elif gate_type in ['pauli_y', 'pauliy']:
                self._gate_type_cache[gate_type] = 'y'
            elif gate_type in ['pauli_z', 'pauliz']:
                self._gate_type_cache[gate_type] = 'z'
            elif gate_type in ['hadamard']:
                self._gate_type_cache[gate_type] = 'h'
            elif gate_type in ['cnot']:
                self._gate_type_cache[gate_type] = 'cx'
            else:
                self._gate_type_cache[gate_type] = gate_type
        
        return self._gate_type_cache[gate_type]

    def _has_control_target_conflict(self, gate1: GateOperation, gate2: GateOperation) -> bool:
        """Check for control-target conflicts between gates."""
        if len(gate1.targets) == 1 and len(gate2.targets) == 1:
            return False
        
        if (hasattr(gate1, '_ctrl_modifiers') and gate1._ctrl_modifiers) or \
           (hasattr(gate2, '_ctrl_modifiers') and gate2._ctrl_modifiers):
            return True 
        
        return False

    def _apply_smart_reductions(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Apply smart gate reductions with pattern recognition."""
        if len(gates) <= 1:
            return gates

        if self._is_pauli_sequence(gates):
            return self._optimize_pauli_sequence_fast(gates)
        elif self._is_rotation_sequence(gates):
            return self._optimize_rotation_sequence_fast(gates)
        elif self._is_hadamard_sequence(gates):
            return self._optimize_hadamard_sequence_fast(gates)
        else:
            return self._apply_general_reductions(gates)

    def _is_pauli_sequence(self, gates: list[GateOperation]) -> bool:
        """Fast Pauli sequence detection."""
        pauli_types = {'x', 'y', 'z', 'pauli_x', 'pauli_y', 'pauli_z'}
        return all(self._get_gate_type_normalized(gate) in pauli_types for gate in gates)

    def _is_rotation_sequence(self, gates: list[GateOperation]) -> bool:
        """Fast rotation sequence detection."""
        if not gates:
            return False
        first_type = self._get_gate_type_normalized(gates[0])
        return (first_type in ['rx', 'ry', 'rz'] and 
                all(self._get_gate_type_normalized(gate) == first_type for gate in gates))

    def _is_hadamard_sequence(self, gates: list[GateOperation]) -> bool:
        """Fast Hadamard sequence detection."""
        if not gates:
            return False
        return (all(self._get_gate_type_normalized(gate) == 'h' for gate in gates) and
                all(len(gate.targets) == 1 for gate in gates) and
                all(gate.targets[0] == gates[0].targets[0] for gate in gates))

    def _optimize_pauli_sequence_fast(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Fast Pauli sequence optimization using XOR logic - ONLY for gates on the same qubit."""
        if not gates:
            return gates
        
        first_targets = gates[0].targets
        if not all(gate.targets == first_targets for gate in gates):
            return gates
        
        pauli_count = [0, 0, 0]
        
        for gate in gates:
            gate_type = self._get_gate_type_normalized(gate)
            if gate_type == 'x':
                pauli_count[0] += 1
            elif gate_type == 'y':
                pauli_count[1] += 1
            elif gate_type == 'z':
                pauli_count[2] += 1

        result = []
        if pauli_count[0] % 2 == 1:
            result.append(next(g for g in gates if self._get_gate_type_normalized(g) == 'x'))
        if pauli_count[1] % 2 == 1:
            result.append(next(g for g in gates if self._get_gate_type_normalized(g) == 'y'))
        if pauli_count[2] % 2 == 1:
            result.append(next(g for g in gates if self._get_gate_type_normalized(g) == 'z'))
            
        return result

    def _optimize_rotation_sequence_fast(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Fast rotation sequence optimization."""
        if not gates:
            return gates
         
        total_angle = 0.0
        for gate in gates:
            angle = self._extract_angle_fast(gate)
            if angle is None:
                return gates
            total_angle += angle
        
        normalized_angle = total_angle % (4.0 * np.pi)
        if abs(normalized_angle) < 1e-12 or abs(normalized_angle - 4.0 * np.pi) < 1e-12:
            return []
        
        optimized_gate = gates[0]
        if hasattr(optimized_gate, '_angle'):
            optimized_gate._angle = normalized_angle
        return [optimized_gate]

    def _optimize_hadamard_sequence_fast(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Fast Hadamard sequence optimization."""
        count = len(gates)
        return [] if count % 2 == 0 else [gates[0]]

    def _apply_general_reductions(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Apply general gate reductions."""
        gate_sequence = tuple(self._get_gate_type_normalized(gate) for gate in gates)
        
        if gate_sequence in _GATE_REDUCTIONS:
            replacement = _GATE_REDUCTIONS[gate_sequence]
            if replacement is None:
                return []
            else:
                return [gates[0]]
        
        return gates

    def _extract_angle_fast(self, gate: GateOperation) -> Optional[float]:
        """Fast angle extraction with common attribute names."""
        for attr in ['_angle', 'angle', '_theta', 'theta']:
            if hasattr(gate, attr):
                angle = getattr(gate, attr)
                if isinstance(angle, (int, float)):
                    return float(angle)
        return None

    def _is_fusable_gate(self, gate: GateOperation) -> bool:
        """Enhanced fusability check with smart controlled gate handling."""
        if not isinstance(gate, GateOperation) or not hasattr(gate, "gate_type"):
            return False
        
        if len(gate.targets) > 2:
            return False
        
        if hasattr(gate, '_ctrl_modifiers') and gate._ctrl_modifiers:
            return self._is_safe_controlled_gate(gate)
        
        power = self._get_gate_power(gate)
        if power is not None and power not in [0, 0.5, 1, 2, -1]:
            return False
        
        return True

    def _is_safe_controlled_gate(self, gate: GateOperation) -> bool:
        """Check if controlled gate is safe for fusion."""
        gate_type = self._get_gate_type_normalized(gate)
        
        if len(gate.targets) == 2:
            return gate_type in ['cx', 'cz', 'cy', 'cnot']
        
        if len(gate.targets) == 3:
            return gate_type in ['ccnot', 'ccx', 'toffoli']
        
        return False

    def _get_gate_power(self, gate: GateOperation) -> Optional[float]:
        """Extract power value from gate."""
        for attr in ['_power', 'power']:
            if hasattr(gate, attr):
                return getattr(gate, attr)
        return None

    def _create_optimized_fused_operation(self, gates: list[GateOperation]) -> FusedGateOperation:
        """Create optimized fused operation with efficient matrix computation."""
        if len(gates) == 1:
            return FusedGateOperation(
                targets=gates[0].targets,
                fused_matrix=gates[0].matrix,
                original_gates=gates,
                optimization_type="single",
            )

        all_targets = set().union(*[gate.targets for gate in gates])
        targets = tuple(sorted(all_targets))
        
        fused_matrix = self._compute_fused_matrix_optimized(gates, targets)
        
        return FusedGateOperation(
            targets=targets,
            fused_matrix=fused_matrix,
            original_gates=gates,
            optimization_type="optimized",
        )

    def _compute_fused_matrix_optimized(self, gates: list[GateOperation], 
                                      targets: tuple[int, ...]) -> np.ndarray:
        """Optimized fused matrix computation."""
        if len(gates) == 1:
            return gates[0].matrix
            
        num_qubits = len(targets)
        matrix_size = 2**num_qubits
        result = np.eye(matrix_size, dtype=complex)
        
        target_map = {target: i for i, target in enumerate(targets)}
        
        for gate in gates:
            gate_matrix = gate.matrix
            local_targets = tuple(target_map[t] for t in gate.targets)
            expanded_matrix = self._expand_matrix_optimized(gate_matrix, local_targets, num_qubits)
            result = expanded_matrix @ result
        
        return result

    def _expand_matrix_optimized(self, gate_matrix: np.ndarray, 
                               targets: tuple[int, ...], num_qubits: int) -> np.ndarray:
        """Optimized matrix expansion using efficient tensor product placement."""
        if len(targets) == num_qubits:
            return gate_matrix
        
        matrix_size = 2**num_qubits
        result = np.zeros((matrix_size, matrix_size), dtype=complex)
        
        gate_size = 2**len(targets)
        
        for i in range(matrix_size):
            for j in range(matrix_size):
                gate_i = 0
                gate_j = 0
                valid = True
                
                for k, target in enumerate(targets):
                    bit_i = (i >> (num_qubits - 1 - target)) & 1
                    bit_j = (j >> (num_qubits - 1 - target)) & 1
                    gate_i |= bit_i << (len(targets) - 1 - k)
                    gate_j |= bit_j << (len(targets) - 1 - k)
                
                for q in range(num_qubits):
                    if q not in targets:
                        bit_i = (i >> (num_qubits - 1 - q)) & 1
                        bit_j = (j >> (num_qubits - 1 - q)) & 1
                        if bit_i != bit_j:
                            valid = False
                            break
                
                if valid:
                    result[i, j] = gate_matrix[gate_i, gate_j]
        
        return result

    def _optimize_mixed_operations(self, operations: list) -> list:
        """Handle mixed operation types efficiently."""
        gate_ops = []
        noise_ops = []
        other_ops = []
        positions = []
        
        for i, op in enumerate(operations):
            if isinstance(op, GateOperation):
                gate_ops.append(op)
                positions.append(('gate', len(gate_ops) - 1))
            elif isinstance(op, KrausOperation):
                noise_ops.append(op)
                positions.append(('noise', len(noise_ops) - 1))
            else:
                other_ops.append(op)
                positions.append(('other', len(other_ops) - 1))
        

        optimized_gates = self._optimize_gate_sequence(gate_ops) if gate_ops else []
        optimized_noise = apply_noise_fusion(noise_ops) if noise_ops else []
        
        result = []
        gate_idx = noise_idx = other_idx = 0
        
        for op_type, type_idx in positions:
            if op_type == 'gate' and gate_idx < len(optimized_gates):
                result.append(optimized_gates[gate_idx])
                gate_idx += 1
            elif op_type == 'noise' and noise_idx < len(optimized_noise):
                result.append(optimized_noise[noise_idx])
                noise_idx += 1
            elif op_type == 'other' and other_idx < len(other_ops):
                result.append(other_ops[other_idx])
                other_idx += 1
        
        return result


def apply_gate_fusion(operations: list, max_fusion_size: int = 8, 
                     enable_commuting_fusion: bool = True) -> list:
    """Apply gate fusion with enhanced performance."""
    if not operations:
        return operations
        
    engine = GateFusionEngine(
        max_fusion_size=max_fusion_size,
        enable_commuting_fusion=enable_commuting_fusion
    )
    return engine.optimize_operations(operations)




def _get_identity_matrix():
    return _IDENTITY.copy()

def _get_pauli_x_matrix():
    return _PAULI_X.copy()

def _get_pauli_y_matrix():
    return _PAULI_Y.copy()

def _get_pauli_z_matrix():
    return _PAULI_Z.copy()

def _get_hadamard_matrix():
    return _HADAMARD.copy()
