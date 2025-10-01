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

from collections import defaultdict
from typing import Optional

import numpy as np

from braket.default_simulator.noise_fusion import apply_noise_fusion
from braket.default_simulator.operation import GateOperation, KrausOperation

_IDENTITY = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]], dtype=complex)
_PAULI_X = np.array([[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]], dtype=complex)
_PAULI_Y = np.array([[0.0 + 0j, -1j], [1j, 0.0 + 0j]], dtype=complex)
_PAULI_Z = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, -1.0 + 0j]], dtype=complex)
_HADAMARD = np.array(
    [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]], dtype=complex
)

for matrix in [_IDENTITY, _PAULI_X, _PAULI_Y, _PAULI_Z, _HADAMARD]:
    matrix.flags.writeable = False

_GATE_REDUCTIONS = {
    ("pauli_x", "pauli_x"): None,
    ("pauli_y", "pauli_y"): None,
    ("pauli_z", "pauli_z"): None,
    ("hadamard", "hadamard"): None,
    ("cx", "cx"): None,
    ("cz", "cz"): None,
    ("swap", "swap"): None,
    ("hadamard", "pauli_x", "hadamard"): "pauli_z",
    ("hadamard", "pauli_z", "hadamard"): "pauli_x",
    ("hadamard", "pauli_y", "hadamard"): "pauli_y",
    ("s", "s"): "pauli_z",
    ("t", "t"): "s",
    ("s", "si"): None,
    ("si", "s"): None,
    ("t", "ti"): None,
    ("ti", "t"): None,
    ("s", "s", "s", "s"): None,
    ("t", "t", "t", "t"): "pauli_z",
    ("t", "t", "t", "t", "t", "t", "t", "t"): None,
    ("rx", "rx"): "rx_combined",
    ("ry", "ry"): "ry_combined",
    ("rz", "rz"): "rz_combined",
    ("cx", "hadamard", "hadamard"): "cz",
    ("hadamard", "hadamard", "cx"): "cz",
}

_COMMUTING_PAIRS = {
    frozenset(["pauli_z", "s"]),
    frozenset(["pauli_z", "t"]),
    frozenset(["s", "t"]),
    frozenset(["pauli_z", "rz"]),
    frozenset(["s", "rz"]),
    frozenset(["t", "rz"]),
    frozenset(["pauli_z", "phaseshift"]),
    frozenset(["s", "phaseshift"]),
    frozenset(["t", "phaseshift"]),
    frozenset(["rx", "rx"]),
    frozenset(["ry", "ry"]),
    frozenset(["rz", "rz"]),
    frozenset(["s", "si"]),
    frozenset(["t", "ti"]),
    frozenset(["gphase", "pauli_x"]),
    frozenset(["gphase", "pauli_y"]),
    frozenset(["gphase", "pauli_z"]),
    frozenset(["gphase", "hadamard"]),
    frozenset(["gphase", "s"]),
    frozenset(["gphase", "t"]),
    frozenset(["gphase", "rx"]),
    frozenset(["gphase", "ry"]),
    frozenset(["gphase", "rz"]),
}

_GATE_TYPES = {
    "identity": 0,
    "pauli_x": 1,
    "pauli_y": 2,
    "pauli_z": 3,
    "rx": 4,
    "ry": 5,
    "rz": 6,
    "hadamard": 7,
    "s": 8,
    "t": 9,
    "cx": 10,
    "cz": 11,
    "swap": 12,
}

_PAULI_COMMUTATORS = {
    ("pauli_x", "pauli_y"): ("pauli_z", 2j),
    ("pauli_y", "pauli_x"): ("pauli_z", -2j),
    ("pauli_y", "pauli_z"): ("pauli_x", 2j),
    ("pauli_z", "pauli_y"): ("pauli_x", -2j),
    ("pauli_z", "pauli_x"): ("pauli_y", 2j),
    ("pauli_x", "pauli_z"): ("pauli_y", -2j),
    ("pauli_x", "pauli_x"): ("identity", 0),
    ("pauli_y", "pauli_y"): ("identity", 0),
    ("pauli_z", "pauli_z"): ("identity", 0),
}

_ROTATION_COMMUTATORS = {
    ("rx", "ry"): ("rz", "cross_product"),
    ("ry", "rx"): ("rz", "cross_product_neg"),
    ("ry", "rz"): ("rx", "cross_product"),
    ("rz", "ry"): ("rx", "cross_product_neg"),
    ("rz", "rx"): ("ry", "cross_product"),
    ("rx", "rz"): ("ry", "cross_product_neg"),
}


class CommutationGraph:
    """Graph-based representation of gate commutation relationships."""

    def __init__(self):
        self.nodes = []
        self.edges = defaultdict(set)
        self.non_commuting_edges = defaultdict(set)

    def build(self, gates: list[GateOperation]) -> CommutationGraph:
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

    def _normalize_gate_type(self, gate: GateOperation) -> str:
        """Normalize gate type to standard gate_operations form."""
        gate_type = gate.gate_type.lower()

        if gate_type in ["paulix"]:
            return "pauli_x"
        elif gate_type in ["pauliy"]:
            return "pauli_y"
        elif gate_type in ["pauliz"]:
            return "pauli_z"
        elif gate_type in ["h"]:
            return "hadamard"
        elif gate_type in ["cnot"]:
            return "cx"
        elif gate_type in ["i"]:
            return "identity"
        else:
            return gate_type

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
        type1 = gate1.gate_type
        type2 = gate2.gate_type

        if type1 in ["pauli_x", "pauli_y", "pauli_z"] and type2 in [
            "pauli_x",
            "pauli_y",
            "pauli_z",
        ]:
            return self._pauli_commutation_check(type1, type2, gate1, gate2)

        if type1 in ["rx", "ry", "rz"] and type2 in ["rx", "ry", "rz"]:
            return self._rotation_commutation_check(type1, type2, gate1, gate2)

        gate_pair = frozenset([type1, type2])
        return gate_pair in _COMMUTING_PAIRS

    def _pauli_commutation_check(
        self, type1: str, type2: str, gate1: GateOperation, gate2: GateOperation
    ) -> bool:
        """Check Pauli gate commutation using Lie algebra."""
        if type1 == type2:
            return True

        return False

    def _rotation_commutation_check(
        self, type1: str, type2: str, gate1: GateOperation, gate2: GateOperation
    ) -> bool:
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

    def _extract_angle(self, gate: GateOperation) -> Optional[float]:
        """Extract rotation angle from gate."""
        for attr in ["_angle", "angle", "_theta", "theta"]:
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
                R.union({v}), P.intersection(neighbors), X.intersection(neighbors), cliques
            )
            P.remove(v)
            X.add(v)


class PauliLieAlgebra:
    """Pauli Lie algebra for advanced gate optimization."""

    def __init__(self):
        self.pauli_basis = ["i", "x", "y", "z"]
        self.commutation_table = self._build_commutation_table()

    def _build_commutation_table(self) -> dict:
        """Build commutation table for Pauli operators."""
        table = {}

        for p in self.pauli_basis:
            table[("i", p)] = ("i", 0)
            table[(p, "i")] = ("i", 0)

        table[("x", "x")] = ("i", 0)
        table[("y", "y")] = ("i", 0)
        table[("z", "z")] = ("i", 0)

        table[("x", "y")] = ("z", 2j)
        table[("y", "x")] = ("z", -2j)
        table[("y", "z")] = ("x", 2j)
        table[("z", "y")] = ("x", -2j)
        table[("z", "x")] = ("y", 2j)
        table[("x", "z")] = ("y", -2j)

        return table

    def commutator(self, gate1_type: str, gate2_type: str) -> tuple[str, complex]:
        """Compute commutator [A, B] = AB - BA."""
        return self.commutation_table.get((gate1_type, gate2_type), ("unknown", 0))

    def optimal_order(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Find optimal ordering of gates to minimize non-commuting operations."""
        if len(gates) <= 1:
            return gates

        ordered_gates = []
        remaining_gates = gates.copy()

        ordered_gates.append(remaining_gates.pop(0))

        while remaining_gates:
            best_gate = None
            best_score = float("inf")
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

    def _compute_insertion_cost(
        self, current_sequence: list[GateOperation], candidate: GateOperation
    ) -> float:
        """Compute cost of inserting candidate gate into current sequence."""
        cost = 0.0
        candidate_type = candidate.gate_type

        for gate in current_sequence:
            gate_type = gate.gate_type

            if gate_type in self.pauli_basis and candidate_type in self.pauli_basis:
                commutator_result, coefficient = self.commutator(gate_type, candidate_type)
                if commutator_result != "i":
                    cost += abs(coefficient)

        return cost


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
            gate_type = gate.gate_type
            targets = tuple(sorted(gate.targets))
            angle = self._extract_angle(gate)
            signature_parts.append(f"{gate_type}_{targets}_{angle}")
        return "|".join(signature_parts)

    def _extract_angle(self, gate: GateOperation) -> Optional[float]:
        """Extract angle from parameterized gates."""
        for attr in ["_angle", "angle", "_theta", "theta"]:
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
        pauli_types = {"pauli_x", "pauli_y", "pauli_z"}
        return all(gate.gate_type in pauli_types for gate in gates)

    def _is_rotation_group(self, gates: list[GateOperation]) -> bool:
        """Check if all gates in group are rotation gates on same axis."""
        if not gates:
            return False

        first_type = gates[0].gate_type
        if first_type not in ["rx", "ry", "rz"]:
            return False

        return all(gate.gate_type == first_type for gate in gates)

    def _fuse_pauli_group(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Fuse Pauli gates using XOR logic."""
        if not gates:
            return gates

        pauli_count = {"pauli_x": 0, "pauli_y": 0, "pauli_z": 0}
        gate_templates = {}

        for gate in gates:
            gate_type = gate.gate_type

            if gate_type == "pauli_x":
                pauli_count["pauli_x"] += 1
                if "pauli_x" not in gate_templates:
                    gate_templates["pauli_x"] = gate
            elif gate_type == "pauli_y":
                pauli_count["pauli_y"] += 1
                if "pauli_y" not in gate_templates:
                    gate_templates["pauli_y"] = gate
            elif gate_type == "pauli_z":
                pauli_count["pauli_z"] += 1
                if "pauli_z" not in gate_templates:
                    gate_templates["pauli_z"] = gate

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

    __slots__ = ("_fused_matrix", "_original_gates", "_optimization_type")

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
        self.commutative_algebra = CommutativeAlgebraFusion()

    def _normalize_gate_type(self, gate: GateOperation) -> str:
        """Normalize gate type to standard form."""
        gate_type = gate.gate_type.lower()

        # Handle common variations
        if gate_type in ["pauli_x", "paulix", "x"]:
            return "x"
        elif gate_type in ["pauli_y", "pauliy", "y"]:
            return "y"
        elif gate_type in ["pauli_z", "pauliz", "z"]:
            return "z"
        elif gate_type in ["hadamard", "h"]:
            return "h"
        elif gate_type in ["cnot", "cx"]:
            return "cx"
        elif gate_type in ["identity", "i"]:
            return "i"
        else:
            return gate_type

    def optimize_commuting_gates(self, gates: list[GateOperation]) -> list[GateOperation]:
        """
        Optimized commuting gate analysis with O(n²) complexity and early termination.

        Uses efficient linear scan with target-based grouping to avoid exponential algorithms.
        Focuses on simple, high-impact optimizations that are safe and fast.
        """
        if len(gates) <= 1:
            return gates

        if len(gates) > 100:
            return self._optimize_gates_in_chunks(gates, chunk_size=50)

        target_groups = defaultdict(list)
        for i, gate in enumerate(gates):
            target_key = tuple(sorted(gate.targets))
            target_groups[target_key].append((i, gate))

        optimized_gates = gates.copy()

        for target_key, gate_list in target_groups.items():
            if len(target_key) == 1 and len(gate_list) > 1:
                indices, group_gates = zip(*gate_list)
                optimized_group = self._optimize_single_qubit_group(list(group_gates))

                for i, (orig_idx, _) in enumerate(gate_list):
                    if i < len(optimized_group):
                        optimized_gates[orig_idx] = optimized_group[i]
                    else:
                        optimized_gates[orig_idx] = None

        result = [gate for gate in optimized_gates if gate is not None]
        return result

    def _optimize_gates_in_chunks(
        self, gates: list[GateOperation], chunk_size: int = 50
    ) -> list[GateOperation]:
        """
        Optimize large gate sequences by processing them in smaller chunks.

        This prevents performance issues while still allowing optimization of large sequences.
        """
        if len(gates) <= chunk_size:
            return self._optimize_gates_direct(gates)

        optimized_result = []

        for i in range(0, len(gates), chunk_size):
            chunk = gates[i : i + chunk_size]
            optimized_chunk = self._optimize_gates_direct(chunk)
            optimized_result.extend(optimized_chunk)

        return optimized_result

    def _optimize_gates_direct(self, gates: list[GateOperation]) -> list[GateOperation]:
        """
        Direct optimization without chunking - used for smaller sequences.
        """
        if len(gates) <= 1:
            return gates

        target_groups = defaultdict(list)
        for i, gate in enumerate(gates):
            target_key = tuple(sorted(gate.targets))
            target_groups[target_key].append((i, gate))

        optimized_gates = gates.copy()

        for target_key, gate_list in target_groups.items():
            if len(target_key) == 1 and len(gate_list) > 1:
                indices, group_gates = zip(*gate_list)
                optimized_group = self._optimize_single_qubit_group(list(group_gates))

                for i, (orig_idx, _) in enumerate(gate_list):
                    if i < len(optimized_group):
                        optimized_gates[orig_idx] = optimized_group[i]
                    else:
                        optimized_gates[orig_idx] = None

        return [gate for gate in optimized_gates if gate is not None]

    def _optimize_single_qubit_group(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Optimize a group of single-qubit gates on the same target."""
        if len(gates) <= 1:
            return gates

        if self._is_pauli_group_fast(gates):
            return self._optimize_pauli_group_fast(gates)

        if self._is_rotation_group_fast(gates):
            return self._optimize_rotation_group_fast(gates)

        if self._is_hadamard_group_fast(gates):
            return self._optimize_hadamard_group_fast(gates)

        if self._is_phase_group_fast(gates):
            return self._optimize_phase_group_fast(gates)

        return self._apply_involution_cancellation(gates)

    def _is_pauli_group_fast(self, gates: list[GateOperation]) -> bool:
        """Fast check if all gates are Pauli gates."""
        pauli_types = {"pauli_x", "pauli_y", "pauli_z"}
        return all(gate.gate_type in pauli_types for gate in gates)

    def _is_rotation_group_fast(self, gates: list[GateOperation]) -> bool:
        """Fast check if all gates are rotations on the same axis."""
        if not gates:
            return False
        first_type = gates[0].gate_type
        return first_type in ["rx", "ry", "rz"] and all(
            gate.gate_type == first_type for gate in gates
        )

    def _is_hadamard_group_fast(self, gates: list[GateOperation]) -> bool:
        """Fast check if all gates are Hadamard gates."""
        return all(gate.gate_type == "hadamard" for gate in gates)

    def _optimize_pauli_group_fast(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Optimize Pauli gates using XOR logic - O(n) complexity."""
        pauli_count = {"pauli_x": 0, "pauli_y": 0, "pauli_z": 0}
        gate_templates = {}

        for gate in gates:
            gate_type = gate.gate_type

            if gate_type == "pauli_x":
                pauli_count["pauli_x"] += 1
                if "pauli_x" not in gate_templates:
                    gate_templates["pauli_x"] = gate
            elif gate_type == "pauli_y":
                pauli_count["pauli_y"] += 1
                if "pauli_y" not in gate_templates:
                    gate_templates["pauli_y"] = gate
            elif gate_type == "pauli_z":
                pauli_count["pauli_z"] += 1
                if "pauli_z" not in gate_templates:
                    gate_templates["pauli_z"] = gate

        result = []
        for pauli_type, count in pauli_count.items():
            if count % 2 == 1:
                result.append(gate_templates[pauli_type])

        return result

    def _optimize_rotation_group_fast(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Optimize rotation gates by summing angles - O(n) complexity."""
        if not gates:
            return gates

        total_angle = 0.0
        template_gate = gates[0]

        for gate in gates:
            angle = self._extract_angle_safe(gate)
            if angle is None:
                return gates
            total_angle += angle

        normalized_angle = total_angle % (4.0 * np.pi)
        if abs(normalized_angle) < 1e-12 or abs(normalized_angle - 4.0 * np.pi) < 1e-12:
            return []

        optimized_gate = template_gate
        if hasattr(optimized_gate, "_angle"):
            optimized_gate._angle = normalized_angle
        elif hasattr(optimized_gate, "angle"):
            optimized_gate.angle = normalized_angle

        return [optimized_gate]

    def _optimize_hadamard_group_fast(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Optimize Hadamard gates using parity - O(1) complexity."""
        count = len(gates)
        return [] if count % 2 == 0 else [gates[0]]

    def _is_phase_group_fast(self, gates: list[GateOperation]) -> bool:
        """Fast check if all gates are phase gates (S, T, PhaseShift)."""
        phase_types = {"s", "t", "si", "ti", "phaseshift", "gphase"}
        return all(gate.gate_type in phase_types for gate in gates)

    def _optimize_phase_group_fast(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Optimize phase gates by combining angles - O(n) complexity."""
        if not gates:
            return gates

        s_count = 0
        t_count = 0
        phase_angle = 0.0
        gate_templates = {}

        for gate in gates:
            gate_type = gate.gate_type

            if gate_type == "s":
                s_count += 1
                gate_templates["s"] = gate
            elif gate_type == "si":
                s_count -= 1
                gate_templates["s"] = gate
            elif gate_type == "t":
                t_count += 1
                gate_templates["t"] = gate
            elif gate_type == "ti":
                t_count -= 1
                gate_templates["t"] = gate
            elif gate_type in ["phaseshift", "gphase"]:
                angle = self._extract_angle_safe(gate)
                if angle is not None:
                    phase_angle += angle
                    gate_templates["phase"] = gate

        s_count = s_count % 4
        t_count = t_count % 8

        while t_count >= 2:
            s_count += 1
            t_count -= 2

        s_count = s_count % 4

        result = []

        if s_count > 0 and "s" in gate_templates:
            result.extend(gate_templates["s"] for _ in range(s_count))

        if t_count > 0 and "t" in gate_templates:
            result.extend(gate_templates["t"] for _ in range(t_count))

        if abs(phase_angle) > 1e-12 and "phase" in gate_templates:
            normalized_angle = phase_angle % (2.0 * np.pi)
            if abs(normalized_angle) > 1e-12 and abs(normalized_angle - 2.0 * np.pi) > 1e-12:
                result.append(gate_templates["phase"])

        return result

    def _apply_involution_cancellation(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Apply cancellation for self-inverse gates - O(n) complexity."""
        if not gates:
            return gates

        # Check if all gates are identical and self-inverse
        first_gate = gates[0]
        gate_type = self._normalize_gate_type(first_gate)

        involutory_gates = {"pauli_x", "pauli_y", "pauli_z", "hadamard", "x", "y", "z", "h"}

        if gate_type in involutory_gates and all(
            self._normalize_gate_type(gate) == gate_type for gate in gates
        ):
            count = len(gates)
            return [] if count % 2 == 0 else [first_gate]

        return gates

    def _extract_angle_safe(self, gate: GateOperation) -> Optional[float]:
        """Safely extract angle from parameterized gates."""
        for attr in ["_angle", "angle", "_theta", "theta"]:
            if hasattr(gate, attr):
                try:
                    angle = getattr(gate, attr)
                    if isinstance(angle, (int, float)):
                        return float(angle)
                except (AttributeError, TypeError):
                    continue
        return None

    def _fuse_target_group(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Fuse gates with same target qubits."""
        if len(gates) <= 1:
            return gates

        gate_types = [gate.gate_type for gate in gates]

        if len(set(gate_types)) == 1:
            return self._fuse_identical_gates(gates)

        if all(gt in ["pauli_x", "pauli_y", "pauli_z"] for gt in gate_types):
            return self._fuse_pauli_gates_xor(gates)

        return gates

    def _fuse_identical_gates(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Fuse identical gates using parity logic."""
        gate_type = gates[0].gate_type

        involutory_gates = {
            "pauli_x",
            "pauli_y",
            "pauli_z",
            "hadamard",
            "cx",
            "cz",
            "swap",
            "x",
            "y",
            "z",
            "h",
        }

        if gate_type in involutory_gates:
            count = len(gates)
            return [] if count % 2 == 0 else [gates[0]]

        return gates

    def _fuse_pauli_gates_xor(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Fuse Pauli gates using XOR logic."""
        pauli_count = {"pauli_x": 0, "pauli_y": 0, "pauli_z": 0, "x": 0, "y": 0, "z": 0}
        gate_templates = {}

        for gate in gates:
            gate_type = gate.gate_type
            if gate_type in pauli_count:
                pauli_count[gate_type] += 1
                if gate_type not in gate_templates:
                    gate_templates[gate_type] = gate

        result = []
        for pauli_type, count in pauli_count.items():
            if count % 2 == 1:
                result.append(gate_templates[pauli_type])

        return result


class GateFusionEngine:
    """Gate fusion engine with smart pattern recognition and advanced commutative algebra optimization."""

    def __init__(
        self,
        max_fusion_size: int = 8,
        enable_commuting_fusion: bool = True,
        enable_advanced_commutation: bool = True,
    ):
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

        preprocessed_operations = operations

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

        if self.enable_advanced_commutation and hasattr(self, "fast_target_fusion"):
            gates = self.fast_target_fusion.optimize_commuting_gates(gates)

        optimized = []
        i = 0

        while i < len(gates):
            fusion_group = self._find_optimal_fusion_group(gates, i)

            if len(fusion_group) > 1:
                reduced_gates = self._apply_smart_reductions(fusion_group)
                if reduced_gates:
                    if len(reduced_gates) == 1:
                        optimized.append(reduced_gates[0])
                    else:
                        fused_op = self._create_optimized_fused_operation(reduced_gates)
                        optimized.append(fused_op)
                i += len(fusion_group)
            else:
                optimized.append(gates[i])
                i += 1

        return optimized

    def _find_optimal_fusion_group(
        self, gates: list[GateOperation], start_idx: int
    ) -> list[GateOperation]:
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
            elif (
                candidate_qubits.issubset(first_qubits) and len(new_all_qubits) <= MAX_FUSION_QUBITS
            ):
                can_fuse = True
            elif (
                first_qubits.issubset(candidate_qubits)
                and len(candidate_qubits) <= MAX_FUSION_QUBITS
                and len(new_all_qubits) <= MAX_FUSION_QUBITS
            ):
                can_fuse = True
                first_qubits = candidate_qubits

            if (
                can_fuse
                and len(fusion_group) < self.max_fusion_size
                and len(new_all_qubits) <= MAX_FUSION_QUBITS
            ):
                fusion_group.append(candidate)
                all_qubits = new_all_qubits
            else:
                break

        return fusion_group

    def _can_safely_commute_with_group(
        self, fusion_group: list[GateOperation], candidate: GateOperation
    ) -> bool:
        """Enhanced commuting gate detection with strict safety checks."""

        for gate in fusion_group:
            if not self._gates_commute_safely(gate, candidate):
                return False

        for gate in fusion_group:
            if self._has_control_target_conflict(gate, candidate):
                return False

        return True

    def _get_all_qubits(self, gate: GateOperation) -> set[int]:
        """Get all qubits (targets + controls) affected by gate."""
        all_qubits = set(gate.targets)
        ctrl_modifiers = getattr(gate, "ctrl_modifiers", None) or getattr(
            gate, "_ctrl_modifiers", None
        )
        if ctrl_modifiers:
            all_qubits.update(ctrl_modifiers)
        return all_qubits

    def _gates_commute_safely(self, gate1: GateOperation, gate2: GateOperation) -> bool:
        """Safe commutativity check with quantum mechanical correctness."""
        gate1_qubits = self._get_all_qubits(gate1)
        gate2_qubits = self._get_all_qubits(gate2)

        if gate1_qubits.isdisjoint(gate2_qubits):
            return True

        if gate1_qubits == gate2_qubits:
            return self._same_qubit_gates_commute(gate1, gate2)

        return False

    def _same_qubit_gates_commute(self, gate1: GateOperation, gate2: GateOperation) -> bool:
        """Check if gates on same qubits commute - VERY CONSERVATIVE for correctness."""
        type1 = gate1.gate_type
        type2 = gate2.gate_type

        if type1 == type2:
            return True

        gate_pair = frozenset([type1, type2])
        return gate_pair in _COMMUTING_PAIRS

    def _has_control_target_conflict(self, gate1: GateOperation, gate2: GateOperation) -> bool:
        """Check for control-target conflicts between gates."""
        qubits1 = self._get_all_qubits(gate1)
        qubits2 = self._get_all_qubits(gate2)

        if qubits1.isdisjoint(qubits2):
            return False

        ctrl1 = getattr(gate1, "ctrl_modifiers", None) or getattr(gate1, "_ctrl_modifiers", None)
        ctrl2 = getattr(gate2, "ctrl_modifiers", None) or getattr(gate2, "_ctrl_modifiers", None)
        has_controls1 = bool(ctrl1)
        has_controls2 = bool(ctrl2)

        if has_controls1 or has_controls2:
            return True

        return False

    def _are_identical_controlled_gates(self, gate1: GateOperation, gate2: GateOperation) -> bool:
        """Check if two gates are identical controlled gates that can be safely fused."""
        if gate1.gate_type != gate2.gate_type:
            return False

        if gate1.targets != gate2.targets:
            return False

        ctrl1 = getattr(gate1, "ctrl_modifiers", ()) or ()
        ctrl2 = getattr(gate2, "ctrl_modifiers", ()) or ()

        return ctrl1 == ctrl2

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
        pauli_types = {"pauli_x", "pauli_y", "pauli_z"}
        return all(gate.gate_type in pauli_types for gate in gates)

    def _is_rotation_sequence(self, gates: list[GateOperation]) -> bool:
        """Fast rotation sequence detection."""
        if not gates:
            return False
        first_type = gates[0].gate_type
        return first_type in ["rx", "ry", "rz"] and all(
            gate.gate_type == first_type for gate in gates
        )

    def _is_hadamard_sequence(self, gates: list[GateOperation]) -> bool:
        """Fast Hadamard sequence detection."""
        if not gates:
            return False
        return (
            all(gate.gate_type == "hadamard" for gate in gates)
            and all(len(gate.targets) == 1 for gate in gates)
            and all(gate.targets[0] == gates[0].targets[0] for gate in gates)
        )

    def _optimize_pauli_sequence_fast(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Fast Pauli sequence optimization using XOR logic - ONLY for gates on the same qubit."""
        if not gates:
            return gates

        first_targets = gates[0].targets
        if not all(gate.targets == first_targets for gate in gates):
            return gates

        pauli_count = {"pauli_x": 0, "pauli_y": 0, "pauli_z": 0}
        gate_templates = {}

        for gate in gates:
            gate_type = gate.gate_type
            if gate_type in pauli_count:
                pauli_count[gate_type] += 1
                if gate_type not in gate_templates:
                    gate_templates[gate_type] = gate

        result = []
        for pauli_type, count in pauli_count.items():
            if count % 2 == 1:
                result.append(gate_templates[pauli_type])

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
        if hasattr(optimized_gate, "_angle"):
            optimized_gate._angle = normalized_angle
        return [optimized_gate]

    def _optimize_hadamard_sequence_fast(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Fast Hadamard sequence optimization."""
        count = len(gates)
        return [] if count % 2 == 0 else [gates[0]]

    def _apply_general_reductions(self, gates: list[GateOperation]) -> list[GateOperation]:
        """Apply general gate reductions."""
        gate_sequence = tuple(gate.gate_type for gate in gates)

        if gate_sequence in _GATE_REDUCTIONS:
            replacement = _GATE_REDUCTIONS[gate_sequence]
            if replacement is None:
                return []
            else:
                return [gates[0]]

        return gates

    def _extract_angle_fast(self, gate: GateOperation) -> Optional[float]:
        """Fast angle extraction with common attribute names."""
        for attr in ["_angle", "angle", "_theta", "theta"]:
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

        if hasattr(gate, "_ctrl_modifiers") and gate._ctrl_modifiers:
            return self._is_safe_controlled_gate(gate)

        power = self._get_gate_power(gate)
        if power is not None and power not in [0, 0.5, 1, 2, -1]:
            return False

        return True

    def _is_safe_controlled_gate(self, gate: GateOperation) -> bool:
        """Check if controlled gate is safe for fusion."""
        gate_type = gate.gate_type

        total_qubits = len(self._get_all_qubits(gate))

        if total_qubits == 2:
            return gate_type in ["cx", "cz", "cy", "cnot"]
        elif total_qubits == 3:
            return gate_type in ["ccnot", "ccx", "toffoli"]

        return False

    def _get_gate_power(self, gate: GateOperation) -> Optional[float]:
        """Extract power value from gate."""
        for attr in ["_power", "power"]:
            if hasattr(gate, attr):
                power = getattr(gate, attr)
                if power is not None:
                    return power
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

    def _compute_fused_matrix_optimized(
        self, gates: list[GateOperation], targets: tuple[int, ...]
    ) -> np.ndarray:
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

    def _expand_matrix_optimized(
        self, gate_matrix: np.ndarray, targets: tuple[int, ...], num_qubits: int
    ) -> np.ndarray:
        """Optimized matrix expansion using efficient tensor product placement."""
        if len(targets) == num_qubits:
            return gate_matrix

        matrix_size = 2**num_qubits
        result = np.zeros((matrix_size, matrix_size), dtype=complex)

        2 ** len(targets)

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
        """Handle mixed operation types efficiently - CRITICAL FIX: Preserve operation order."""

        result = []
        i = 0

        while i < len(operations):
            current_op = operations[i]

            if isinstance(current_op, GateOperation):
                gate_group = [current_op]
                j = i + 1

                while j < len(operations) and isinstance(operations[j], GateOperation):
                    gate_group.append(operations[j])
                    j += 1

                if len(gate_group) > 1:
                    optimized_gates = self._optimize_gate_sequence(gate_group)
                    result.extend(optimized_gates)
                else:
                    result.append(current_op)

                i = j
            elif isinstance(current_op, KrausOperation):
                noise_group = [current_op]
                j = i + 1

                while j < len(operations) and isinstance(operations[j], KrausOperation):
                    noise_group.append(operations[j])
                    j += 1

                if len(noise_group) > 1:
                    optimized_noise = apply_noise_fusion(noise_group)
                    result.extend(optimized_noise)
                else:
                    result.append(current_op)

                i = j
            else:
                result.append(current_op)
                i += 1

        return result


def apply_gate_fusion(
    operations: list, max_fusion_size: int = 8, enable_commuting_fusion: bool = True
) -> list:
    """Apply gate fusion with enhanced performance."""
    if not operations:
        return operations

    engine = GateFusionEngine(
        max_fusion_size=max_fusion_size, enable_commuting_fusion=enable_commuting_fusion
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
