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

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from braket.default_simulator.operation import GateOperation


class CircuitClass(Enum):
    PRODUCT = auto()
    CLIFFORD = auto()
    DIAGONAL = auto()
    MATCHGATE = auto()
    LOW_ENTANGLEMENT = auto()
    QFT_LIKE = auto()
    GENERAL = auto()


CLIFFORD_GATES = frozenset({
    "hadamard", "pauli_x", "pauli_y", "pauli_z", "s", "si", "cx", "cz", "swap",
    "h", "x", "y", "z", "cnot", "sdg"
})

DIAGONAL_GATES = frozenset({
    "pauli_z", "s", "si", "t", "ti", "rz", "phaseshift", "cz", "cphaseshift",
    "cphaseshift01", "cphaseshift00", "cphaseshift10", "zz", "z", "sdg", "tdg"
})

PRODUCT_GATES = frozenset({
    "hadamard", "pauli_x", "pauli_y", "pauli_z", "s", "si", "t", "ti",
    "rx", "ry", "rz", "phaseshift", "u", "h", "x", "y", "z", "sdg", "tdg",
    "identity", "gpi", "gpi2", "v", "vi", "prx"
})

CONTROLLED_PHASE_GATES = frozenset({
    "cphaseshift", "cphaseshift00", "cphaseshift01", "cphaseshift10", "cz", "zz"
})

QFT_GATES = frozenset({
    "h", "hadamard", "cphaseshift", "swap"
})


@dataclass
class AnalysisReport:
    n_qubits: int
    gate_count: int
    gate_distribution: dict[str, int] = field(default_factory=dict)
    circuit_class: CircuitClass = CircuitClass.GENERAL
    entanglement_structure: dict[int, set[int]] = field(default_factory=dict)
    connected_components: list[set[int]] = field(default_factory=list)
    estimated_bond_dimension: int | None = None
    recommended_backend: str = "full"
    max_gate_distance: int = 0
    is_nearest_neighbor: bool = False


class CircuitAnalyzer:
    def __init__(self, operations: list[GateOperation], n_qubits: int):
        self.operations = operations
        self.n_qubits = n_qubits

    def analyze(self) -> AnalysisReport:
        if not self.operations:
            return AnalysisReport(
                n_qubits=self.n_qubits,
                gate_count=0,
                circuit_class=CircuitClass.PRODUCT,
                connected_components=[{i} for i in range(self.n_qubits)],
                estimated_bond_dimension=1,
                recommended_backend="product",
                is_nearest_neighbor=True,
            )

        gate_dist = {}
        has_two_qubit = False
        all_clifford = True
        all_diagonal = True
        two_qubit_count = 0
        max_distance = 0
        controlled_phase_count = 0
        hadamard_count = 0

        max_qubit_idx = max(
            max(op.targets) for op in self.operations if op.targets
        )
        effective_n_qubits = max(self.n_qubits, max_qubit_idx + 1)

        parent = list(range(effective_n_qubits))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for op in self.operations:
            gate_type = getattr(op, "gate_type", "unknown").lower()
            gate_dist[gate_type] = gate_dist.get(gate_type, 0) + 1
            targets = op.targets
            n_targets = len(targets)

            if gate_type in ("hadamard", "h"):
                hadamard_count += 1

            if n_targets >= 2:
                has_two_qubit = True
                two_qubit_count += 1
                distance = max(targets) - min(targets)
                max_distance = max(max_distance, distance)

                if gate_type in CONTROLLED_PHASE_GATES:
                    controlled_phase_count += 1

                for i in range(1, n_targets):
                    union(targets[0], targets[i])

            if gate_type not in CLIFFORD_GATES:
                all_clifford = False
            if gate_type not in DIAGONAL_GATES:
                all_diagonal = False

        components_map = {}
        for i in range(effective_n_qubits):
            root = find(i)
            if root not in components_map:
                components_map[root] = set()
            components_map[root].add(i)
        components = list(components_map.values())

        is_nearest_neighbor = max_distance <= 1
        is_qft_like = (
            hadamard_count >= self.n_qubits // 2 and
            controlled_phase_count >= two_qubit_count * 0.7 and
            two_qubit_count > 0
        )

        if not has_two_qubit:
            circuit_class = CircuitClass.PRODUCT
            backend = "product"
            bond_dim = 1
        elif is_qft_like:
            circuit_class = CircuitClass.QFT_LIKE
            backend = "mps"
            bond_dim = min(2 ** (self.n_qubits // 4 + 1), 64)
        elif all_clifford:
            circuit_class = CircuitClass.CLIFFORD
            backend = "clifford"
            bond_dim = 2 ** min(two_qubit_count, self.n_qubits // 2)
        elif all_diagonal:
            circuit_class = CircuitClass.DIAGONAL
            backend = "diagonal"
            bond_dim = 1
        elif is_nearest_neighbor and two_qubit_count <= self.n_qubits * 3:
            circuit_class = CircuitClass.LOW_ENTANGLEMENT
            backend = "mps"
            bond_dim = min(2 ** (two_qubit_count // self.n_qubits + 1), 64)
        else:
            bond_dim = min(2 ** two_qubit_count, 2 ** (self.n_qubits // 2))
            if bond_dim <= 64:
                circuit_class = CircuitClass.LOW_ENTANGLEMENT
                backend = "mps"
            else:
                circuit_class = CircuitClass.GENERAL
                backend = "full"

        if len(components) > 1 and backend == "full":
            max_size = max(len(c) for c in components)
            if max_size < self.n_qubits:
                backend = "partitioned"

        return AnalysisReport(
            n_qubits=self.n_qubits,
            gate_count=len(self.operations),
            gate_distribution=gate_dist,
            circuit_class=circuit_class,
            entanglement_structure={},
            connected_components=components,
            estimated_bond_dimension=bond_dim,
            recommended_backend=backend,
            max_gate_distance=max_distance,
            is_nearest_neighbor=is_nearest_neighbor,
        )

    def classify(self) -> CircuitClass:
        return self.analyze().circuit_class

    def get_entanglement_graph(self) -> dict[int, set[int]]:
        max_qubit = self.n_qubits - 1
        if self.operations:
            for op in self.operations:
                if op.targets:
                    max_qubit = max(max_qubit, max(op.targets))
        graph = {i: set() for i in range(max_qubit + 1)}
        for op in self.operations:
            targets = op.targets
            if len(targets) >= 2:
                for i, q1 in enumerate(targets):
                    for q2 in targets[i + 1:]:
                        graph[q1].add(q2)
                        graph[q2].add(q1)
        return graph

    def get_connected_components(self) -> list[set[int]]:
        return self.analyze().connected_components

    def estimate_bond_dimension(self) -> int:
        return self.analyze().estimated_bond_dimension

    def identify_subcircuit_classes(self) -> list[tuple[range, CircuitClass]]:
        if not self.operations:
            return []

        regions = []
        start_idx = 0
        current_class = self._classify_op(self.operations[0])

        for i, op in enumerate(self.operations[1:], 1):
            op_class = self._classify_op(op)
            if op_class != current_class:
                regions.append((range(start_idx, i), current_class))
                start_idx = i
                current_class = op_class

        regions.append((range(start_idx, len(self.operations)), current_class))
        return regions

    def _classify_op(self, op: GateOperation) -> CircuitClass:
        gate_type = getattr(op, "gate_type", "unknown").lower()
        if len(op.targets) == 1:
            return CircuitClass.PRODUCT
        if gate_type in CLIFFORD_GATES:
            return CircuitClass.CLIFFORD
        if gate_type in DIAGONAL_GATES:
            return CircuitClass.DIAGONAL
        return CircuitClass.GENERAL

    def get_fusable_blocks(self, max_block_qubits: int = 4) -> list[list[int]]:
        if not self.operations:
            return []

        blocks = []
        current_block = [0]
        current_qubits = set(self.operations[0].targets)

        for i, op in enumerate(self.operations[1:], 1):
            op_qubits = set(op.targets)
            merged_qubits = current_qubits | op_qubits

            if len(merged_qubits) <= max_block_qubits and (op_qubits & current_qubits):
                current_block.append(i)
                current_qubits = merged_qubits
            else:
                if len(current_block) > 1:
                    blocks.append(current_block)
                current_block = [i]
                current_qubits = op_qubits

        if len(current_block) > 1:
            blocks.append(current_block)

        return blocks
