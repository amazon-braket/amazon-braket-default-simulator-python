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
partitioner.py

Qubit partitioning for independent subsystem simulation.

This module provides tools for identifying and exploiting independent
qubit partitions in quantum circuits, enabling parallel simulation
of disconnected subsystems.

Mathematical Basis:
- Graph-based connectivity analysis
- Independent subsystem identification
- Result combination via tensor products

Dependencies:
- numpy
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from braket.default_simulator.circuit_analyzer import CircuitAnalyzer, CircuitClass

if TYPE_CHECKING:
    from braket.default_simulator.operation import GateOperation


@dataclass
class QubitPartition:
    qubits: set[int]
    circuit_class: CircuitClass
    operations: list[Any] = field(default_factory=list)
    qubit_map: dict[int, int] = field(default_factory=dict)


class QubitPartitioner:
    """
    Partitions quantum circuits into independent subsystems.

    This partitioner analyzes entanglement structure to identify
    qubits that can be simulated independently.
    """

    def __init__(self, operations: list[GateOperation], n_qubits: int):
        self.operations = operations
        self.n_qubits = n_qubits
        self.entanglement_graph: dict[int, set[int]] = defaultdict(set)
        self.partitions: list[QubitPartition] = []
        self._analyzed = False

    def analyze_entanglement(self) -> dict[int, set[int]]:
        for i in range(self.n_qubits):
            self.entanglement_graph[i] = set()

        for op in self.operations:
            targets = op.targets
            if len(targets) >= 2:
                for i, q1 in enumerate(targets):
                    for q2 in targets[i + 1:]:
                        self.entanglement_graph[q1].add(q2)
                        self.entanglement_graph[q2].add(q1)

        return dict(self.entanglement_graph)

    def find_connected_components(self) -> list[set[int]]:
        if not self.entanglement_graph:
            self.analyze_entanglement()

        visited = set()
        components = []

        for qubit in range(self.n_qubits):
            if qubit not in visited:
                component = set()
                self._dfs(qubit, visited, component)
                components.append(component)

        return components

    def _dfs(self, qubit: int, visited: set[int], component: set[int]) -> None:
        visited.add(qubit)
        component.add(qubit)
        for neighbor in self.entanglement_graph.get(qubit, set()):
            if neighbor not in visited:
                self._dfs(neighbor, visited, component)

    def partition(self) -> list[QubitPartition]:
        if self._analyzed:
            return self.partitions

        components = self.find_connected_components()
        self.partitions = []

        for component in components:
            local_ops = self.extract_local_operations(component)

            qubit_list = sorted(component)
            qubit_map = {q: i for i, q in enumerate(qubit_list)}

            analyzer = CircuitAnalyzer(local_ops, len(component))
            circuit_class = analyzer.classify()

            partition = QubitPartition(
                qubits=component,
                circuit_class=circuit_class,
                operations=local_ops,
                qubit_map=qubit_map,
            )
            self.partitions.append(partition)

        self._analyzed = True
        return self.partitions

    def extract_local_operations(self, qubits: set[int]) -> list[GateOperation]:
        local_ops = []
        for op in self.operations:
            if set(op.targets) <= qubits:
                local_ops.append(op)
        return local_ops

    def create_remapped_operations(
        self, partition: QubitPartition
    ) -> list[tuple[np.ndarray, tuple[int, ...]]]:
        remapped = []
        for op in partition.operations:
            new_targets = tuple(partition.qubit_map[q] for q in op.targets)
            remapped.append((op.matrix, new_targets))
        return remapped

    def simulate_partitioned(self, shots: int) -> dict[str, int]:
        from braket.default_simulator.product_simulator import ProductStateSimulator
        from braket.default_simulator.stabilizer_simulator import StabilizerSimulator

        partitions = self.partition()

        if len(partitions) == 1 and len(partitions[0].qubits) == self.n_qubits:
            return None

        partition_results = []

        for partition in partitions:
            n_local = len(partition.qubits)

            if partition.circuit_class == CircuitClass.PRODUCT:
                sim = ProductStateSimulator(n_local)
                for op in partition.operations:
                    new_target = partition.qubit_map[op.targets[0]]
                    sim.apply_gate(op.matrix, new_target)
                result = sim.sample(shots)

            elif partition.circuit_class == CircuitClass.CLIFFORD:
                sim = StabilizerSimulator(n_local)
                for op in partition.operations:
                    gate_type = getattr(op, "gate_type", "unknown")
                    new_targets = [partition.qubit_map[q] for q in op.targets]
                    sim.apply_gate(gate_type, new_targets)
                result = sim.sample(shots)

            else:
                from braket.default_simulator.simulation_strategies import (
                    single_operation_strategy,
                )

                state = np.zeros(2 ** n_local, dtype=np.complex128)
                state[0] = 1.0
                state_tensor = state.reshape([2] * n_local)

                class RemappedOp:
                    def __init__(self, matrix, targets, ctrl_state, gate_type):
                        self.matrix = matrix
                        self.targets = targets
                        self.control_state = ctrl_state
                        self.gate_type = gate_type

                remapped_ops = []
                for op in partition.operations:
                    new_targets = tuple(partition.qubit_map[q] for q in op.targets)
                    remapped_ops.append(RemappedOp(
                        op.matrix,
                        new_targets,
                        getattr(op, "control_state", ()),
                        getattr(op, "gate_type", None)
                    ))

                final_state = single_operation_strategy.apply_operations(
                    state_tensor, n_local, remapped_ops
                )
                probs = np.abs(final_state.flatten()) ** 2

                rng = np.random.default_rng()
                samples = rng.choice(2 ** n_local, size=shots, p=probs)
                counts = Counter(samples)
                result = {format(k, f"0{n_local}b"): v for k, v in counts.items()}

            partition_results.append((partition, result))

        return self.combine_partition_results(partition_results, shots)

    def combine_partition_results(
        self,
        partition_results: list[tuple[QubitPartition, dict[str, int]]],
        shots: int
    ) -> dict[str, int]:
        rng = np.random.default_rng()
        combined_results = []

        for _ in range(shots):
            full_bitstring = ["0"] * self.n_qubits

            for partition, results in partition_results:
                outcomes = list(results.keys())
                counts = list(results.values())
                probs = np.array(counts, dtype=float)
                probs /= probs.sum()

                chosen = rng.choice(outcomes, p=probs)

                sorted_qubits = sorted(partition.qubits)
                for i, qubit in enumerate(sorted_qubits):
                    full_bitstring[qubit] = chosen[i]

            combined_results.append("".join(full_bitstring))

        return dict(Counter(combined_results))
