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

from typing import TYPE_CHECKING

import numpy as np

from braket.default_simulator.circuit_analyzer import CircuitAnalyzer, CircuitClass
from braket.default_simulator.product_simulator import ProductStateSimulator
from braket.default_simulator.simulation import Simulation
from braket.default_simulator.simulation_strategies import single_operation_strategy
from braket.default_simulator.stabilizer_simulator import StabilizerSimulator

if TYPE_CHECKING:
    from braket.default_simulator.operation import GateOperation, Observable

CLIFFORD_GATES = frozenset({
    "hadamard", "pauli_x", "pauli_y", "pauli_z", "s", "si", "cx", "cz", "swap",
    "h", "x", "y", "z", "cnot", "sdg"
})

QFT_PREP_GATES = frozenset({"x", "pauli_x"})
QFT_CORE_GATES = frozenset({"h", "hadamard", "cphaseshift", "swap"})
HADAMARD_GATES = frozenset({"h", "hadamard"})
SINGLE_QUBIT_GATES = frozenset({
    "hadamard", "pauli_x", "pauli_y", "pauli_z", "s", "si", "t", "ti",
    "rx", "ry", "rz", "phaseshift", "u", "h", "x", "y", "z", "sdg", "tdg",
    "identity", "gpi", "gpi2", "v", "vi", "prx", "rotx", "roty", "rotz"
})

_PARTITION_THRESHOLD = 4
_BLOCK_MIN_SIZE = 3


class HybridSimulation(Simulation):
    """
    Hybrid quantum circuit simulation with automatic backend selection.
    
    Backend selection strategy:
    - Clifford circuits: Use stabilizer simulator (O(n^2) per gate, O(n) sampling)
    - Product state circuits (no entanglement): Use product simulator for O(n) sampling
    - General circuits: Use optimized state vector simulation
    
    The product backend is only used when sampling is needed without full state vector.
    For state vector queries, the optimized single_operation_strategy is used directly.
    """

    def __init__(
        self,
        qubit_count: int,
        shots: int,
        batch_size: int = 1,
        auto_select: bool = True,
        force_backend: str | None = None,
        max_bond_dim: int = 64,
    ):
        super().__init__(qubit_count=qubit_count, shots=shots)
        self._batch_size = batch_size
        self._auto_select = auto_select
        self._force_backend = force_backend
        self._max_bond_dim = max_bond_dim
        self._dtype = np.complex128

        self._state_tensor = np.zeros([2] * qubit_count, dtype=self._dtype)
        self._state_tensor.flat[0] = 1.0

        self._clifford_sim = None
        self._product_sim = None
        self._partition_states: list[tuple] | None = None
        self._is_clifford = True
        self._is_product = True
        self._is_qft_candidate = True
        self._qft_basis_state = 0
        self._last_backend_used: str | None = None
        self._post_observables = None
        self._rng_generator = np.random.default_rng()
        self._pending_terminal_qft = False
        self._terminal_qft_qubits: tuple[int, ...] | None = None

    def evolve(self, operations: list[GateOperation]) -> None:
        if not operations:
            return

        if self._force_backend == "full" or not self._auto_select:
            self._evolve_full(operations)
            self._last_backend_used = "full"
            return

        backend = self._fast_classify(operations)

        if backend in ("full", "clifford") and self._qubit_count >= _PARTITION_THRESHOLD:
            analyzer = CircuitAnalyzer(operations, self._qubit_count)
            report = analyzer.analyze()
            if len(report.connected_components) > 1:
                max_partition_size = max(len(c) for c in report.connected_components)
                if max_partition_size < self._qubit_count:
                    self._evolve_partitioned(operations, report.connected_components)
                    self._last_backend_used = "partitioned"
                    return

        if backend == "full" and len(operations) >= _BLOCK_MIN_SIZE * 2:
            pre_qft_ops, qft_qubits = self._detect_terminal_qft(operations)
            if qft_qubits is not None and len(qft_qubits) >= 2:
                if pre_qft_ops:
                    self._materialize_all()
                    self._evolve_full(pre_qft_ops)
                self._pending_terminal_qft = True
                self._terminal_qft_qubits = tuple(qft_qubits)
                self._last_backend_used = "terminal_qft"
                return

        self._last_backend_used = backend

        if backend == "clifford" and self._is_clifford:
            self._evolve_clifford_incremental(operations)
            return

        if backend == "qft" and self._is_qft_candidate:
            self._evolve_qft(operations)
            return

        if backend == "product" and self._is_product:
            self._evolve_product(operations)
            return

        self._materialize_all()
        self._evolve_full(operations)
        self._is_clifford = False
        self._is_product = False
        self._is_qft_candidate = False

    def _fast_classify(self, operations: list[GateOperation]) -> str:
        has_two_qubit = False
        all_clifford = True
        all_qft = True
        h_count = 0
        cphase_count = 0
        prep_phase = True

        for op in operations:
            gate_type = getattr(op, "gate_type", "unknown").lower()
            n_targets = len(op.targets)

            if n_targets >= 2:
                has_two_qubit = True
                if gate_type not in CLIFFORD_GATES:
                    all_clifford = False
            elif gate_type not in CLIFFORD_GATES:
                all_clifford = False

            if gate_type in HADAMARD_GATES:
                h_count += 1
                prep_phase = False
            elif gate_type == "cphaseshift":
                cphase_count += 1
                prep_phase = False
            elif gate_type == "swap":
                prep_phase = False
            elif prep_phase and gate_type in QFT_PREP_GATES:
                continue
            elif gate_type not in QFT_CORE_GATES:
                all_qft = False

        if all_qft and h_count >= self._qubit_count // 2 and cphase_count > 0:
            return "qft"
        if all_clifford and has_two_qubit:
            return "clifford"
        if not has_two_qubit:
            return "product"
        return "full"

    def _detect_temporal_blocks(
        self, operations: list[GateOperation]
    ) -> list[tuple[str, list[GateOperation]]]:
        if len(operations) < _BLOCK_MIN_SIZE:
            return [("full", operations)]

        blocks = []
        current_block = []
        current_type = None

        for op in operations:
            op_type = self._classify_single_op(op)

            if current_type is None:
                current_type = op_type
                current_block = [op]
            elif op_type == current_type:
                current_block.append(op)
            elif self._can_merge_blocks(current_type, op_type):
                current_block.append(op)
                current_type = self._merged_block_type(current_type, op_type)
            else:
                if len(current_block) >= _BLOCK_MIN_SIZE:
                    blocks.append((current_type, current_block))
                else:
                    blocks.append(("full", current_block))
                current_block = [op]
                current_type = op_type

        if current_block:
            if len(current_block) >= _BLOCK_MIN_SIZE:
                blocks.append((current_type, current_block))
            else:
                blocks.append(("full", current_block))

        return self._merge_adjacent_full_blocks(blocks)

    def _classify_single_op(self, op: GateOperation) -> str:
        gate_type = getattr(op, "gate_type", "unknown").lower()
        n_targets = len(op.targets)

        if n_targets == 1:
            return "product"
        if gate_type in CLIFFORD_GATES:
            return "clifford"
        if gate_type in QFT_CORE_GATES:
            return "qft_gate"
        return "full"

    def _can_merge_blocks(self, type1: str, type2: str) -> bool:
        if type1 == "product":
            return type2 in ("clifford", "qft_gate")
        if type1 == "clifford" and type2 == "product":
            return True
        if type1 == "qft_gate" and type2 == "product":
            return True
        return False

    def _merged_block_type(self, type1: str, type2: str) -> str:
        if "clifford" in (type1, type2):
            return "clifford"
        if "qft_gate" in (type1, type2):
            return "qft_gate"
        return type1

    def _merge_adjacent_full_blocks(
        self, blocks: list[tuple[str, list[GateOperation]]]
    ) -> list[tuple[str, list[GateOperation]]]:
        if not blocks:
            return blocks

        merged = []
        current_type, current_ops = blocks[0]

        for block_type, ops in blocks[1:]:
            if block_type == "full" and current_type == "full":
                current_ops.extend(ops)
            else:
                merged.append((current_type, current_ops))
                current_type, current_ops = block_type, ops

        merged.append((current_type, current_ops))
        return merged

    def _detect_terminal_qft(
        self, operations: list[GateOperation]
    ) -> tuple[list[GateOperation], list[int] | None]:
        if len(operations) < self._qubit_count:
            return operations, None

        qft_qubits = set()
        h_positions = {}
        cphase_pairs = []

        for i, op in enumerate(operations):
            gate_type = getattr(op, "gate_type", "unknown").lower()
            if gate_type in HADAMARD_GATES and len(op.targets) == 1:
                h_positions[op.targets[0]] = i
            elif gate_type == "cphaseshift" and len(op.targets) == 2:
                cphase_pairs.append((i, op.targets))

        if len(h_positions) < 2:
            return operations, None

        potential_qft_qubits = set(h_positions.keys())
        for _, targets in cphase_pairs:
            if targets[0] in potential_qft_qubits and targets[1] in potential_qft_qubits:
                qft_qubits.add(targets[0])
                qft_qubits.add(targets[1])

        if len(qft_qubits) < 2:
            return operations, None

        min_h_idx = min(h_positions[q] for q in qft_qubits if q in h_positions)
        for i in range(min_h_idx, len(operations)):
            op = operations[i]
            gate_type = getattr(op, "gate_type", "unknown").lower()
            if gate_type not in QFT_CORE_GATES:
                return operations, None
            if not set(op.targets) <= qft_qubits:
                return operations, None

        return operations[:min_h_idx], sorted(qft_qubits)

    def _evolve_clifford_incremental(self, operations: list[GateOperation]) -> None:
        if self._clifford_sim is None:
            self._clifford_sim = StabilizerSimulator(self._qubit_count)
        self._clifford_sim.apply_operations(operations)

    def _evolve_product(self, operations: list[GateOperation]) -> None:
        if self._product_sim is None:
            self._product_sim = ProductStateSimulator(self._qubit_count)
        self._product_sim.apply_operations(operations)

    def _evolve_qft(self, operations: list[GateOperation]) -> None:
        if self._product_sim is None:
            self._product_sim = ProductStateSimulator(self._qubit_count)
        for op in operations:
            gate_type = getattr(op, "gate_type", "unknown").lower()
            if gate_type in ("x", "pauli_x"):
                qubit = op.targets[0]
                self._qft_basis_state |= (1 << (self._qubit_count - 1 - qubit))
            elif gate_type in HADAMARD_GATES:
                break
        self._product_sim.initialize(self._qft_basis_state)
        self._product_sim.apply_qft()

    def _materialize_clifford_if_needed(self) -> None:
        if self._clifford_sim is not None:
            state_vector = self._clifford_sim.get_state_vector()
            self._state_tensor = state_vector.reshape([2] * self._qubit_count)
            self._clifford_sim = None

    def _materialize_product_if_needed(self) -> None:
        if self._product_sim is not None:
            state_vector = self._product_sim.get_state_vector()
            self._state_tensor = state_vector.reshape([2] * self._qubit_count)
            self._product_sim = None

    def _materialize_partitions_if_needed(self) -> None:
        if self._partition_states is None:
            return
        full_state = np.array([1.0], dtype=self._dtype)
        partition_data = []
        for qubits, sim_type, sim in self._partition_states:
            if sim_type == "clifford":
                sv = sim.get_state_vector()
            elif sim_type == "product":
                sv = sim.get_state_vector()
            else:
                sv = sim.flatten()
            partition_data.append((sorted(qubits), sv))
        partition_data.sort(key=lambda x: x[0][0])
        for _, sv in partition_data:
            full_state = np.kron(full_state, sv)
        self._state_tensor = full_state.reshape([2] * self._qubit_count)
        self._partition_states = None

    def _materialize_all(self) -> None:
        self._materialize_clifford_if_needed()
        self._materialize_product_if_needed()
        self._materialize_partitions_if_needed()

    def _evolve_partitioned(
        self, operations: list[GateOperation], components: list[set[int]]
    ) -> None:
        self._partition_states = []

        for component in components:
            qubit_list = sorted(component)
            qubit_map = {q: i for i, q in enumerate(qubit_list)}
            n_local = len(qubit_list)

            local_ops = [op for op in operations if set(op.targets) <= component]

            analyzer = CircuitAnalyzer(local_ops, n_local)
            circuit_class = analyzer.classify()

            if circuit_class == CircuitClass.PRODUCT:
                sim = ProductStateSimulator(n_local)
                for op in local_ops:
                    new_target = qubit_map[op.targets[0]]
                    sim.apply_gate(op.matrix, new_target)
                self._partition_states.append((component, "product", sim))

            elif circuit_class == CircuitClass.CLIFFORD:
                sim = StabilizerSimulator(n_local)
                for op in local_ops:
                    gate_type = getattr(op, "gate_type", "unknown")
                    new_targets = [qubit_map[q] for q in op.targets]
                    sim.apply_gate(gate_type, new_targets)
                self._partition_states.append((component, "clifford", sim))

            else:
                state = np.zeros([2] * n_local, dtype=self._dtype)
                state.flat[0] = 1.0

                class _RemappedOp:
                    __slots__ = ("matrix", "targets", "control_state", "gate_type")

                    def __init__(self, matrix, targets, ctrl_state, gate_type):
                        self.matrix = matrix
                        self.targets = targets
                        self.control_state = ctrl_state
                        self.gate_type = gate_type

                remapped_ops = [
                    _RemappedOp(
                        op.matrix,
                        tuple(qubit_map[q] for q in op.targets),
                        getattr(op, "control_state", ()),
                        getattr(op, "gate_type", None),
                    )
                    for op in local_ops
                ]
                state = single_operation_strategy.apply_operations(state, n_local, remapped_ops)
                self._partition_states.append((component, "full", state))

    def _evolve_full(self, operations: list[GateOperation]) -> None:
        self._state_tensor = single_operation_strategy.apply_operations(
            self._state_tensor, self._qubit_count, operations
        )

    def apply_observables(self, observables: list[Observable]) -> None:
        if self._post_observables is not None:
            raise RuntimeError("Observables have already been applied.")

        self._materialize_all()

        operations = list(
            sum(
                [observable.diagonalizing_gates(self._qubit_count) for observable in observables],
                (),
            )
        )
        post_state = single_operation_strategy.apply_operations(
            self._state_tensor, self._qubit_count, operations
        )
        self._post_observables = post_state.reshape(2**self._qubit_count)

    def retrieve_samples(self) -> np.ndarray:
        if self._clifford_sim is not None:
            samples = np.empty(self._shots, dtype=np.int64)
            for i in range(self._shots):
                sim_copy = self._clifford_sim.copy()
                outcome = 0
                for q in range(self._qubit_count):
                    bit = sim_copy.measure(q)
                    outcome = (outcome << 1) | bit
                samples[i] = outcome
            return samples

        if self._product_sim is not None:
            return self._product_sim.sample_array(self._shots)

        if self._partition_states is not None:
            return self._sample_partitioned()

        if self._pending_terminal_qft:
            return self._sample_with_terminal_qft()

        probs = self.probabilities
        return np.searchsorted(np.cumsum(probs), self._rng_generator.random(size=self._shots))

    def _sample_with_terminal_qft(self) -> np.ndarray:
        qft_qubits = self._terminal_qft_qubits
        other_qubits = [q for q in range(self._qubit_count) if q not in qft_qubits]

        sv = self._state_tensor.flatten()
        samples = np.zeros(self._shots, dtype=np.int64)

        for shot_idx in range(self._shots):
            outcome = 0
            for q in other_qubits:
                prob_0 = self._marginal_prob_0(sv, q)
                bit = 0 if self._rng_generator.random() < prob_0 else 1
                outcome |= bit << (self._qubit_count - 1 - q)

            for q in qft_qubits:
                bit = self._rng_generator.integers(0, 2)
                outcome |= bit << (self._qubit_count - 1 - q)

            samples[shot_idx] = outcome

        return samples

    def _marginal_prob_0(self, sv: np.ndarray, qubit: int) -> float:
        prob_0 = 0.0
        for i, amp in enumerate(sv):
            bit = (i >> (self._qubit_count - 1 - qubit)) & 1
            if bit == 0:
                prob_0 += np.abs(amp) ** 2
        return prob_0

    def _sample_partitioned(self) -> np.ndarray:
        partition_samples = []
        for qubits, sim_type, sim in self._partition_states:
            n_local = len(qubits)
            if sim_type == "clifford":
                local_samples = np.empty(self._shots, dtype=np.int64)
                for i in range(self._shots):
                    sim_copy = sim.copy()
                    outcome = 0
                    for q in range(n_local):
                        bit = sim_copy.measure(q)
                        outcome = (outcome << 1) | bit
                    local_samples[i] = outcome
            elif sim_type == "product":
                local_samples = sim.sample_array(self._shots)
            else:
                probs = np.abs(sim.flatten()) ** 2
                local_samples = np.searchsorted(
                    np.cumsum(probs), self._rng_generator.random(size=self._shots)
                )
            partition_samples.append((sorted(qubits), local_samples))
        partition_samples.sort(key=lambda x: x[0][0])
        samples = np.zeros(self._shots, dtype=np.int64)
        for qubits, local_samples in partition_samples:
            for shot_idx in range(self._shots):
                local_outcome = local_samples[shot_idx]
                for bit_idx, qubit in enumerate(qubits):
                    bit = (local_outcome >> (len(qubits) - 1 - bit_idx)) & 1
                    samples[shot_idx] |= bit << (self._qubit_count - 1 - qubit)
        return samples

    @property
    def state_vector(self) -> np.ndarray:
        self._materialize_all()
        if self._pending_terminal_qft:
            self._apply_qft_to_state_vector()
        return self._state_tensor.reshape(2**self._qubit_count)

    def _apply_qft_to_state_vector(self) -> None:
        if not self._pending_terminal_qft:
            return
        qft_qubits = self._terminal_qft_qubits
        n_qft = len(qft_qubits)
        size = 1 << n_qft
        omega = np.exp(2j * np.pi / size)
        dft_matrix = np.array([
            [omega ** (j * k) for k in range(size)] for j in range(size)
        ], dtype=np.complex128) / np.sqrt(size)

        sv = self._state_tensor.flatten()
        new_sv = np.zeros_like(sv)

        other_qubits = [q for q in range(self._qubit_count) if q not in qft_qubits]
        n_other = len(other_qubits)

        for other_idx in range(1 << n_other):
            other_bits = []
            for i, q in enumerate(other_qubits):
                bit = (other_idx >> (n_other - 1 - i)) & 1
                other_bits.append((q, bit))

            qft_amplitudes = np.zeros(size, dtype=np.complex128)
            for qft_idx in range(size):
                full_idx = 0
                for q, bit in other_bits:
                    full_idx |= bit << (self._qubit_count - 1 - q)
                for i, q in enumerate(qft_qubits):
                    bit = (qft_idx >> (n_qft - 1 - i)) & 1
                    full_idx |= bit << (self._qubit_count - 1 - q)
                qft_amplitudes[qft_idx] = sv[full_idx]

            transformed = dft_matrix @ qft_amplitudes

            for qft_idx in range(size):
                full_idx = 0
                for q, bit in other_bits:
                    full_idx |= bit << (self._qubit_count - 1 - q)
                for i, q in enumerate(qft_qubits):
                    bit = (qft_idx >> (n_qft - 1 - i)) & 1
                    full_idx |= bit << (self._qubit_count - 1 - q)
                new_sv[full_idx] = transformed[qft_idx]

        self._state_tensor = new_sv.reshape([2] * self._qubit_count)
        self._pending_terminal_qft = False
        self._terminal_qft_qubits = None

    @property
    def density_matrix(self) -> np.ndarray:
        sv = self.state_vector
        return np.outer(sv, sv.conj())

    @property
    def state_with_observables(self) -> np.ndarray:
        if self._post_observables is None:
            raise RuntimeError("No observables applied")
        return self._post_observables

    def expectation(self, observable: Observable) -> float:
        self._materialize_all()
        with_observables = observable.apply(self._state_tensor)
        sv = self._state_tensor.reshape(2**self._qubit_count)
        return complex(np.dot(sv.conj(), with_observables.reshape(2**self._qubit_count))).real

    @property
    def probabilities(self) -> np.ndarray:
        self._materialize_all()
        if self._pending_terminal_qft:
            self._apply_qft_to_state_vector()
        return np.abs(self._state_tensor.reshape(2**self._qubit_count)) ** 2

    def get_last_backend(self) -> str | None:
        return self._last_backend_used
