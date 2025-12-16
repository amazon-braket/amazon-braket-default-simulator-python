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

from braket.default_simulator.mps_simulator import MPSSimulator
from braket.default_simulator.product_simulator import ProductStateSimulator
from braket.default_simulator.simulation import Simulation
from braket.default_simulator.simulation_strategies import single_operation_strategy
from braket.default_simulator.sparse_simulator import SparseStateSimulator
from braket.default_simulator.stabilizer_simulator import StabilizerSimulator

if TYPE_CHECKING:
    from braket.default_simulator.operation import GateOperation, Observable

CLIFFORD_GATES = frozenset(
    {
        "hadamard",
        "pauli_x",
        "pauli_y",
        "pauli_z",
        "s",
        "si",
        "cx",
        "cz",
        "swap",
        "h",
        "x",
        "y",
        "z",
        "cnot",
        "sdg",
    }
)

QFT_GATES = frozenset({"x", "pauli_x", "h", "hadamard", "cphaseshift", "swap"})
HADAMARD_GATES = frozenset({"h", "hadamard"})
MPS_GATES = frozenset(
    {
        "h",
        "hadamard",
        "x",
        "y",
        "z",
        "pauli_x",
        "pauli_y",
        "pauli_z",
        "s",
        "si",
        "t",
        "ti",
        "rx",
        "ry",
        "rz",
        "rotx",
        "roty",
        "rotz",
        "cx",
        "cnot",
        "cz",
        "swap",
        "cphaseshift",
        "phaseshift",
        "u",
        "gphase",
        "identity",
        "i",
    }
)
SPARSE_FRIENDLY_GATES = frozenset(
    {
        "x",
        "pauli_x",
        "cx",
        "cnot",
        "ccx",
        "toffoli",
        "swap",
        "cswap",
        "h",
        "hadamard",
        "y",
        "pauli_y",
    }
)

_PARTITION_THRESHOLD = 4
_FAST_PATH_GATE_THRESHOLD = 50
_MPS_BOND_THRESHOLD = 64
_MPS_QUBIT_THRESHOLD = 8
_SPARSE_QUBIT_THRESHOLD = 12
_SPARSE_GATE_RATIO_THRESHOLD = 0.7


class _RemappedOp:
    __slots__ = ("matrix", "targets", "control_state", "gate_type")

    def __init__(self, matrix, targets, ctrl_state, gate_type):
        self.matrix = matrix
        self.targets = targets
        self.control_state = ctrl_state
        self.gate_type = gate_type


def _remap_ops(local_ops, qubit_map):
    return [
        _RemappedOp(
            op.matrix,
            tuple(qubit_map[q] for q in op.targets),
            getattr(op, "control_state", ()),
            getattr(op, "gate_type", None),
        )
        for op in local_ops
    ]


def _simulate_partition(args):
    component, local_ops, qubit_map, n_local, backend, max_bond_dim, dtype = args

    if backend == "product":
        sim = ProductStateSimulator(n_local)
        for op in local_ops:
            sim.apply_gate(op.matrix, qubit_map[op.targets[0]])
        return (component, "product", sim)

    if backend == "clifford":
        sim = StabilizerSimulator(n_local)
        for op in local_ops:
            gate_type = getattr(op, "gate_type", "unknown")
            new_targets = [qubit_map[q] for q in op.targets]
            sim.apply_gate(gate_type, new_targets)
        return (component, "clifford", sim)

    if backend == "qft":
        sim = ProductStateSimulator(n_local)
        basis_state = 0
        for op in local_ops:
            gate_type = getattr(op, "gate_type", "unknown").lower()
            if gate_type in ("x", "pauli_x"):
                local_qubit = qubit_map[op.targets[0]]
                basis_state |= 1 << (n_local - 1 - local_qubit)
        sim.initialize(basis_state)
        sim.apply_qft()
        return (component, "product", sim)

    if backend == "mps":
        sim = MPSSimulator(n_local, max_bond_dim)
        sim.apply_operations(_remap_ops(local_ops, qubit_map))
        return (component, "mps", sim)

    if backend == "sparse":
        sim = SparseStateSimulator(n_local)
        sim.apply_operations(_remap_ops(local_ops, qubit_map))
        return (component, "sparse", sim)

    state = np.zeros([2] * n_local, dtype=dtype)
    state.flat[0] = 1.0
    state = single_operation_strategy.apply_operations(
        state, n_local, _remap_ops(local_ops, qubit_map)
    )
    return (component, "full", state)


class HybridSimulation(Simulation):
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
        self._mps_sim = None
        self._sparse_sim = None
        self._partition_states: list[tuple] | None = None
        self._is_clifford = True
        self._is_product = True
        self._is_qft_candidate = True
        self._is_mps_candidate = True
        self._is_sparse_candidate = True
        self._qft_basis_state = 0
        self._last_backend_used: str | None = None
        self._post_observables = None
        self._rng_generator = np.random.default_rng()

    def evolve(self, operations: list[GateOperation]) -> None:
        if not operations:
            return

        if self._force_backend == "full" or not self._auto_select:
            self._evolve_full(operations)
            self._last_backend_used = "full"
            return

        backend = self._classify(operations)

        if self._qubit_count >= _PARTITION_THRESHOLD and backend != "qft" and backend != "product":
            components = self._get_connected_components(operations)
            if len(components) > 1 and max(len(c) for c in components) < self._qubit_count:
                self._evolve_partitioned(operations, components)
                self._last_backend_used = "partitioned"
                return

        self._last_backend_used = backend
        self._dispatch_backend(backend, operations)

    def _dispatch_backend(self, backend: str, operations: list[GateOperation]) -> None:
        dispatch = {
            "clifford": (self._is_clifford, self._evolve_clifford),
            "qft": (self._is_qft_candidate, self._evolve_qft),
            "product": (self._is_product, self._evolve_product),
            "mps": (self._is_mps_candidate, self._evolve_mps),
            "sparse": (self._is_sparse_candidate, self._evolve_sparse),
        }
        if backend in dispatch:
            is_valid, evolve_fn = dispatch[backend]
            if is_valid:
                evolve_fn(operations)
                return
        self._materialize_all()
        self._evolve_full(operations)
        self._invalidate_specialized_backends()

    def _classify(self, operations: list[GateOperation]) -> str:
        return self._classify_ops(operations, self._qubit_count)

    def _get_connected_components(self, operations: list[GateOperation]) -> list[set[int]]:
        max_qubit = max(max(op.targets) for op in operations if op.targets)
        n = max(self._qubit_count, max_qubit + 1)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for op in operations:
            targets = op.targets
            if len(targets) >= 2:
                for i in range(1, len(targets)):
                    union(targets[0], targets[i])

        components_map = {}
        for i in range(n):
            root = find(i)
            if root not in components_map:
                components_map[root] = set()
            components_map[root].add(i)

        return list(components_map.values())

    def _invalidate_specialized_backends(self) -> None:
        self._is_clifford = False
        self._is_product = False
        self._is_qft_candidate = False
        self._is_mps_candidate = False
        self._is_sparse_candidate = False

    def _evolve_clifford(self, operations: list[GateOperation]) -> None:
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
                self._qft_basis_state |= 1 << (self._qubit_count - 1 - qubit)
        self._product_sim.initialize(self._qft_basis_state)
        self._product_sim.apply_qft()

    def _evolve_mps(self, operations: list[GateOperation]) -> None:
        if self._mps_sim is None:
            self._mps_sim = MPSSimulator(self._qubit_count, self._max_bond_dim)
        self._mps_sim.apply_operations(operations)
        bond_dims = self._mps_sim.get_bond_dimensions()
        if bond_dims and max(bond_dims) > _MPS_BOND_THRESHOLD:
            self._materialize_mps_if_needed()
            self._is_mps_candidate = False

    def _evolve_sparse(self, operations: list[GateOperation]) -> None:
        if self._sparse_sim is None:
            self._sparse_sim = SparseStateSimulator(self._qubit_count)
        self._sparse_sim.apply_operations(operations)
        if not self._sparse_sim.is_sparse():
            self._materialize_sparse_if_needed()
            self._is_sparse_candidate = False

    def _evolve_full(self, operations: list[GateOperation]) -> None:
        self._state_tensor = single_operation_strategy.apply_operations(
            self._state_tensor, self._qubit_count, operations
        )

    def _evolve_partitioned(
        self, operations: list[GateOperation], components: list[set[int]]
    ) -> None:
        self._partition_states = []
        for component in components:
            qubit_list = sorted(component)
            qubit_map = {q: i for i, q in enumerate(qubit_list)}
            n_local = len(qubit_list)
            local_ops = [op for op in operations if set(op.targets) <= component]
            backend = self._classify_ops(local_ops, n_local)

            args = (component, local_ops, qubit_map, n_local, backend, self._max_bond_dim, self._dtype)
            self._partition_states.append(_simulate_partition(args))

    def _classify_ops(self, operations: list[GateOperation], n_qubits: int) -> str:
        if not operations:
            return "product"

        has_two_qubit = False
        all_clifford = True
        all_qft = True
        all_mps = True
        sparse_friendly_count = 0
        h_count = 0
        cphase_count = 0
        two_qubit_count = 0
        max_gate_distance = 0

        for op in operations:
            gate_type = getattr(op, "gate_type", "unknown").lower()
            n_targets = len(op.targets)

            sparse_friendly_count += gate_type in SPARSE_FRIENDLY_GATES
            all_clifford = all_clifford and gate_type in CLIFFORD_GATES
            all_mps = all_mps and gate_type in MPS_GATES
            all_qft = all_qft and gate_type in QFT_GATES

            if gate_type in HADAMARD_GATES:
                h_count += 1
            elif gate_type == "cphaseshift":
                cphase_count += 1

            if n_targets >= 2:
                has_two_qubit = True
                two_qubit_count += 1
                max_gate_distance = max(max_gate_distance, abs(op.targets[0] - op.targets[-1]))
                all_mps = all_mps and n_targets <= 2

        if all_qft and h_count >= n_qubits // 2 and cphase_count > 0:
            return "qft"
        if all_clifford and has_two_qubit:
            return "clifford"
        if not has_two_qubit:
            return "product"
        if (
            all_mps
            and n_qubits >= _MPS_QUBIT_THRESHOLD
            and max_gate_distance <= 2
            and two_qubit_count <= n_qubits * 4
        ):
            return "mps"
        if (
            n_qubits >= _SPARSE_QUBIT_THRESHOLD
            and sparse_friendly_count >= len(operations) * _SPARSE_GATE_RATIO_THRESHOLD
        ):
            return "sparse"
        return "full"

    def _materialize_clifford_if_needed(self) -> None:
        if self._clifford_sim is not None:
            self._state_tensor = self._clifford_sim.get_state_vector().reshape(
                [2] * self._qubit_count
            )
            self._clifford_sim = None

    def _materialize_product_if_needed(self) -> None:
        if self._product_sim is not None:
            self._state_tensor = self._product_sim.get_state_vector().reshape(
                [2] * self._qubit_count
            )
            self._product_sim = None

    def _materialize_mps_if_needed(self) -> None:
        if self._mps_sim is not None:
            self._state_tensor = self._mps_sim.get_state_vector().reshape([2] * self._qubit_count)
            self._mps_sim = None

    def _materialize_sparse_if_needed(self) -> None:
        if self._sparse_sim is not None:
            self._state_tensor = self._sparse_sim.get_state_vector().reshape(
                [2] * self._qubit_count
            )
            self._sparse_sim = None

    def _materialize_partitions_if_needed(self) -> None:
        if self._partition_states is None:
            return

        partition_data = []
        for qubits, sim_type, sim in self._partition_states:
            if sim_type in ("clifford", "product", "mps", "sparse"):
                sv = sim.get_state_vector()
            else:
                sv = sim.flatten()
            partition_data.append((sorted(qubits), sv))

        partition_data.sort(key=lambda x: x[0][0])
        full_state = np.array([1.0], dtype=self._dtype)
        for _, sv in partition_data:
            full_state = np.kron(full_state, sv)

        self._state_tensor = full_state.reshape([2] * self._qubit_count)
        self._partition_states = None

    def _materialize_all(self) -> None:
        self._materialize_clifford_if_needed()
        self._materialize_product_if_needed()
        self._materialize_mps_if_needed()
        self._materialize_sparse_if_needed()
        self._materialize_partitions_if_needed()

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
            return self._sample_clifford()
        if self._product_sim is not None:
            return self._product_sim.sample_array(self._shots)
        if self._mps_sim is not None:
            return self._sample_mps()
        if self._sparse_sim is not None:
            return self._sparse_sim.sample_array(self._shots)
        if self._partition_states is not None:
            return self._sample_partitioned()
        return self._sample_from_probabilities()

    def _sample_clifford(self) -> np.ndarray:
        samples = np.empty(self._shots, dtype=np.int64)
        for i in range(self._shots):
            sim_copy = self._clifford_sim.copy()
            outcome = 0
            for q in range(self._qubit_count):
                bit = sim_copy.measure(q)
                outcome = (outcome << 1) | bit
            samples[i] = outcome
        return samples

    def _sample_mps(self) -> np.ndarray:
        result_dict = self._mps_sim.sample(self._shots)
        samples = []
        for bitstring, count in result_dict.items():
            val = int(bitstring, 2)
            samples.extend([val] * count)
        return np.array(samples, dtype=np.int64)

    def _sample_partition_sim(self, sim_type, sim, n_local) -> np.ndarray:
        if sim_type == "clifford":
            local_samples = np.empty(self._shots, dtype=np.int64)
            for i in range(self._shots):
                sim_copy = sim.copy()
                outcome = 0
                for q in range(n_local):
                    bit = sim_copy.measure(q)
                    outcome = (outcome << 1) | bit
                local_samples[i] = outcome
            return local_samples
        if sim_type in ("product", "sparse"):
            return sim.sample_array(self._shots)
        if sim_type == "mps":
            result_dict = sim.sample(self._shots)
            local_list = []
            for bitstring, count in result_dict.items():
                local_list.extend([int(bitstring, 2)] * count)
            return np.array(local_list, dtype=np.int64)
        probs = np.abs(sim.flatten()) ** 2
        return np.searchsorted(np.cumsum(probs), self._rng_generator.random(size=self._shots))

    def _sample_partitioned(self) -> np.ndarray:
        partition_samples = []
        for qubits, sim_type, sim in self._partition_states:
            local_samples = self._sample_partition_sim(sim_type, sim, len(qubits))
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

    def _sample_from_probabilities(self) -> np.ndarray:
        probs = self.probabilities
        return np.searchsorted(np.cumsum(probs), self._rng_generator.random(size=self._shots))

    @property
    def state_vector(self) -> np.ndarray:
        self._materialize_all()
        return self._state_tensor.reshape(2**self._qubit_count)

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
        return np.abs(self._state_tensor.reshape(2**self._qubit_count)) ** 2

    def get_last_backend(self) -> str | None:
        return self._last_backend_used
