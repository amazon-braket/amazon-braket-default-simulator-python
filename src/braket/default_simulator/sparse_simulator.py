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

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from braket.default_simulator.operation import GateOperation

_SPARSITY_THRESHOLD = 0.1
_PRUNE_THRESHOLD = 1e-14
_MAX_AMPLITUDES = 1 << 20


class SparseStateSimulator:
    __slots__ = ("n_qubits", "dtype", "amplitudes", "_rng")

    def __init__(self, n_qubits: int, dtype=np.complex128):
        self.n_qubits = n_qubits
        self.dtype = dtype
        self.amplitudes: dict[int, complex] = {0: 1.0 + 0j}
        self._rng = np.random.default_rng()

    def initialize(self, basis_state: int) -> None:
        self.amplitudes = {basis_state: 1.0 + 0j}

    def sparsity(self) -> float:
        return len(self.amplitudes) / (2**self.n_qubits)

    def is_sparse(self) -> bool:
        return self.sparsity() < _SPARSITY_THRESHOLD

    def apply_single_qubit_gate(self, gate: np.ndarray, qubit: int) -> None:
        if gate.shape != (2, 2):
            raise ValueError("Single-qubit gate must be 2x2")

        bit_mask = 1 << (self.n_qubits - 1 - qubit)
        new_amplitudes: dict[int, complex] = {}

        processed = set()
        for basis in list(self.amplitudes.keys()):
            if basis in processed:
                continue

            partner = basis ^ bit_mask
            processed.add(basis)
            processed.add(partner)

            amp0 = self.amplitudes.get(basis if (basis & bit_mask) == 0 else partner, 0j)
            amp1 = self.amplitudes.get(basis if (basis & bit_mask) != 0 else partner, 0j)

            basis0 = basis & ~bit_mask
            basis1 = basis | bit_mask

            new_amp0 = gate[0, 0] * amp0 + gate[0, 1] * amp1
            new_amp1 = gate[1, 0] * amp0 + gate[1, 1] * amp1

            if abs(new_amp0) > _PRUNE_THRESHOLD:
                new_amplitudes[basis0] = new_amp0
            if abs(new_amp1) > _PRUNE_THRESHOLD:
                new_amplitudes[basis1] = new_amp1

        self.amplitudes = new_amplitudes

    def apply_two_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int) -> None:
        if gate.shape != (4, 4):
            raise ValueError("Two-qubit gate must be 4x4")

        bit1 = 1 << (self.n_qubits - 1 - qubit1)
        bit2 = 1 << (self.n_qubits - 1 - qubit2)
        combined_mask = bit1 | bit2

        new_amplitudes: dict[int, complex] = {}
        processed = set()

        for basis in list(self.amplitudes.keys()):
            base = basis & ~combined_mask
            if base in processed:
                continue
            processed.add(base)

            states = [base, base | bit2, base | bit1, base | bit1 | bit2]
            amps = np.array([self.amplitudes.get(s, 0j) for s in states], dtype=self.dtype)

            new_amps = gate @ amps

            for idx, s in enumerate(states):
                if abs(new_amps[idx]) > _PRUNE_THRESHOLD:
                    new_amplitudes[s] = new_amps[idx]

        self.amplitudes = new_amplitudes
        self._check_density()

    def _check_density(self) -> None:
        if len(self.amplitudes) > _MAX_AMPLITUDES:
            raise ValueError(
                f"State too dense for sparse simulation ({len(self.amplitudes)} amplitudes)"
            )

    def apply_operations(self, operations: list[GateOperation]) -> None:
        # Apply single-qubit fusion to reduce gate count
        from braket.default_simulator.gate_fusion import fuse_single_qubit_gates

        if len(operations) >= 4:
            operations = fuse_single_qubit_gates(operations)

        for op in operations:
            n_targets = len(op.targets)
            if n_targets == 1:
                self.apply_single_qubit_gate(op.matrix, op.targets[0])
            elif n_targets == 2:
                self.apply_two_qubit_gate(op.matrix, op.targets[0], op.targets[1])
            else:
                self._apply_multi_qubit_gate(op.matrix, op.targets)

    def _apply_multi_qubit_gate(self, gate: np.ndarray, targets: tuple[int, ...]) -> None:
        n_targets = len(targets)
        gate_dim = 2**n_targets

        if gate.shape != (gate_dim, gate_dim):
            raise ValueError(f"Gate dimension mismatch: expected {gate_dim}x{gate_dim}")

        bit_masks = [1 << (self.n_qubits - 1 - q) for q in targets]
        combined_mask = sum(bit_masks)

        new_amplitudes: dict[int, complex] = {}
        processed = set()

        for basis in list(self.amplitudes.keys()):
            base = basis & ~combined_mask
            if base in processed:
                continue
            processed.add(base)

            states = []
            for i in range(gate_dim):
                state = base
                for j, mask in enumerate(bit_masks):
                    if (i >> (n_targets - 1 - j)) & 1:
                        state |= mask
                states.append(state)

            amps = np.array([self.amplitudes.get(s, 0j) for s in states], dtype=self.dtype)
            new_amps = gate @ amps

            for idx, s in enumerate(states):
                if abs(new_amps[idx]) > _PRUNE_THRESHOLD:
                    new_amplitudes[s] = new_amps[idx]

        self.amplitudes = new_amplitudes
        self._check_density()

    def get_amplitude(self, basis_state: int) -> complex:
        return self.amplitudes.get(basis_state, 0j)

    def get_probabilities(self) -> np.ndarray:
        dim = 2**self.n_qubits
        probs = np.zeros(dim, dtype=np.float64)
        for basis, amp in self.amplitudes.items():
            probs[basis] = abs(amp) ** 2
        return probs

    def sample(self, shots: int) -> dict[str, int]:
        bases = list(self.amplitudes.keys())
        probs = np.array([abs(self.amplitudes[b]) ** 2 for b in bases])
        probs /= probs.sum()

        samples = self._rng.choice(bases, size=shots, p=probs)
        counts = Counter(samples)
        return {format(int(k), f"0{self.n_qubits}b"): int(v) for k, v in counts.items()}

    def sample_array(self, shots: int) -> np.ndarray:
        bases = np.array(list(self.amplitudes.keys()), dtype=np.int64)
        probs = np.array([abs(self.amplitudes[b]) ** 2 for b in self.amplitudes.keys()])
        probs /= probs.sum()
        return self._rng.choice(bases, size=shots, p=probs)

    def get_state_vector(self) -> np.ndarray:
        if self.n_qubits > 26:
            raise ValueError("State vector too large")
        dim = 2**self.n_qubits
        state = np.zeros(dim, dtype=self.dtype)
        for basis, amp in self.amplitudes.items():
            state[basis] = amp
        return state

    def normalize(self) -> None:
        norm_sq = sum(abs(amp) ** 2 for amp in self.amplitudes.values())
        # this is a defensive check to avoid div by 0. I was not able to make a test case...
        if norm_sq > 0:  # pragma: no cover
            norm = np.sqrt(norm_sq)
            self.amplitudes = {k: v / norm for k, v in self.amplitudes.items()}

    def prune(self, threshold: float = _PRUNE_THRESHOLD) -> None:
        self.amplitudes = {k: v for k, v in self.amplitudes.items() if abs(v) > threshold}
        if not self.amplitudes:
            self.amplitudes = {0: 1.0 + 0j}

    def copy(self) -> SparseStateSimulator:
        new_sim = SparseStateSimulator(self.n_qubits, self.dtype)
        new_sim.amplitudes = self.amplitudes.copy()
        return new_sim

    @staticmethod
    def from_dense(state: np.ndarray, n_qubits: int) -> SparseStateSimulator:
        sim = SparseStateSimulator(n_qubits)
        sim.amplitudes = {}
        for i, amp in enumerate(state.flatten()):
            if abs(amp) > _PRUNE_THRESHOLD:
                sim.amplitudes[i] = complex(amp)
        if not sim.amplitudes:
            sim.amplitudes = {0: 1.0 + 0j}
        return sim
