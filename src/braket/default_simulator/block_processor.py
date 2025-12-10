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
block_processor.py

Block-based quantum state processing for structured circuits.

This module provides tools for processing quantum states that have
block-diagonal structure, enabling efficient simulation of certain
controlled operations.

Mathematical Basis:
- Block-diagonal unitary decomposition
- Controlled operation structure exploitation
- Sparse block amplitude tracking

Dependencies:
- numpy
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from braket.default_simulator.circuit_analyzer import CircuitClass

if TYPE_CHECKING:
    from braket.default_simulator.operation import GateOperation


@dataclass
class BlockStructure:
    control_qubits: list[int]
    target_qubits: list[int]
    active_blocks: set[int] = field(default_factory=set)
    block_types: dict[int, CircuitClass] = field(default_factory=dict)


class BlockMatrixProcessor:
    """
    Processes quantum states with block-diagonal structure.

    This processor exploits the structure of controlled operations
    to simulate circuits more efficiently.
    """

    def __init__(self, n_qubits: int, control_qubits: int):
        self.n_qubits = n_qubits
        self.control_qubits = control_qubits
        self.target_qubits = n_qubits - control_qubits
        self.block_size = 2 ** self.target_qubits
        self.n_blocks = 2 ** control_qubits
        self.blocks: dict[int, np.ndarray] = {}
        self.block_amplitudes: dict[int, complex] = {}
        self._dtype = np.complex128

    def identify_block_structure(
        self, operations: list[GateOperation]
    ) -> BlockStructure | None:
        if not operations:
            return None

        control_candidates = set(range(self.control_qubits))
        target_candidates = set(range(self.control_qubits, self.n_qubits))

        for op in operations:
            targets = set(op.targets)
            if len(targets) >= 2:
                if targets & control_candidates and targets & target_candidates:
                    return None

        return BlockStructure(
            control_qubits=list(range(self.control_qubits)),
            target_qubits=list(range(self.control_qubits, self.n_qubits)),
            active_blocks={0},
            block_types={0: CircuitClass.GENERAL},
        )

    def initialize_blocks(self, block_types: dict[int, CircuitClass]) -> None:
        self.blocks = {}
        self.block_amplitudes = {}

        for block_idx, block_class in block_types.items():
            state = np.zeros(self.block_size, dtype=self._dtype)
            state[0] = 1.0
            self.blocks[block_idx] = state
            self.block_amplitudes[block_idx] = 1.0 if block_idx == 0 else 0.0

    def set_block_state(self, block_idx: int, state: np.ndarray) -> None:
        if state.shape[0] != self.block_size:
            raise ValueError(
                f"State size {state.shape[0]} doesn't match block size {self.block_size}"
            )
        self.blocks[block_idx] = state.astype(self._dtype)

    def set_block_amplitude(self, block_idx: int, amplitude: complex) -> None:
        self.block_amplitudes[block_idx] = amplitude

    def apply_block_local_operation(
        self, block_idx: int, matrix: np.ndarray
    ) -> None:
        if block_idx not in self.blocks:
            return

        state = self.blocks[block_idx]
        if matrix.shape[0] == self.block_size:
            self.blocks[block_idx] = matrix @ state
        else:
            raise ValueError("Matrix size doesn't match block size")

    def apply_block_diagonal_unitary(
        self, block_unitaries: dict[int, np.ndarray]
    ) -> None:
        for block_idx, unitary in block_unitaries.items():
            if block_idx in self.blocks:
                self.blocks[block_idx] = unitary @ self.blocks[block_idx]

    def apply_block_mixing_operation(self, mixing_matrix: np.ndarray) -> None:
        if mixing_matrix.shape[0] != self.n_blocks:
            raise ValueError("Mixing matrix size doesn't match number of blocks")

        old_amplitudes = dict(self.block_amplitudes)
        new_amplitudes = {}

        for i in range(self.n_blocks):
            amp = 0.0j
            for j in range(self.n_blocks):
                if j in old_amplitudes:
                    amp += mixing_matrix[i, j] * old_amplitudes[j]
            if abs(amp) > 1e-15:
                new_amplitudes[i] = amp

        self.block_amplitudes = new_amplitudes

        for block_idx in new_amplitudes:
            if block_idx not in self.blocks:
                state = np.zeros(self.block_size, dtype=self._dtype)
                state[0] = 1.0
                self.blocks[block_idx] = state

    def get_amplitude(self, basis_state: int) -> complex:
        control_state = basis_state >> self.target_qubits
        target_state = basis_state & (self.block_size - 1)

        if control_state not in self.blocks:
            return 0.0j

        block_amp = self.block_amplitudes.get(control_state, 0.0j)
        state_amp = self.blocks[control_state][target_state]

        return block_amp * state_amp

    def sample(self, shots: int) -> dict[str, int]:
        rng = np.random.default_rng()
        results = []

        block_probs = {}
        total_prob = 0.0
        for block_idx, amp in self.block_amplitudes.items():
            if block_idx in self.blocks:
                block_norm = np.sum(np.abs(self.blocks[block_idx]) ** 2)
                prob = (np.abs(amp) ** 2) * block_norm
                block_probs[block_idx] = prob
                total_prob += prob

        if total_prob > 0:
            for k in block_probs:
                block_probs[k] /= total_prob

        block_indices = list(block_probs.keys())
        block_prob_values = [block_probs[k] for k in block_indices]

        for _ in range(shots):
            block_idx = rng.choice(block_indices, p=block_prob_values)

            state = self.blocks[block_idx]
            state_probs = np.abs(state) ** 2
            state_probs /= np.sum(state_probs)
            target_state = rng.choice(self.block_size, p=state_probs)

            full_state = (block_idx << self.target_qubits) | target_state
            results.append(format(full_state, f"0{self.n_qubits}b"))

        return dict(Counter(results))

    def get_active_blocks(self) -> list[int]:
        return [
            idx for idx, amp in self.block_amplitudes.items()
            if abs(amp) > 1e-15
        ]

    def prune_negligible_blocks(self, threshold: float = 1e-10) -> None:
        to_remove = []
        for block_idx, amp in self.block_amplitudes.items():
            if abs(amp) < threshold:
                to_remove.append(block_idx)

        for block_idx in to_remove:
            del self.block_amplitudes[block_idx]
            if block_idx in self.blocks:
                del self.blocks[block_idx]

        total_prob = sum(
            np.abs(amp) ** 2 * np.sum(np.abs(self.blocks.get(idx, [0])) ** 2)
            for idx, amp in self.block_amplitudes.items()
        )
        if total_prob > 0:
            norm_factor = 1.0 / np.sqrt(total_prob)
            for idx in self.block_amplitudes:
                self.block_amplitudes[idx] *= norm_factor

    def get_state_vector(self) -> np.ndarray:
        full_state = np.zeros(2 ** self.n_qubits, dtype=self._dtype)

        for block_idx, amp in self.block_amplitudes.items():
            if block_idx in self.blocks:
                start_idx = block_idx << self.target_qubits
                end_idx = start_idx + self.block_size
                full_state[start_idx:end_idx] = amp * self.blocks[block_idx]

        return full_state
