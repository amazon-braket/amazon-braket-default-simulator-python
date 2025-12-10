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
mps_simulator.py

Matrix Product State (MPS) simulator for low-entanglement quantum circuits.

This module provides an MPS-based simulator that efficiently handles circuits
with limited entanglement by representing the quantum state as a tensor train.

Mathematical Basis:
- MPS representation: |ψ⟩ = Σ A[0]^{i_0} A[1]^{i_1} ... A[n-1]^{i_{n-1}} |i_0...i_{n-1}⟩
- SVD-based truncation for bond dimension control
- Canonical forms for numerical stability

Complexity:
- Space: O(n * χ^2 * d) where χ is bond dimension, d=2 is physical dimension
- Time: O(n * χ^3) per two-qubit gate

Dependencies:
- numpy
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from braket.default_simulator.operation import GateOperation


class MPSTensor:
    """
    Single tensor in an MPS representation.

    Shape convention: (left_bond, physical_dim, right_bond)
    """

    def __init__(self, data: np.ndarray):
        if data.ndim != 3:
            raise ValueError(f"MPSTensor requires 3D array, got {data.ndim}D")
        self.data = data

    @property
    def left_bond(self) -> int:
        return self.data.shape[0]

    @property
    def physical_dim(self) -> int:
        return self.data.shape[1]

    @property
    def right_bond(self) -> int:
        return self.data.shape[2]

    def left_canonicalize(self) -> tuple[MPSTensor, np.ndarray]:
        shape = self.data.shape
        matrix = self.data.reshape(shape[0] * shape[1], shape[2])
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        new_tensor = MPSTensor(u.reshape(shape[0], shape[1], -1))
        remainder = np.diag(s) @ vh
        return new_tensor, remainder

    def right_canonicalize(self) -> tuple[np.ndarray, MPSTensor]:
        shape = self.data.shape
        matrix = self.data.reshape(shape[0], shape[1] * shape[2])
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        remainder = u @ np.diag(s)
        new_tensor = MPSTensor(vh.reshape(-1, shape[1], shape[2]))
        return remainder, new_tensor

    def copy(self) -> MPSTensor:
        return MPSTensor(self.data.copy())


class MPSSimulator:
    """
    Matrix Product State simulator for quantum circuits.
    """

    def __init__(
        self,
        n_qubits: int,
        max_bond_dim: int = 64,
        svd_cutoff: float = 1e-10,
        dtype=np.complex128
    ):
        self.n_qubits = n_qubits
        self.max_bond_dim = max_bond_dim
        self.svd_cutoff = svd_cutoff
        self.dtype = dtype
        self.tensors: list[MPSTensor] = []
        self.truncation_error: float = 0.0
        self._initialize_product_state()

    def _initialize_product_state(self) -> None:
        self.tensors = []
        for i in range(self.n_qubits):
            data = np.zeros((1, 2, 1), dtype=self.dtype)
            data[0, 0, 0] = 1.0
            self.tensors.append(MPSTensor(data))
        self.truncation_error = 0.0

    def initialize(self, basis_state: int) -> None:
        self.tensors = []
        for i in range(self.n_qubits):
            bit = (basis_state >> (self.n_qubits - 1 - i)) & 1
            data = np.zeros((1, 2, 1), dtype=self.dtype)
            data[0, bit, 0] = 1.0
            self.tensors.append(MPSTensor(data))
        self.truncation_error = 0.0

    def apply_single_qubit_gate(self, gate: np.ndarray, qubit: int) -> None:
        if gate.shape != (2, 2):
            raise ValueError("Single-qubit gate must be 2x2")
        if qubit < 0 or qubit >= self.n_qubits:
            raise ValueError(f"Qubit {qubit} out of range")

        tensor = self.tensors[qubit]
        new_data = np.einsum("ij,ljr->lir", gate, tensor.data)
        self.tensors[qubit] = MPSTensor(new_data)

    def apply_two_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int) -> None:
        if gate.shape != (4, 4):
            raise ValueError("Two-qubit gate must be 4x4")

        if abs(qubit1 - qubit2) == 1:
            if qubit1 < qubit2:
                self._apply_two_qubit_gate_adjacent(gate, qubit1, qubit2)
            else:
                gate_swapped = self._swap_gate_qubits(gate)
                self._apply_two_qubit_gate_adjacent(gate_swapped, qubit2, qubit1)
        else:
            self._apply_two_qubit_gate_non_adjacent(gate, qubit1, qubit2)

    def _swap_gate_qubits(self, gate: np.ndarray) -> np.ndarray:
        gate_tensor = gate.reshape(2, 2, 2, 2)
        swapped = np.transpose(gate_tensor, (1, 0, 3, 2))
        return swapped.reshape(4, 4)

    def _apply_two_qubit_gate_adjacent(
        self, gate: np.ndarray, qubit1: int, qubit2: int
    ) -> None:
        t1 = self.tensors[qubit1]
        t2 = self.tensors[qubit2]

        theta = np.einsum("lir,rjs->lijs", t1.data, t2.data)

        gate_tensor = gate.reshape(2, 2, 2, 2)
        theta_new = np.einsum("abij,lijs->labs", gate_tensor, theta)

        left_bond = theta_new.shape[0]
        right_bond = theta_new.shape[3]
        matrix = theta_new.reshape(left_bond * 2, 2 * right_bond)

        u, s, vh = np.linalg.svd(matrix, full_matrices=False)

        mask = s > self.svd_cutoff
        if self.max_bond_dim is not None:
            mask[self.max_bond_dim:] = False

        if np.sum(mask) == 0:
            mask[0] = True

        truncated_error = np.sum(s[~mask] ** 2)
        self.truncation_error += truncated_error

        s_trunc = s[mask]
        u_trunc = u[:, mask]
        vh_trunc = vh[mask, :]

        new_bond = len(s_trunc)
        u_s = u_trunc @ np.diag(s_trunc)

        new_t1_data = u_s.reshape(left_bond, 2, new_bond)
        new_t2_data = vh_trunc.reshape(new_bond, 2, right_bond)

        self.tensors[qubit1] = MPSTensor(new_t1_data)
        self.tensors[qubit2] = MPSTensor(new_t2_data)

    def _apply_two_qubit_gate_non_adjacent(
        self, gate: np.ndarray, qubit1: int, qubit2: int
    ) -> None:
        if qubit1 > qubit2:
            qubit1, qubit2 = qubit2, qubit1
            gate = self._swap_gate_qubits(gate)

        swap_gate = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=self.dtype)

        for i in range(qubit1, qubit2 - 1):
            self._apply_two_qubit_gate_adjacent(swap_gate, i, i + 1)

        self._apply_two_qubit_gate_adjacent(gate, qubit2 - 1, qubit2)

        for i in range(qubit2 - 2, qubit1 - 1, -1):
            self._apply_two_qubit_gate_adjacent(swap_gate, i, i + 1)

    def apply_operations(self, operations: list[GateOperation]) -> None:
        for op in operations:
            n_targets = len(op.targets)
            if n_targets == 1:
                self.apply_single_qubit_gate(op.matrix, op.targets[0])
            elif n_targets == 2:
                self.apply_two_qubit_gate(op.matrix, op.targets[0], op.targets[1])
            else:
                raise ValueError(f"MPS simulator supports up to 2-qubit gates, got {n_targets}")

    def get_amplitude(self, basis_state: int) -> complex:
        bit = (basis_state >> (self.n_qubits - 1)) & 1
        result = self.tensors[0].data[0, bit, :]

        for i in range(1, self.n_qubits):
            bit = (basis_state >> (self.n_qubits - 1 - i)) & 1
            tensor_slice = self.tensors[i].data[:, bit, :]
            result = result @ tensor_slice

        return result[0]

    def sample(self, shots: int) -> dict[str, int]:
        rng = np.random.default_rng()
        results = []

        for _ in range(shots):
            outcome = ""
            left_vector = np.array([1.0], dtype=self.dtype)

            for i in range(self.n_qubits):
                tensor = self.tensors[i].data

                prob_0_vec = left_vector @ tensor[:, 0, :]
                prob_1_vec = left_vector @ tensor[:, 1, :]

                prob_0 = np.sum(np.abs(prob_0_vec) ** 2)
                prob_1 = np.sum(np.abs(prob_1_vec) ** 2)
                total = prob_0 + prob_1

                if total > 0:
                    prob_0 /= total

                if rng.random() < prob_0:
                    outcome += "0"
                    left_vector = prob_0_vec
                    if np.linalg.norm(left_vector) > 0:
                        left_vector /= np.linalg.norm(left_vector)
                else:
                    outcome += "1"
                    left_vector = prob_1_vec
                    if np.linalg.norm(left_vector) > 0:
                        left_vector /= np.linalg.norm(left_vector)

            results.append(outcome)

        return dict(Counter(results))

    def get_bond_dimensions(self) -> list[int]:
        if not self.tensors:
            return []
        return [self.tensors[i].right_bond for i in range(self.n_qubits - 1)]

    def get_truncation_error(self) -> float:
        return self.truncation_error

    def canonicalize(self, center: int) -> None:
        for i in range(center):
            tensor, remainder = self.tensors[i].left_canonicalize()
            self.tensors[i] = tensor
            next_data = np.einsum("ij,jkl->ikl", remainder, self.tensors[i + 1].data)
            self.tensors[i + 1] = MPSTensor(next_data)

        for i in range(self.n_qubits - 1, center, -1):
            remainder, tensor = self.tensors[i].right_canonicalize()
            self.tensors[i] = tensor
            prev_data = np.einsum("ijk,kl->ijl", self.tensors[i - 1].data, remainder)
            self.tensors[i - 1] = MPSTensor(prev_data)

    def compute_entanglement_entropy(self, cut: int) -> float:
        if cut < 0 or cut >= self.n_qubits - 1:
            raise ValueError(f"Cut position {cut} out of range")

        self.canonicalize(cut)

        tensor = self.tensors[cut]
        shape = tensor.data.shape
        matrix = tensor.data.reshape(shape[0] * shape[1], shape[2])

        _, s, _ = np.linalg.svd(matrix, full_matrices=False)

        s_normalized = s / np.linalg.norm(s)
        s_squared = s_normalized ** 2
        s_squared = s_squared[s_squared > 1e-15]

        entropy = -np.sum(s_squared * np.log2(s_squared))
        return float(entropy)

    def get_state_vector(self) -> np.ndarray:
        if self.n_qubits > 20:
            raise ValueError("State vector too large to compute explicitly")

        result = self.tensors[0].data.squeeze(axis=0)

        for i in range(1, self.n_qubits):
            tensor = self.tensors[i].data
            if i == self.n_qubits - 1:
                tensor = tensor.squeeze(axis=2)
            result = np.tensordot(result, tensor, axes=([-1], [0]))

        return result.flatten()

    def copy(self) -> MPSSimulator:
        new_sim = MPSSimulator(
            self.n_qubits,
            self.max_bond_dim,
            self.svd_cutoff,
            self.dtype
        )
        new_sim.tensors = [t.copy() for t in self.tensors]
        new_sim.truncation_error = self.truncation_error
        return new_sim
